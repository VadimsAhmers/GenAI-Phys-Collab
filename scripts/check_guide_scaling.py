"""Does the near-perfect reflection survive being rescaled into a WIDER guide?

The chain that leads here (notes/chiral-freq-2026-08-11.md): the eps=10 crescent
reaches |r_RR|^2 = 0.9977 -- loss 0.0023 -- but only at 6.4385 GHz, when the disk
fills ~83% of the 32 mm guide. Three attempts to bring that peak to 5.8 GHz
inside the SAME guide all failed (rescaling changes the fill fraction that
creates the peak; thickness barely moves f and wrecks the peak; eps moves f well
and wrecks the peak faster). What is left is to scale the guide too.

Scaling EVERYTHING -- guide and meta-atom together -- is an exact Maxwell
transformation at fixed, dispersionless eps: the solution is the same field, read
at a shifted frequency. So the physics answer is already known. What is NOT known
is whether Maksim's .mph SURVIVES a different `a_wg`: whether the ports, their
mode definitions and the mesh follow the parameter, or whether something in the
model is pinned to 32 mm. That is a property of the file, not of Maxwell, and it
has to be measured before we propose a wider guide to Maksim.

`a_wg` is a real parameter of the model, but no bounds/CLI machinery of ours
mentions it, and threading it through `CHIRAL_PARAMS` would change `PARAM_NAMES`
-- which is the CSV schema and the resume key of every sweep we have run. So this
script extends the solver's parameter map at RUNTIME instead, and leaves the
library and every existing CSV alone.

L_wg is deliberately NOT scaled: in a matched guide the port planes see the same
|r|, length only adds phase.

Usage (on the machine with COMSOL, from the repo root):
    uv run python scripts/check_guide_scaling.py            # ~8 solves, ~4 min
    uv run python scripts/check_guide_scaling.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.opt import ChiralSolver, ChiralReflectanceObjective  # noqa: E402

# The measured anchor: `legacy` x 0.90 in the stock 32 mm guide, whose spectrum is
# in runs/freq/freq_legacy_scale_2.csv.
A_WG_STOCK_MM = 32.0
ANCHOR = {"r_mm": 13.230, "h_mm": 17.100, "y_cut_mm": 4.860, "r_cut_mm": 12.600,
          "eps_r": 10.0}
ANCHOR_FREQ_GHZ = 6.4385   # its peak
ANCHOR_PEAK = 0.9977       # |r_RR|^2 there

TARGET_FREQ_GHZ = 5.8      # where we want that peak instead

_C_MM_GHZ = 299.792458


def cutoffs(a_wg_mm: float) -> tuple[float, float]:
    """(TE10, TE11) cutoffs of a square guide -- the single-mode window."""
    te10 = _C_MM_GHZ / (2 * a_wg_mm)
    return te10, te10 * 2**0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/chiral.mph")
    ap.add_argument("--span", type=float, default=0.15,
                    help="half-width in GHz of the sweep around 5.8 (default: 0.15)")
    ap.add_argument("--step", type=float, default=0.05, help="grid step in GHz")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the expected numbers, touch no COMSOL")
    args = ap.parse_args()

    scale = ANCHOR_FREQ_GHZ / TARGET_FREQ_GHZ
    a_wg_new = A_WG_STOCK_MM * scale
    scaled = {k: (v * scale if k != "eps_r" else v) for k, v in ANCHOR.items()}

    lo, hi = cutoffs(a_wg_new)
    print(f"scale factor {scale:.6f}  ->  guide {A_WG_STOCK_MM} -> {a_wg_new:.3f} mm")
    print("  " + ", ".join(f"{k}={v:.4f}" for k, v in scaled.items()))
    print(f"  single-mode window of the new guide: {lo:.3f} - {hi:.3f} GHz "
          f"({(hi - TARGET_FREQ_GHZ) * 1000:.0f} MHz of margin above 5.8)\n")

    freqs = [round(TARGET_FREQ_GHZ - args.span + i * args.step, 6)
             for i in range(int(round(2 * args.span / args.step)) + 1)]
    freqs = [f for f in freqs if f < hi * 0.995]
    print(f"plan: 1 control solve + {len(freqs)} frequencies "
          f"({freqs[0]:g}-{freqs[-1]:g} GHz)")
    print(f"expected: control reproduces {ANCHOR_PEAK:.4f}; the scaled geometry "
          f"peaks at {ANCHOR_PEAK:.4f} on {TARGET_FREQ_GHZ} GHz\n")
    if args.dry_run:
        return 0

    obj = ChiralReflectanceObjective(ChiralSolver(args.model))
    # Runtime extension -- see the module docstring. The solver's loop skips any
    # key absent from `params`, so every other script keeps its exact behaviour.
    ChiralSolver.PARAM_MAP = {**ChiralSolver.PARAM_MAP, "a_wg_mm": "a_wg"}
    ChiralSolver.PARAM_UNITS = {**ChiralSolver.PARAM_UNITS, "a_wg_mm": "mm"}

    # Control: the anchor with a_wg written EXPLICITLY at its stock 32 mm. If this
    # does not reproduce the number measured without ever setting a_wg, then the
    # parameter is not what we think it is and the test below means nothing.
    print("control: anchor geometry, a_wg set explicitly to 32 mm")
    control = obj.evaluate({**ANCHOR, "a_wg_mm": A_WG_STOCK_MM,
                            "freq_ghz": ANCHOR_FREQ_GHZ})
    got = control.aux["abs_r_RR_sq"]
    print(f"  {ANCHOR_FREQ_GHZ} GHz -> |r_RR|^2 = {got:.4f} "
          f"(measured without a_wg: {ANCHOR_PEAK:.4f})")
    if abs(got - ANCHOR_PEAK) > 5e-3:
        print("\nFAIL: writing a_wg=32 changed the answer. The model does not take "
              "this parameter the way we assume -- stop and inspect the .mph.")
        return 1
    print("  OK: setting a_wg at its stock value is a no-op.\n")

    print(f"test: everything scaled x{scale:.4f}, guide {a_wg_new:.3f} mm")
    spectrum = {}
    for f in freqs:
        res = obj.evaluate({**scaled, "a_wg_mm": a_wg_new, "freq_ghz": f})
        spectrum[f] = res.aux["abs_r_RR_sq"]
        flag = "  <- design point" if abs(f - TARGET_FREQ_GHZ) < 1e-9 else ""
        print(f"  {f:.3f} GHz -> |r_RR|^2 = {spectrum[f]:.4f}{flag}")

    at_target = spectrum.get(TARGET_FREQ_GHZ)
    peak_f = max(spectrum, key=spectrum.get)
    print(f"\npeak {spectrum[peak_f]:.4f} at {peak_f:.3f} GHz")
    if at_target is not None:
        print(f"at 5.800 GHz: {at_target:.4f}  (loss 1-|r|^2 = {1 - at_target:.4f})")
        if abs(at_target - ANCHOR_PEAK) < 0.01:
            print("\nPASS: scale invariance carries the 0.998 into the wider guide, "
                  "and the model handles a_wg correctly. Route A is real.")
        else:
            print("\nMISMATCH: exact invariance says these must agree. Something in "
                  "the .mph does not follow a_wg (ports, their mode settings, or a "
                  "hard-coded mesh size) -- inspect before quoting this to Maksim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
