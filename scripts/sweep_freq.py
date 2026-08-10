"""Frequency sweeps of the chiral-mirror oracle: |r_RR|^2 vs f (needs COMSOL).

Where `sweep_chiral.py` moves the GEOMETRY at the model's fixed 5.800 GHz, this
moves the FREQUENCY at a fixed geometry -- the measurement the band pivot rests
on. Maksim T. (2026-07-30) proposed optimizing over 5.5-6 GHz instead of the
single design point; before paying ~5x per evaluation for that, three things
should be measured rather than assumed:

  1. the true lineshape at his reference (we only ever INFERRED ~150 MHz from an
     eps sweep) -- and whether his reference actually peaks at 5.8 GHz;
  2. whether f_res really scales as 1/size with the waveguide held fixed at
     32x32 mm, which is what makes "found at 6.4 GHz -> rescale to 5.8" legal;
  3. how the peak HEIGHT moves with detuning/size, i.e. whether a band search
     can find taller peaks that the single-frequency objective reads as zero.

The frequency is applied through `ChiralSolver.set_frequency` (the model's study
step, not a global parameter); nothing is written back to the .mph.

Rows are appended to the CSV as they are computed and re-runs skip what is
already there, so an interrupted overnight sweep resumes with the same command.

Usage (on the machine with COMSOL, from the repo root):
    # 0. sanity check first -- 3 solves, ~2 min: does frequency reach the solve?
    uv run python scripts/sweep_freq.py --check

    # 1. lineshape at Maksim's reference
    uv run python scripts/sweep_freq.py --band 5.4:6.2 --step 0.025 \
        --out ~/Downloads/freq_reference.csv

    # 2. does f_res scale as 1/a? (all four lengths scaled together)
    uv run python scripts/sweep_freq.py --scales 0.85,0.9,0.95,1.05,1.1,1.15 \
        --band 4.9:7.0 --step 0.1 --refine 4 --out ~/Downloads/freq_scale.csv

    # 3. is f_res smooth in a SINGLE parameter (what the optimizer actually moves)?
    uv run python scripts/sweep_freq.py --vary r_mm --values 5.4,6.6 \
        --band 4.9:7.0 --step 0.1 --refine 4 --out ~/Downloads/freq_r.csv

    # 4. is the converged eps<=10 crescent on resonance, or just mistuned?
    uv run python scripts/sweep_freq.py --base legacy --band 4.5:7.0 --step 0.05 \
        --out ~/Downloads/freq_legacy.csv

    # print the plan (and the cost) without touching COMSOL
    uv run python scripts/sweep_freq.py --scales 0.9,1.1 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Run as `python scripts/sweep_freq.py` and neither the repo root (for `src`) nor
# this directory (for `sweep_chiral`) is on the path -- only the script's own
# directory is. Add both, so the documented command works without PYTHONPATH.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

# parse_base/PARAM_NAMES live in the geometry sweep; reuse them so both scripts
# accept exactly the same --base spellings (named point, params.json, k=v pairs).
from sweep_chiral import NAMED_BASES, PARAM_NAMES, parse_base  # noqa: E402

from src.opt import (  # noqa: E402
    ChiralSolver,
    FakeChiralSolver,
    ChiralReflectanceObjective,
    CHIRAL_FREQ_GHZ,
    CHIRAL_REFERENCE,
    cut_engulfs_disk,
    spectrum_summary,
    format_summary,
)

SECONDS_PER_SOLVE = 26.0

# The 32x32 mm square guide only carries the two degenerate fundamental modes
# between these cutoffs (c/2a and c*sqrt(2)/2a). Outside them r_RR stops meaning
# what we think it means: ABOVE 6.625 GHz the TE11/TM11 pair propagates too, so a
# 2-mode-per-port S-matrix no longer accounts for all the power, and just ABOVE
# 4.684 GHz the fundamental is barely propagating, which drives |r| towards 1 for
# any structure at all -- a spurious "resonance" at the band edge. The margins
# keep a few percent away from both.
A_WG_MM = 32.0
_C_MM_GHZ = 299.792458  # speed of light in mm*GHz
F_CUTOFF_TE10 = _C_MM_GHZ / (2 * A_WG_MM)          # 4.684 GHz
F_CUTOFF_TE11 = _C_MM_GHZ * 2**0.5 / (2 * A_WG_MM)  # 6.625 GHz
USABLE_BAND = (F_CUTOFF_TE10 * 1.04, F_CUTOFF_TE11 * 0.99)  # ~4.87 .. 6.56 GHz

# Lengths that scale together under --scales. eps_r is deliberately NOT scaled:
# Maxwell scale invariance moves resonances as 1/a at FIXED material.
LENGTH_KEYS = ("r_mm", "h_mm", "y_cut_mm", "r_cut_mm")

# Frequencies are matched on this tolerance when resuming (1 kHz).
FREQ_TOL = 1e-6

FIELDS = [
    "geom", "scale", "r_mm", "h_mm", "y_cut_mm", "r_cut_mm", "eps_r",
    "freq_ghz", "abs_r_RR_sq", "score", "stage", "status",
]


def frange(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive grid from lo to hi, rounded to kHz to keep keys exact."""
    if step <= 0:
        raise SystemExit("--step must be positive")
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


def build_geometries(base: dict, args) -> list[tuple[str, float | str, dict]]:
    """(label, scale, params) triples to sweep, in the order they are solved."""
    if args.scales and args.vary:
        raise SystemExit("--scales and --vary are alternatives, not a combination")

    if args.scales:
        scales = [float(s) for s in args.scales.split(",")]
        if 1.0 not in scales and not args.no_anchor:
            scales = sorted(scales + [1.0])  # the unscaled point anchors the fit
        out = []
        for a in scales:
            params = {**base, **{k: round(base[k] * a, 4) for k in LENGTH_KEYS}}
            out.append((f"scale={a:g}", a, params))
        return out

    if args.vary:
        if args.vary not in PARAM_NAMES:
            raise SystemExit(f"unknown --vary {args.vary!r}; expected one of {PARAM_NAMES}")
        if not args.values:
            raise SystemExit("--vary needs --values")
        out = []
        for value in (float(v) for v in args.values.split(",")):
            out.append((f"{args.vary}={value:g}", "", {**base, args.vary: value}))
        return out

    return [("base", 1.0, dict(base))]


def in_usable_band(freqs: list[float], label: str, allow_multimode: bool) -> list[float]:
    """Drop frequencies where |r_RR|^2 is not a trustworthy number (see USABLE_BAND)."""
    if allow_multimode:
        return freqs
    lo, hi = USABLE_BAND
    kept = [f for f in freqs if lo <= f <= hi]
    dropped = len(freqs) - len(kept)
    if dropped:
        print(f"  {label}: dropped {dropped} of {len(freqs)} frequencies outside the "
              f"guide's single-mode window {lo:.2f}-{hi:.2f} GHz "
              f"(--allow-multimode to keep them)")
    return kept


def geom_key(params: dict) -> tuple:
    """Resume identity of a geometry: its numbers, not its label.

    Keying on the label would be wrong across bases -- `--vary r_mm --values 14.7`
    means a different structure on `reference` than on `legacy`, yet both would
    be labelled `r_mm=14.7`, and appending both to one CSV would make the second
    silently inherit the first one's spectrum.
    """
    return tuple(round(float(params[k]), 6) for k in PARAM_NAMES)


def load_done(path: Path) -> dict[tuple, dict]:
    """What the CSV already holds, per geometry.

    {geom_key: {"values": {freq_ghz: |r_RR|^2}, "failed": bool}}. Resuming needs
    the VALUES, not just which points were done: the refinement grid and the
    printed summary are computed from the whole spectrum, so a resumed run must
    see the points from the earlier one.
    """
    done: dict[tuple, dict] = {}
    if not path or not path.exists():
        return done
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                key = geom_key(row)
                freq = round(float(row["freq_ghz"]), 6)
                value = float(row["abs_r_RR_sq"])
            except (KeyError, ValueError, TypeError):
                continue  # header/format from another script version
            entry = done.setdefault(key, {"values": {}, "failed": False})
            if row.get("status") == "solve_failed":
                entry["failed"] = True
            else:
                entry["values"][freq] = value
    return done


def open_csv(path: Path | None):
    """Append-mode writer that flushes every row (so a killed run keeps its data)."""
    if path is None:
        return None, None
    is_new = not path.exists() or path.stat().st_size == 0
    fh = path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def run_check(obj, model_freqs=(5.8, 5.6, 6.0)) -> int:
    """Verify the frequency actually reaches the solve before spending a night on it.

    The reference geometry at 5.8 GHz must reproduce the known 0.8851 baseline,
    and detuning by +-200 MHz must MOVE the number. If it does not move, the
    study step being rewritten is not the one that feeds the r_RR solutions and
    every spectrum would be a flat line.
    """
    print("frequency plumbing check (3 solves, ~1.5 min)\n")
    if isinstance(obj.solver, FakeChiralSolver):
        print("NOTE: --solver fake has no COMSOL baseline to reproduce, so this "
              "check is expected to fail; it only means the arguments parse.\n")
    values = {}
    for f in model_freqs:
        res = obj.evaluate({**CHIRAL_REFERENCE, "freq_ghz": f})
        values[f] = res.aux["abs_r_RR_sq"]
        print(f"  {f:5.2f} GHz -> |r_RR|^2 = {values[f]:.4f}")

    ok = True
    baseline = values.get(CHIRAL_FREQ_GHZ)
    if baseline is None:  # only reachable if model_freqs is overridden
        return 0
    if abs(baseline - 0.8851) > 5e-3:
        print(f"\nFAIL: 5.8 GHz gave {baseline:.4f}, expected the 0.8851 baseline.")
        ok = False
    detuned = [v for f, v in values.items() if f != CHIRAL_FREQ_GHZ]
    if detuned and all(abs(v - baseline) < 1e-6 for v in detuned):
        print("\nFAIL: detuning changed nothing -- the frequency is NOT reaching "
              "the solutions r_RR reads. Do not start the sweep.")
        ok = False
    if ok:
        print("\nOK: baseline reproduced and detuning moves |r_RR|^2. Sweep is safe to run.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--base", default="reference",
        help=f"base geometry: {list(NAMED_BASES)}, a path to params.json, or inline "
             f"'k=v,k=v' overrides on the reference (default: reference)")
    ap.add_argument(
        "--scales",
        help="comma-separated size factors applied to ALL lengths at once "
             "(eps_r untouched), e.g. '0.9,0.95,1.05,1.1'; 1.0 is always included")
    ap.add_argument(
        "--no-anchor", action="store_true",
        help="with --scales, do not add the unscaled (1.0) geometry")
    ap.add_argument("--vary", help=f"instead of --scales, move ONE parameter: {PARAM_NAMES}")
    ap.add_argument("--values", help="comma-separated values for --vary")
    ap.add_argument(
        "--window", type=float, metavar="GHZ",
        help="with --scales: instead of one band for everything, give each "
             "geometry its own +-GHZ window centred on the PREDICTED resonance "
             "f-ref/scale. A narrow line needs a fine grid, and a fine grid over "
             "a whole 2 GHz band is unaffordable; the prediction is what makes a "
             "narrow window possible (and testing it is the point of the sweep).")
    ap.add_argument(
        "--f-ref", type=float, default=CHIRAL_FREQ_GHZ,
        help=f"resonance of the unscaled geometry, for --window "
             f"(default: {CHIRAL_FREQ_GHZ})")
    ap.add_argument(
        "--allow-multimode", action="store_true",
        help=f"keep frequencies outside the guide's single-mode window "
             f"({USABLE_BAND[0]:.2f}-{USABLE_BAND[1]:.2f} GHz), where |r_RR|^2 "
             f"is not a trustworthy number (default: drop them)")
    ap.add_argument(
        "--band", default="5.4:6.2",
        help="frequency band in GHz as LO:HI (default: 5.4:6.2)")
    ap.add_argument("--step", type=float, default=0.05, help="grid step in GHz (default: 0.05)")
    ap.add_argument(
        "--freqs", help="explicit comma-separated frequencies in GHz (overrides --band/--step)")
    ap.add_argument(
        "--refine", type=int, default=0, metavar="N",
        help="after the coarse scan, re-scan around its maximum at step/N "
             "(0 = off). Cheaper than a fine grid over the whole band.")
    ap.add_argument("--model", default="models/chiral.mph")
    ap.add_argument(
        "--solver", default="comsol", choices=["comsol", "fake"],
        help="'fake' is a synthetic Lorentzian for checking the plumbing (never results)")
    ap.add_argument("--out", help="append results to this CSV (resumable)")
    ap.add_argument(
        "--check", action="store_true",
        help="only verify that the frequency reaches the solve, then exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    solver = FakeChiralSolver() if args.solver == "fake" else ChiralSolver(args.model)
    obj = ChiralReflectanceObjective(solver)

    if args.check:
        return run_check(obj)

    base = parse_base(args.base)
    geometries = build_geometries(base, args)

    if args.freqs:
        if args.refine:
            raise SystemExit(
                "--refine works off the --step grid; with an explicit --freqs list "
                "there is no step to refine against. Use --band/--step, or list the "
                "fine frequencies in --freqs yourself.")
        freqs = sorted(round(float(f), 6) for f in args.freqs.split(","))
    else:
        try:
            lo, hi = (float(x) for x in args.band.split(":"))
        except ValueError:
            raise SystemExit(f"--band {args.band!r}: expected LO:HI in GHz, e.g. 5.4:6.2")
        freqs = frange(lo, hi, args.step)

    if args.window and not args.scales:
        raise SystemExit("--window needs --scales (it centres on f-ref/scale)")

    # One frequency list per geometry: with --window each is centred on that
    # geometry's predicted resonance, otherwise they all share the CLI band.
    plan = []
    for label, scale, params in geometries:
        if args.window:
            centre = args.f_ref / float(scale)
            grid = frange(centre - args.window, centre + args.window, args.step)
        else:
            grid = list(freqs)
        plan.append((label, scale, params, in_usable_band(grid, label,
                                                          args.allow_multimode)))

    out_path = Path(args.out).expanduser() if args.out else None
    done = load_done(out_path)

    n_planned = sum(len(g) for _, _, _, g in plan)
    n_todo = sum(1 for _, _, params, grid in plan
                 for f in grid if not _is_done(done, geom_key(params), f))
    print("base: " + ", ".join(f"{k}={base[k]:g}" for k in PARAM_NAMES))
    if args.window:
        print(f"{len(plan)} geometries, each on its own +-{args.window * 1000:.0f} MHz "
              f"window around {args.f_ref:g}/scale")
    else:
        print(f"{len(plan)} geometries x {len(freqs)} frequencies "
              f"({freqs[0]:g}-{freqs[-1]:g} GHz)")
    if done:
        print(f"{n_planned - n_todo} points already in {out_path} -- resuming")
    if args.refine:
        print(f"plus up to ~{2 * args.refine} refinement points per geometry "
              f"at {args.step / args.refine * 1000:.0f} MHz")
    print(f"~{n_todo * SECONDS_PER_SOLVE / 3600:.1f} h of COMSOL at "
          f"~{SECONDS_PER_SOLVE:.0f} s/solve (before refinement)\n")
    if args.dry_run:
        for label, _, params, grid in plan:
            span = f"{grid[0]:g}-{grid[-1]:g} GHz, {len(grid)} pts" if grid else "NO POINTS"
            print(f"  {label:>16}: " +
                  ", ".join(f"{k}={params[k]:g}" for k in PARAM_NAMES) + f"  [{span}]")
        return 0

    fh, writer = open_csv(out_path)
    try:
        for label, scale, params, freqs in plan:
            print(f"=== {label}: " +
                  ", ".join(f"{k}={params[k]:g}" for k in PARAM_NAMES) + " ===")
            if cut_engulfs_disk(params):
                print("  skipped: the cut engulfs the disk (empty geometry)\n")
                continue
            if not freqs:
                print("  skipped: no frequencies left inside the single-mode window\n")
                continue
            key = geom_key(params)
            previous = done.get(key, {"values": {}, "failed": False})
            if previous["failed"]:
                print("  skipped: this geometry already failed to solve\n")
                continue

            spectrum: dict[float, float] = dict(previous["values"])
            failed = _scan(obj, params, freqs, label, scale, spectrum,
                           writer, fh, done, stage="coarse")
            if failed:
                print("  geometry failed to solve -- skipping its remaining "
                      "frequencies\n")
                continue

            # The refinement window is anchored to the COARSE grid, never to the
            # current best sample: anchoring it to the sample would move it once
            # the fine points land, so re-running a finished sweep would refine
            # around the new maximum and keep solving. Off the coarse grid it is
            # idempotent -- a resumed run adds exactly what an uninterrupted one
            # would have.
            coarse = {f: v for f, v in spectrum.items() if f in set(freqs)}
            if args.refine and coarse:
                summary = spectrum_summary(list(coarse), list(coarse.values()))
                if summary["at_edge"]:
                    print("  no refinement: the maximum is on the band edge\n")
                else:
                    fine_step = args.step / args.refine
                    peak = min(coarse, key=lambda f: abs(f - summary["peak_ghz"]))
                    fine = [f for f in frange(peak - args.step, peak + args.step, fine_step)
                            if all(abs(f - g) > FREQ_TOL for g in spectrum)]
                    _scan(obj, params, fine, label, scale, spectrum,
                          writer, fh, done, stage="fine")

            if spectrum:
                summary = spectrum_summary(list(spectrum), list(spectrum.values()))
                print(f"  -> {format_summary(summary)}\n")
    finally:
        if fh is not None:
            fh.close()

    if out_path:
        print(f"data appended to {out_path}")
    return 0


def _is_done(done: dict, key: tuple, freq: float) -> bool:
    entry = done.get(key)
    return bool(entry) and freq in entry["values"]


def _scan(obj, params, freqs, label, scale, spectrum, writer, fh, done, stage) -> bool:
    """Solve `params` at each frequency, recording as we go. True if the geometry failed."""
    key = geom_key(params)
    for f in sorted(freqs):
        if _is_done(done, key, f):
            continue
        res = obj.evaluate({**params, "freq_ghz": f})
        r2 = res.aux["abs_r_RR_sq"]
        failed = res.aux.get("solve_failed", False)
        status = "solve_failed" if failed else "ok"
        print(f"  {f:7.4f} GHz  |r_RR|^2 = {r2:.4f}" + ("  (SOLVE FAILED)" if failed else ""))
        if writer is not None:
            writer.writerow({
                "geom": label, "scale": scale,
                # .10g, not :g: the geometry is the resume key, and 6 significant
                # digits would round a params.json base away from its own rows.
                **{k: f"{params[k]:.10g}" for k in PARAM_NAMES},
                "freq_ghz": f"{f:.6f}", "abs_r_RR_sq": f"{r2:.6f}",
                "score": f"{res.score:.6f}", "stage": stage, "status": status,
            })
            fh.flush()
        if failed:
            return True
        spectrum[f] = r2
    return False


if __name__ == "__main__":
    raise SystemExit(main())
