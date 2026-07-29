"""Where in parameter space did a run's *good* points land?

Reads a finished run's `steps/step_*.json` and lists every evaluation above a
score threshold together with its distance to Maksim T.'s reference geometry,
so a cluster around the reference can be told apart from a separate resonance
elsewhere in the box.

Motivating question (2026-07-29): the high-eps run
`chiral_comsol_llm_s0_eps_20260727_194518` produced 7 points above half of the
peak |r_RR|^2 (best 0.814) yet never beat its own starting point. If those 7 sit
next to the reference, the region is explored and its local maximum is the
reference; if any sits far away, that is a *second* resonance worth a focused
local run -- the only cheap route to beating 0.885.

Distances are reported in two ways: raw per-parameter deltas, and a normalized
distance where each axis is scaled by its search range (so 1.0 means "a whole
box width away"). No COMSOL needed -- this only reads JSON.

Usage (on the machine holding the run):
    uv run python scripts/inspect_hits.py runs/chiral_comsol_llm_s0_eps_20260727_194518
    uv run python scripts/inspect_hits.py <run_dir> --threshold 0.443
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.opt import CHIRAL_PARAMS, CHIRAL_EPS_PARAM, CHIRAL_REFERENCE

# Search box of the high-eps run (compact caps, eps opened). Only used to
# normalize distances, so an approximate box is fine for other runs.
DEFAULT_BOX = {
    "r_mm": (4.0, 10.0),
    "h_mm": (2.0, 10.0),
    "y_cut_mm": (0.0, 17.5),
    "r_cut_mm": (0.0, 17.5),
    "eps_r": (10.0, 120.0),
}

PARAM_NAMES = list({**CHIRAL_PARAMS, **CHIRAL_EPS_PARAM})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run directory containing steps/step_*.json")
    ap.add_argument(
        "--threshold", type=float, default=0.443,
        help="report evaluations with |r_RR|^2 above this (default: half of the "
             "reference peak 0.8856)")
    args = ap.parse_args()

    steps = sorted(Path(args.run_dir).glob("steps/step_*.json"))
    if not steps:
        raise SystemExit(f"no steps/step_*.json under {args.run_dir}")

    hits = []
    for path in steps:
        data = json.loads(path.read_text())
        r2 = (data.get("aux") or {}).get("abs_r_RR_sq")
        if r2 is None or r2 < args.threshold:
            continue
        hits.append((data["step"], r2, data["params"]))

    print(f"{len(steps)} steps, {len(hits)} above |r_RR|^2 = {args.threshold}\n")
    ref = CHIRAL_REFERENCE
    print("reference: " + ", ".join(f"{k}={ref[k]:g}" for k in PARAM_NAMES) + "\n")

    for step, r2, params in sorted(hits, key=lambda h: -h[1]):
        # Normalized distance: each axis scaled by its range, so the number is
        # comparable across parameters with very different units.
        norm = math.sqrt(sum(
            ((params[k] - ref[k]) / (DEFAULT_BOX[k][1] - DEFAULT_BOX[k][0])) ** 2
            for k in PARAM_NAMES if k in params))
        deltas = ", ".join(
            f"{k}={params[k]:g} ({params[k] - ref[k]:+g})"
            for k in PARAM_NAMES if k in params)
        near = "NEAR reference" if norm < 0.1 else "far from reference"
        print(f"step {step:>3}  |r_RR|^2={r2:.4f}  dist={norm:.3f}  [{near}]")
        print(f"           {deltas}")

    if hits:
        print("\nRule of thumb: dist < 0.1 means the point sits inside ~10% of the "
              "box on every axis, i.e. the same resonance as the reference; a "
              "clearly larger distance means a separate one, which is worth its "
              "own focused local run.")


if __name__ == "__main__":
    main()
