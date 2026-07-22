"""1-D parameter sweep around a fixed chiral-mirror geometry (needs COMSOL).

Evaluates the oracle along ONE parameter while holding the others fixed, which
is the cheap way to ask "does this parameter want to go past its optimizer
bound?" without paying for a whole optimization run (~26 s per point).

Values outside the optimizer's `CHIRAL_PARAMS` bounds are allowed on purpose --
probing past a bound is the point. Geometry failures are scored as |r_RR|^2 = 0
by the objective, same as during a run.

Usage (on the machine with COMSOL, from the repo root):
    uv run python scripts/sweep_chiral.py
    uv run python scripts/sweep_chiral.py --param r_mm --values 12,13,14,14.7,15.5
"""
from __future__ import annotations

import argparse

from src.opt import ChiralSolver, ChiralReflectanceObjective

# Best point of run chiral_comsol_llm_s0_eps_20260721_185156 (balanced, h<=20,
# eps free): score 0.18249, |r_RR|^2 = 0.8175.
BEST = {
    "r_mm": 14.7,
    "h_mm": 19.0,
    "y_cut_mm": 5.4,
    "r_cut_mm": 14.0,
    "eps_r": 10.0,
}

# Default sweep: thickness, which ended the run pinned near its 20 mm bound.
DEFAULT_PARAM = "h_mm"
DEFAULT_VALUES = [10.0, 14.0, 17.0, 19.0, 22.0, 25.0, 28.0, 31.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--param", default=DEFAULT_PARAM, help="parameter to sweep")
    ap.add_argument("--values", help="comma-separated values (default: h_mm sweep)")
    ap.add_argument("--model", default="models/chiral.mph")
    args = ap.parse_args()

    if args.param not in BEST:
        raise SystemExit(f"unknown param {args.param!r}; expected one of {list(BEST)}")
    values = (
        [float(v) for v in args.values.split(",")] if args.values else DEFAULT_VALUES
    )

    obj = ChiralReflectanceObjective(ChiralSolver(args.model))

    base = ", ".join(f"{k}={v}" for k, v in BEST.items() if k != args.param)
    print(f"sweeping {args.param} with {base}\n")
    print(f"{args.param:>8}  {'|r_RR|^2':>9}  {'score':>7}")

    results = []
    for value in values:
        params = {**BEST, args.param: value}
        res = obj.evaluate(params)
        r2 = res.aux["abs_r_RR_sq"]
        flag = " (solve failed)" if res.aux.get("solve_failed") else ""
        print(f"{value:8.2f}  {r2:9.4f}  {res.score:7.4f}{flag}")
        results.append((value, res.score))

    best_value, best_score = min(results, key=lambda vs: vs[1])
    print(f"\nbest: {args.param}={best_value:g}  score={best_score:.4f}")


if __name__ == "__main__":
    main()
