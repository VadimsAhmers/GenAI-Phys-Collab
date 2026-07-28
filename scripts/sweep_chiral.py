"""1-D parameter sweeps around a fixed chiral-mirror geometry (needs COMSOL).

Evaluates the oracle along ONE parameter at a time while holding the others
fixed, which is the cheap, deterministic way to ask "what does the landscape
look like around this point?" without paying for a whole optimization run
(~26 s per point).

The motivating case (2026-07-28): the high-eps run
`chiral_comsol_llm_s0_eps_20260727_194518` never improved on its own starting
point (Maksim T.'s reference, score 0.1149) in 365 evaluations, and its
proposals drifted *away* over time. A sweep answers directly whether that point
sits on a narrow peak -- which the LLM cannot be expected to hill-climb -- or
whether some axis still wants to move.

Values outside the optimizer's `CHIRAL_PARAMS` bounds are allowed on purpose --
probing past a bound is the point. Geometries where the cut engulfs the disk are
skipped without solving (they only waste ~26 s); other geometry failures are
scored as |r_RR|^2 = 0 by the objective, same as during a run.

Usage (on the machine with COMSOL, from the repo root):
    # all five axes through Maksim's reference point -- the default
    uv run python scripts/sweep_chiral.py --out sweep_reference.csv

    # one axis, auto-generated grid around the base
    uv run python scripts/sweep_chiral.py --param eps_r

    # one axis, explicit values, around a different base
    uv run python scripts/sweep_chiral.py --base legacy --param h_mm \
        --values 10,14,17,19,22,25,28,31

    # base taken from a finished run's params.json
    uv run python scripts/sweep_chiral.py --base runs/<run>/params.json

    # print the plan without touching COMSOL
    uv run python scripts/sweep_chiral.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.opt import (
    ChiralSolver,
    FakeChiralSolver,
    ChiralReflectanceObjective,
    CHIRAL_PARAMS,
    CHIRAL_EPS_PARAM,
    CHIRAL_REFERENCE,
    cut_engulfs_disk,
)

# Upper bounds are only used to build a grid for a parameter whose base value is
# 0 (a multiplicative grid would collapse to all-zeros there).
_BOUNDS = {k: (lo, hi) for k, (lo, hi, _d) in {**CHIRAL_PARAMS, **CHIRAL_EPS_PARAM}.items()}
PARAM_NAMES = list(_BOUNDS)

# Best point of run chiral_comsol_llm_s0_eps_20260721_185156 (balanced, h<=20,
# eps free): score 0.18249, |r_RR|^2 = 0.8175. Kept as a named base so the
# earlier eps<=10 sweeps stay reproducible.
LEGACY_BEST = {
    "r_mm": 14.7,
    "h_mm": 19.0,
    "y_cut_mm": 5.4,
    "r_cut_mm": 14.0,
    "eps_r": 10.0,
}

NAMED_BASES = {"reference": CHIRAL_REFERENCE, "legacy": LEGACY_BEST}

# Multiplicative grid: dense near the base (to resolve a narrow peak) and a
# couple of coarse points either side (to see the wider landscape).
FACTORS = [0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5]

# Fractions of a parameter's upper bound, used when its base value is 0.
ZERO_BASE_FRACTIONS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]

# Grid values are rounded to 3 decimals, so a base carrying more decimals (e.g.
# read from a run's params.json) lands up to 5e-4 away from its own grid point --
# match on a tolerance rather than on float equality.
VALUE_TOL = 1e-3

# A win smaller than this is solver noise, not a real improvement; treat it as a
# tie and credit the base, so a flat axis doesn't read as "wants to move".
SCORE_TOL = 1e-6


def parse_base(spec: str) -> dict:
    """Resolve --base: a named point, a params.json path, or inline k=v pairs."""
    if spec in NAMED_BASES:
        return dict(NAMED_BASES[spec])

    path = Path(spec)
    if path.exists():
        data = json.loads(path.read_text())
        # A run's params.json is {"params": {...}, "score": ...}; a bare dict works too.
        params = data.get("params", data) if isinstance(data, dict) else data
        missing = [k for k in PARAM_NAMES if k not in params]
        if missing:
            raise SystemExit(f"{spec}: missing parameters {missing}")
        return {k: float(params[k]) for k in PARAM_NAMES}

    # Inline overrides on top of the reference point, e.g. "eps_r=60,h_mm=4".
    base = dict(CHIRAL_REFERENCE)
    for item in spec.split(","):
        if "=" not in item:
            raise SystemExit(
                f"--base {spec!r}: expected a name {list(NAMED_BASES)}, a path to "
                f"params.json, or inline 'k=v,k=v' pairs"
            )
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in PARAM_NAMES:
            raise SystemExit(f"unknown parameter {key!r}; expected one of {PARAM_NAMES}")
        base[key] = float(value)
    return base


def grid_for(param: str, base_value: float) -> list[float]:
    """Sweep values for one axis, always including the base value itself."""
    if abs(base_value) < 1e-9:
        hi = _BOUNDS[param][1]
        values = [f * hi for f in ZERO_BASE_FRACTIONS]
    else:
        values = [f * base_value for f in FACTORS]
    values.append(base_value)
    # Round before de-duplicating so near-identical points collapse to one solve.
    return sorted({round(v, 3) for v in values})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--param", default="all",
        help="parameter to sweep, or 'all' for every axis (default: all)")
    ap.add_argument(
        "--values",
        help="comma-separated absolute values (single --param only; "
             "default: an auto grid around the base)")
    ap.add_argument(
        "--base", default="reference",
        help=f"base point: {list(NAMED_BASES)}, a path to params.json, or inline "
             f"'k=v,k=v' overrides on the reference (default: reference)")
    ap.add_argument("--model", default="models/chiral.mph")
    ap.add_argument(
        "--solver", default="comsol", choices=["comsol", "fake"],
        help="'fake' uses the synthetic oracle to check the plumbing (never results)")
    ap.add_argument("--out", help="also write the results to this CSV path")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the planned points and exit without running COMSOL")
    args = ap.parse_args()

    base = parse_base(args.base)
    if args.param == "all":
        params_to_sweep = PARAM_NAMES
    elif args.param in PARAM_NAMES:
        params_to_sweep = [args.param]
    else:
        raise SystemExit(
            f"unknown param {args.param!r}; expected 'all' or one of {PARAM_NAMES}")

    if args.values and len(params_to_sweep) > 1:
        raise SystemExit("--values needs a single --param (not 'all')")
    explicit = [float(v) for v in args.values.split(",")] if args.values else None

    plan = {p: (explicit or grid_for(p, base[p])) for p in params_to_sweep}
    n_points = sum(len(v) for v in plan.values())

    print("base point: " + ", ".join(f"{k}={base[k]:g}" for k in PARAM_NAMES))
    print(f"sweeping {', '.join(params_to_sweep)} -- {n_points} points "
          f"(~{n_points * 26 / 60:.0f} min at ~26 s/solve, minus skips/repeats)\n")
    if args.dry_run:
        for param, values in plan.items():
            print(f"{param}: " + ", ".join(f"{v:g}" for v in values))
        return

    solver = FakeChiralSolver() if args.solver == "fake" else ChiralSolver(args.model)
    obj = ChiralReflectanceObjective(solver)

    # One cache across axes: the base point appears in every grid, so this pays
    # for it once instead of once per axis.
    cache: dict[tuple, tuple[float, float] | None] = {}

    def evaluate(params: dict) -> tuple[float, float] | None:
        """(|r_RR|^2, score), or None when the geometry is degenerate."""
        key = tuple(round(params[k], 6) for k in PARAM_NAMES)
        if key not in cache:
            if cut_engulfs_disk(params):
                cache[key] = None
            else:
                res = obj.evaluate(params)
                cache[key] = (res.aux["abs_r_RR_sq"], res.score)
        return cache[key]

    rows: list[dict] = []
    verdicts: list[tuple[str, float, float, float, float]] = []

    for param in params_to_sweep:
        held = ", ".join(f"{k}={base[k]:g}" for k in PARAM_NAMES if k != param)
        print(f"=== {param} (holding {held}) ===")
        print(f"{param:>10}  {'|r_RR|^2':>9}  {'score':>7}")

        scored: list[tuple[float, float]] = []
        for value in plan[param]:
            params = {**base, param: value}
            out = evaluate(params)
            is_base = abs(value - base[param]) < VALUE_TOL
            mark = "  <- base" if is_base else ""
            if out is None:
                print(f"{value:10.3f}  {'--':>9}  {'--':>7}  (skipped: cut engulfs disk)")
                rows.append({"param": param, "value": value, "is_base": is_base,
                             "abs_r_RR_sq": "", "score": "", "status": "engulfed"})
                continue
            r2, score = out
            print(f"{value:10.3f}  {r2:9.4f}  {score:7.4f}{mark}")
            rows.append({"param": param, "value": value, "is_base": is_base,
                         "abs_r_RR_sq": f"{r2:.6f}", "score": f"{score:.6f}",
                         "status": "ok"})
            scored.append((value, score))

        if not scored:
            print("  (no evaluable points on this axis)\n")
            continue
        # Break ties toward the base, so an axis that is flat (or wins only by
        # noise) reports the base as its minimum instead of an arbitrary endpoint.
        best_value, best_score = min(
            scored, key=lambda vs: (vs[1], abs(vs[0] - base[param])))
        base_hits = [s for v, s in scored if abs(v - base[param]) < VALUE_TOL]
        if not base_hits:
            print(f"  best: {param}={best_value:g}  score={best_score:.4f}\n")
            continue
        base_score = base_hits[0]
        if best_score >= base_score - SCORE_TOL:
            best_value, best_score = base[param], base_score
            print(f"  -> base IS the minimum on this axis (score={base_score:.4f})\n")
        else:
            print(f"  -> better at {param}={best_value:g}: score={best_score:.4f} "
                  f"vs base {base_score:.4f} (delta={best_score - base_score:+.4f})\n")
        verdicts.append((param, base[param], base_score, best_value, best_score))

    if len(verdicts) > 1:
        print("=== summary (lower score is better) ===")
        print(f"{'param':>10}  {'base':>8}  {'base score':>10}  {'best at':>8}  "
              f"{'best score':>10}  verdict")
        for param, bval, bscore, vval, vscore in verdicts:
            at_base = abs(vval - bval) < VALUE_TOL
            verdict = "base is min" if at_base else f"improves by {vscore - bscore:+.4f}"
            print(f"{param:>10}  {bval:8.3f}  {bscore:10.4f}  {vval:8.3f}  "
                  f"{vscore:10.4f}  {verdict}")
        n_improved = sum(1 for _, b, _, v, _ in verdicts if abs(v - b) > VALUE_TOL)
        if n_improved == 0:
            print("\nBase is a local minimum along every swept axis.")
        else:
            print(f"\n{n_improved} of {len(verdicts)} axes want to move off the base.")

    if args.out:
        out = Path(args.out)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["param", "value", "is_base", "abs_r_RR_sq",
                                "score", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
