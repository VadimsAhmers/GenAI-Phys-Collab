"""Run point-reflectance or common RR/LL bandwidth optimization.

The default point objective is 1-|r_RR|^2. --objective bandwidth maximizes the
common continuous RR/LL >= 0.8 band within 5.5–6 GHz, starting from Maksim's
October sample, with r<=13 mm, eps=6..11 and exploratory h=2..40 mm.

Runs are written to `runs/<run_name>/` at the repo root. When `--run-name` is not
given the name is timestamped (`chiral_<solver>_<proposer>_<YYYYmmdd_HHMMSS>`) so
successive runs never overwrite each other's steps.

Dry run (no COMSOL, no API key) -- exercises the whole loop with a synthetic oracle:

    python -m src.opt.run_chiral --solver fake --proposer random --max-iters 12

Real LLM-driven run against COMSOL (needs OPENROUTER_API_KEY and a COMSOL license):

    python -m src.opt.run_chiral --solver comsol --proposer llm --max-iters 500

Continue a run that stopped/crashed (re-uses saved steps, no re-evaluation of them):

    python -m src.opt.run_chiral --solver comsol --proposer llm \
        --run-name chiral_comsol_llm_20260702_101500 --resume --max-iters 500

Rebuild the outputs (summary + plots) of a crashed run from its saved steps only:

    python -m src.opt.run_chiral --run-name <name> --finalize
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_opt import Config, Optimizer
from src.opt.optimize import (
    CHIRAL_EPS_PARAM,
    CHIRAL_PARAMS,
    CHIRAL_REFERENCE,
    ChiralParametrization,
    ChiralReflectanceObjective,
)
from src.opt.solvers import ChiralSolver, FakeChiralSolver, FakeBandwidthSolver
from src.opt.bandwidth import (
    BandwidthSettings, ChiralBandwidthObjective, BandwidthReporter,
    MAKSIM_2026_BOUNDS, MAKSIM_2026_START,
)
from src.opt.bandwidth_reuse import BaselineSpectrum
from src.opt.bandwidth_search import BandwidthOptimizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BANDWIDTH_CODE_FILES = (
    "src/opt/bandwidth.py", "src/opt/spectrum.py", "src/opt/solvers.py",
    "src/opt/bandwidth_reuse.py", "src/opt/bandwidth_search.py",
    "src/opt/run_chiral.py", "src/opt/optimize.py",
    "llm_opt/loop.py", "llm_opt/llm.py", "llm_opt/config.py",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", default="fake", choices=["fake", "comsol"])
    p.add_argument("--proposer", default="random", choices=["random", "llm"])
    p.add_argument("--objective", default="point", choices=["point", "bandwidth"])
    p.add_argument("--bandwidth-step", type=float, default=0.01, metavar="GHZ",
                   help="Bandwidth grid spacing (default 10 MHz).")
    p.add_argument("--bandwidth-edge-tol", type=float, default=0.0025, metavar="GHZ",
                   help="Refine observed threshold brackets to this width (default 2.5 MHz).")
    p.add_argument("--reflection-level", type=float, default=0.8,
                   help="Absolute power-reflection threshold for BOTH channels in bandwidth mode.")
    p.add_argument("--baseline-from", type=Path, metavar="RUN_DIR",
                   help="Reuse verified starting-geometry frequency points from a bandwidth run; "
                        "recompute the score on this run's grid. Same solver/model required.")
    p.add_argument(
        "--both-polarizations", action="store_true",
        help="In point mode record LL diagnostically (score remains 1-RR). "
             "Bandwidth mode always uses both channels.",
    )
    p.add_argument(
        "--aggressiveness",
        default="balanced",
        choices=["minimal", "balanced", "maximal"],
        help="LLM proposal spread (maximal = bolder moves; helps escape plateaus)",
    )
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (random proposer AND random init). Vary it across a "
        "multistart sweep so each run explores a different basin.",
    )
    p.add_argument(
        "--random-init",
        action="store_true",
        help="Start step 0 from a random in-bounds point (seeded by --seed) "
        "instead of the fixed midpoint; use for multistart with random init.",
    )
    p.add_argument(
        "--optimize-eps",
        action="store_true",
        help="Add the dielectric permittivity eps_r as a 5th search variable "
        "(bounds 5..10), same objective. Without this the geometry is optimized "
        "at the model's fixed eps_r.",
    )
    p.add_argument(
        "--bound",
        action="append",
        nargs=3,
        metavar=("NAME", "LO", "HI"),
        default=[],
        help="Override a parameter's bounds, e.g. --bound r_mm 4 10. Repeatable. "
        "Applied on top of the defaults (and eps_r's default when --optimize-eps).",
    )
    p.add_argument(
        "--start-reference",
        action="store_true",
        help="Start step 0 from Maksim's reference structure (r=6, h=3.4, "
        "y_cut=3.25, r_cut=2.9, eps_r=80) clamped into bounds, instead of the "
        "midpoint. Mutually exclusive with --random-init.",
    )
    p.add_argument("--model", default=str(REPO_ROOT / "models" / "chiral.mph"))
    p.add_argument(
        "--max-consecutive-fails",
        type=int,
        default=10,
        help="Abort after this many consecutive COMSOL solve failures (0 = never); "
        "guards against a dead solver/license silently poisoning the run.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Run directory name under runs/. Default appends a timestamp so runs "
        "never overwrite each other; pass an explicit name to --resume/--finalize.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing run (by --run-name) from its last saved step.",
    )
    p.add_argument(
        "--finalize",
        action="store_true",
        help="Rebuild a run's outputs (summary/plots) from saved steps without "
        "re-evaluating -- use to recover a crashed run.",
    )
    args = p.parse_args()

    # Resolve a (possibly relative) --model against the launch cwd BEFORE chdir,
    # so a user-supplied relative path still points where they meant.
    model_path = str(Path(args.model).resolve())
    baseline_path = args.baseline_from.resolve() if args.baseline_from else None

    # Anchor to the repo root so runs/ and models/ resolve there regardless of the
    # directory the process was launched from (tmux, cron, a different shell, ...).
    os.chdir(REPO_ROOT)
    load_dotenv()

    if args.resume and args.finalize:
        raise SystemExit("--resume and --finalize are mutually exclusive.")
    if (args.resume or args.finalize) and not args.run_name:
        raise SystemExit(
            "--resume/--finalize require --run-name (the run to continue/rebuild)."
        )
    if args.start_reference and args.random_init:
        raise SystemExit("--start-reference and --random-init are mutually exclusive.")
    band_mode = args.objective == "bandwidth"
    if baseline_path and not args.finalize and (not band_mode or args.random_init):
        raise SystemExit("--baseline-from requires bandwidth mode and the fixed October start.")
    if band_mode and args.start_reference:
        raise SystemExit("Bandwidth mode starts from the October sample; omit --start-reference.")
    if args.max_iters < 1 or args.max_consecutive_fails < 0:
        raise SystemExit("Need --max-iters >= 1 and --max-consecutive-fails >= 0.")
    settings = BandwidthSettings(
        step_ghz=args.bandwidth_step, edge_tolerance_ghz=args.bandwidth_edge_tol,
        level=args.reflection_level,
    ) if band_mode else None
    if band_mode:
        args.optimize_eps = True

    # Build the bounds dict: defaults (+ eps default when optimizing eps), then
    # apply any --bound overrides. Passing bounds explicitly bypasses the class's
    # internal default merge, so eps must be folded in here when it is a variable.
    bounds = {k: (lo, hi) for k, (lo, hi, _d) in CHIRAL_PARAMS.items()}
    if args.optimize_eps:
        bounds.update({k: (lo, hi) for k, (lo, hi, _d) in CHIRAL_EPS_PARAM.items()})
    if band_mode:
        bounds = dict(MAKSIM_2026_BOUNDS)
    for name, lo_s, hi_s in args.bound:
        if name not in bounds:
            raise SystemExit(
                f"--bound {name}: unknown parameter (have {list(bounds)}); "
                f"note eps_r needs --optimize-eps to be a variable."
            )
        lo, hi = float(lo_s), float(hi_s)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise SystemExit(f"--bound {name}: LO ({lo}) must be < HI ({hi}).")
        bounds[name] = (lo, hi)
    if band_mode:
        if bounds["r_mm"][1] > 13 or bounds["eps_r"][0] < 6 or bounds["eps_r"][1] > 11:
            raise SystemExit("Bandwidth constraints: radius <=13 mm, eps_r within [6,11].")
        if (any(bounds[k][0] <= 0 for k in ("r_mm", "h_mm"))
                or any(bounds[k][0] < 0 for k in ("y_cut_mm", "r_cut_mm"))):
            raise SystemExit("Lengths must be positive (cut radius/offset may be zero).")
        if not args.random_init and any(
            not lo <= MAKSIM_2026_START[k] <= hi for k, (lo, hi) in bounds.items()
        ):
            raise SystemExit("Bounds exclude the October start; no silent clipping. "
                             "Include it or explicitly use --random-init.")

    # Explicit name is used as-is (needed to resume/finalize a specific run);
    # otherwise timestamp it so each fresh run gets its own directory.
    if args.run_name:
        run_name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eps_tag = "_eps" if args.optimize_eps else ""
        objective_tag = "_bandwidth" if band_mode else ""
        run_name = f"chiral_{args.solver}_{args.proposer}_s{args.seed}{eps_tag}{objective_tag}_{stamp}"

    run_dir = REPO_ROOT / "runs" / run_name
    reusing = args.resume or args.finalize
    existing_steps = (
        sorted((run_dir / "steps").glob("step_*.json")) if run_dir.exists() else []
    )
    saved_config = Config.from_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else None
    saved_task = saved_config.extra.get("chiral", {}) if saved_config else {}
    if reusing and not existing_steps and not (args.resume and saved_task.get("objective") == "bandwidth"):
        raise SystemExit(
            f"--resume/--finalize needs an existing run with saved steps; "
            f"none found at {run_dir}"
        )
    if not reusing and (existing_steps or
                       (saved_config and (band_mode or saved_task.get("objective") == "bandwidth"))):
        raise SystemExit(
            f"run '{run_name}' already has {len(existing_steps)} saved steps at "
            f"{run_dir}.\nRefusing to overwrite. Use --resume to continue it, "
            f"--finalize to rebuild its outputs, or pass a different --run-name."
        )

    # Finalization reads recorded settings and never starts COMSOL.
    if args.finalize and saved_task.get("objective") == "bandwidth":
        band_mode = True
        bounds = saved_task["bounds"]
        args.optimize_eps = True
        settings = BandwidthSettings(**saved_task["settings"])

    manifest = {
        "objective": args.objective,
        "bounds": {k: list(v) for k, v in bounds.items()},
        "solver": args.solver, "seed": args.seed, "random_init": args.random_init,
        "proposer": args.proposer, "aggressiveness": args.aggressiveness,
        "start_reference": args.start_reference,
        "both_polarizations": args.both_polarizations or band_mode,
    }
    if band_mode and not args.finalize:
        manifest.update({
            "settings": asdict(settings), "start": MAKSIM_2026_START,
            "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            if args.solver == "comsol" else None,
            "code_sha256": {
                name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                for name in BANDWIDTH_CODE_FILES
            },
            "spectral_history": "best_and_recent_5",
        })
    baseline = None
    if baseline_path and not args.finalize:
        if baseline_path == run_dir.resolve():
            raise SystemExit("Baseline source must be a different run directory.")
        try:
            baseline = BaselineSpectrum(baseline_path, manifest, MAKSIM_2026_START)
        except (ValueError, OSError, KeyError) as exc:
            raise SystemExit(f"Cannot reuse baseline: {exc}") from exc
        manifest["baseline_reuse"] = baseline.manifest
    if args.resume and (band_mode or saved_task.get("objective") == "bandwidth"):
        if manifest != saved_task:
            raise SystemExit("Resume settings/model/code differ from saved bandwidth run. "
                             "Use the original options or a new --run-name.")

    solver = None if args.finalize else (
        (FakeBandwidthSolver() if band_mode else FakeChiralSolver())
        if args.solver == "fake" else ChiralSolver(model_path)
    )
    param = ChiralParametrization(
        bounds=bounds,
        seed=args.seed,
        random_init=args.random_init,
        optimize_eps=args.optimize_eps,
        start_params=(MAKSIM_2026_START if band_mode and not args.random_init
                      else CHIRAL_REFERENCE if args.start_reference else None),
        frequency_window=(settings.f_min, settings.f_max) if band_mode else None,
    )
    objective = ChiralBandwidthObjective(
        solver, settings=settings, cache_dir=run_dir / "spectra",
        max_consecutive_failures=args.max_consecutive_fails,
        baseline=baseline,
    ) if band_mode else ChiralReflectanceObjective(
        solver=solver, max_consecutive_failures=args.max_consecutive_fails,
        both_polarizations=args.both_polarizations,
    )

    config = Config(run_name=run_name, proposer=args.proposer)
    config.seed = args.seed
    config.llm.aggressiveness = args.aggressiveness
    if band_mode:
        config.llm.recent_n = 5
    config.stopping.max_iters = args.max_iters
    config.stopping.score_threshold = 0.0 if band_mode else 0.01
    config.extra["chiral"] = manifest
    if args.finalize and saved_config:
        config = saved_config

    optimizer_class = BandwidthOptimizer if band_mode else Optimizer
    opt = optimizer_class(config, param, objective,
                          reporter=BandwidthReporter() if band_mode else None)
    print(
        f"task=chiral solver={args.solver} proposer={args.proposer} run={run_name}"
        + (" [finalize]" if args.finalize else " [resume]" if args.resume else "")
    )
    if band_mode and not args.finalize:
        print(f"Common RR/LL >= {settings.level:g}; {len(settings.grid())} base frequencies "
              f"+ threshold refinement. Bounds: {json.dumps(bounds)}", flush=True)
        if baseline:
            print(f"Baseline cache: {baseline.root} ({len(baseline.points)} available points; "
                  "only requested frequencies will be reused).", flush=True)
    try:
        summary = opt.finalize_from_disk() if args.finalize else opt.run(resume=args.resume)
    finally:
        if isinstance(solver, ChiralSolver):
            solver.close()
    return summary


if __name__ == "__main__":
    main()
