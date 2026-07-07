"""Run the chiral-mirror optimizer (one-handed circular-polarization reflection).

The task is scalar: maximize |r_RR|^2 (score = 1 - |r_RR|^2 -> 0). Geometry is the
4 mm params of `ChiralParametrization`; the physics is COMSOL via `ChiralSolver`.

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
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_opt import Config, Optimizer
from src.opt.optimize import ChiralParametrization, ChiralReflectanceObjective
from src.opt.solvers import ChiralSolver, FakeChiralSolver

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", default="fake", choices=["fake", "comsol"])
    p.add_argument("--proposer", default="random", choices=["random", "llm"])
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

    # Explicit name is used as-is (needed to resume/finalize a specific run);
    # otherwise timestamp it so each fresh run gets its own directory.
    if args.run_name:
        run_name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"chiral_{args.solver}_{args.proposer}_s{args.seed}_{stamp}"

    run_dir = REPO_ROOT / "runs" / run_name
    reusing = args.resume or args.finalize
    existing_steps = (
        sorted((run_dir / "steps").glob("step_*.json")) if run_dir.exists() else []
    )
    if reusing and not existing_steps:
        raise SystemExit(
            f"--resume/--finalize needs an existing run with saved steps; "
            f"none found at {run_dir}"
        )
    if not reusing and existing_steps:
        raise SystemExit(
            f"run '{run_name}' already has {len(existing_steps)} saved steps at "
            f"{run_dir}.\nRefusing to overwrite. Use --resume to continue it, "
            f"--finalize to rebuild its outputs, or pass a different --run-name."
        )

    solver = FakeChiralSolver() if args.solver == "fake" else ChiralSolver(model_path)
    param = ChiralParametrization(seed=args.seed, random_init=args.random_init)
    objective = ChiralReflectanceObjective(
        solver=solver, max_consecutive_failures=args.max_consecutive_fails
    )

    config = Config(run_name=run_name, proposer=args.proposer)
    config.seed = args.seed
    config.llm.aggressiveness = args.aggressiveness
    config.stopping.max_iters = args.max_iters
    config.stopping.score_threshold = 0.01

    opt = Optimizer(config, param, objective)
    print(
        f"task=chiral solver={args.solver} proposer={args.proposer} run={run_name}"
        + (" [finalize]" if args.finalize else " [resume]" if args.resume else "")
    )
    if args.finalize:
        summary = opt.finalize_from_disk()
    else:
        summary = opt.run(resume=args.resume)

    if args.solver == "comsol":
        solver.close()
    return summary


if __name__ == "__main__":
    main()
