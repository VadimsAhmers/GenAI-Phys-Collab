"""Run the chiral-mirror optimizer (one-handed circular-polarization reflection).

The task is scalar: maximize |r_RR|^2 (score = 1 - |r_RR|^2 -> 0). Geometry is the
4 mm params of `ChiralParametrization`; the physics is COMSOL via `ChiralSolver`.

Dry run (no COMSOL, no API key) -- exercises the whole loop with a synthetic oracle:

    python -m src.opt.run_chiral --solver fake --proposer random --max-iters 12

Real LLM-driven run against COMSOL (needs OPENROUTER_API_KEY and a COMSOL license):

    python -m src.opt.run_chiral --solver comsol --proposer llm --max-iters 50
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from llm_opt import Config, Optimizer
from src.opt.optimize import ChiralParametrization, ChiralReflectanceObjective
from src.opt.solvers import ChiralSolver, FakeChiralSolver


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", default="fake", choices=["fake", "comsol"])
    p.add_argument("--proposer", default="random", choices=["random", "llm"])
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--model", default="models/chiral.mph")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    load_dotenv()

    solver = FakeChiralSolver() if args.solver == "fake" else ChiralSolver(args.model)
    param = ChiralParametrization()
    objective = ChiralReflectanceObjective(solver=solver)

    run_name = args.run_name or f"chiral_{args.solver}_{args.proposer}"
    config = Config(run_name=run_name, proposer=args.proposer)
    config.stopping.max_iters = args.max_iters
    config.stopping.score_threshold = 0.01

    print(f"task=chiral solver={args.solver} proposer={args.proposer}")
    summary = Optimizer(config, param, objective).run()

    if args.solver == "comsol":
        solver.close()
    return summary


if __name__ == "__main__":
    main()
