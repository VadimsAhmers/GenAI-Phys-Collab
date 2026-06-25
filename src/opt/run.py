"""Run the multipole-channel optimizer.

Dry run (no COMSOL, no API key) -- exercises the whole loop:

    python -m src.opt.run --task ed --solver fake --proposer random --max-iters 12

Real LLM-driven run against COMSOL (needs OPENROUTER_API_KEY and a COMSOL license):

    python -m src.opt.run --task ed --solver comsol --proposer llm --max-iters 50
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from llm_opt import Config, Optimizer
from src.opt.optimize import RadiiParametrization, MultipoleChannelObjective
from src.opt.solvers import FakeSolver, ComsolSolver
from src.opt.reporter import MultipoleReporter


# Channel targets from new_conversation/formulation.md.
TASKS: dict[str, dict[str, float]] = {
    "ed": {"ED": 1.0},
    "eq": {"EQ": 1.0},
    "edmd": {"ED": 0.5, "MD": 0.5},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="ed", choices=sorted(TASKS))
    p.add_argument("--solver", default="fake", choices=["fake", "comsol"])
    p.add_argument("--proposer", default="random", choices=["random", "llm"])
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    load_dotenv()

    target = TASKS[args.task]
    solver = FakeSolver() if args.solver == "fake" else ComsolSolver()
    param = RadiiParametrization()
    objective = MultipoleChannelObjective(solver=solver, target=target)
    reporter = MultipoleReporter(target=target)

    run_name = args.run_name or f"{args.task}_{args.solver}_{args.proposer}"
    config = Config(run_name=run_name, proposer=args.proposer)
    config.stopping.max_iters = args.max_iters
    config.stopping.score_threshold = 0.01

    print(f"task={args.task} target={target} "
          f"solver={args.solver} proposer={args.proposer}")
    summary = Optimizer(config, param, objective, reporter=reporter).run()

    if args.solver == "comsol":
        solver.close()
    return summary


if __name__ == "__main__":
    main()
