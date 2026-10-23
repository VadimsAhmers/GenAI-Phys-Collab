"""Expanded search with fixed 0.5 mm fillets, starting from the verified finalist.

Default: reused baseline plus five proposal attempts. After reviewing the pilot,
continue the same run with --resume --proposals 25 (25 attempts total, not extra).
Run from the repository root after updating the server through GitHub.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.opt import run_chiral
from src.opt.fillet import FILLET_START


ROOT = Path(__file__).resolve().parents[1]
# Research bounds agreed with Vadim; only r<=13 and eps=6..11 are Maksim's limits.
BOUNDS = dict(r_mm=(10, 13), h_mm=(20, 30), y_cut_mm=(2, 8),
              r_cut_mm=(6, 12), eps_r=(6, 11))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="fillet_search_20261023")
    parser.add_argument("--baseline-from", type=Path, default=ROOT / "runs/fillet_check_comsol_v2")
    parser.add_argument("--solver", choices=("comsol", "fake"), default="comsol")
    parser.add_argument("--proposer", choices=("llm", "random"), default="llm")
    parser.add_argument("--proposals", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.proposals <= 30:
        parser.error("--proposals must be between 1 and 30 (total proposal attempts)")
    command = [
        "run_chiral", "--objective", "bandwidth",
        "--fillet", "--model", str(ROOT / "models/chiral_fillet_0p5.mph"),
        "--start-params", *map(str, FILLET_START.values()),
        "--solver", args.solver, "--proposer", args.proposer,
        "--aggressiveness", "minimal", "--seed", "0",
        "--bandwidth-step", "0.02", "--bandwidth-edge-tol", "0.0025",
        "--reflection-level", "0.8", "--max-consecutive-fails", "3",
        "--max-iters", str(args.proposals + 1), "--run-name", args.run_name,
        "--baseline-from", str(args.baseline_from.resolve()),
    ]
    for name, (lo, hi) in BOUNDS.items():
        command.extend(["--bound", name, str(lo), str(hi)])
    if args.resume:
        command.append("--resume")
    # Existing CLI owns all solver/provenance logic; use its returned status
    # directly so an API failure cannot masquerade as successful completion.
    previous = sys.argv
    try:
        sys.argv = command
        summary = run_chiral.main()
    finally:
        sys.argv = previous
    if summary["stop_reason"] not in ("max_iters", "threshold"):
        raise SystemExit(f"Search stopped early: {summary['stop_reason']}; see saved run logs.")


if __name__ == "__main__":
    main()
