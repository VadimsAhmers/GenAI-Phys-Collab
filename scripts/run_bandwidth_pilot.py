"""Short local pilot: reused baseline + at most five new proposal attempts.

Run from the repository root, after updating the server through GitHub:
    .venv/bin/python -m scripts.run_bandwidth_pilot --run-name bandwidth_local_20261021

Defaults use COMSOL/LLM. --solver fake --proposer random --baseline-from <fake-run>
exercises the same workflow without COMSOL or API calls. Add --resume with the
same options after interruption. Fine-grid verification of finalists is separate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Temporary exploration bounds around the October sample, not hardware limits.
LOCAL_BOUNDS = {
    "r_mm": (12.5, 13.0), "h_mm": (24.3, 25.3),
    "y_cut_mm": (4.6, 5.6), "r_cut_mm": (7.8, 8.8), "eps_r": (6.9, 7.3),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--baseline-from", default="runs/bandwidth_baseline_20261020")
    p.add_argument("--solver", choices=["comsol", "fake"], default="comsol")
    p.add_argument("--proposer", choices=["llm", "random"], default="llm")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    command = [
        sys.executable, "-u", "-m", "src.opt.run_chiral",
        "--objective", "bandwidth", "--solver", args.solver,
        "--proposer", args.proposer, "--aggressiveness", "minimal",
        "--bandwidth-step", "0.02", "--bandwidth-edge-tol", "0.0025",
        "--reflection-level", "0.8", "--max-iters", "6", "--seed", "0",
        "--run-name", args.run_name, "--baseline-from", args.baseline_from,
    ]
    for name, (lo, hi) in LOCAL_BOUNDS.items():
        command.extend(["--bound", name, str(lo), str(hi)])
    if args.resume:
        command.append("--resume")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(command, cwd=root, check=True)
    # The generic engine finalizes gracefully (exit 0) even after an API error.
    # Make the launcher's exit status reflect that incomplete pilot explicitly.
    summary = json.loads((root / "runs" / args.run_name / "summary.json").read_text())
    if summary["stop_reason"] not in ("max_iters", "threshold"):
        raise SystemExit(f"Pilot stopped early: {summary['stop_reason']}; see saved run logs.")


if __name__ == "__main__":
    main()
