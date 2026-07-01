"""End-to-end smoke test for the chiral-mirror oracle (needs COMSOL + a license).

Runs the real set-params -> solve -> read cycle through `ChiralSolver`:
  1. reference geometry must reproduce the known baseline (|r_RR|^2 ~ 0.8851),
     which checks the read path and model loading;
  2. a different geometry must give a different value, which checks that the
     geometry parameters actually reach the solve.

Usage (on the machine with COMSOL):
    uv run python scripts/smoke_chiral.py
"""
from __future__ import annotations

from src.opt import ChiralSolver, ChiralReflectanceObjective

# Reference (Maksim's provisional) geometry and its known baseline at eps_r=80.
REFERENCE = {"r_mm": 6.0, "h_mm": 3.4, "y_cut_mm": 3.25, "r_cut_mm": 2.9}
BASELINE_R2 = 0.8851
ALT = {"r_mm": 8.0, "h_mm": 4.0, "y_cut_mm": 2.0, "r_cut_mm": 3.5}


def main() -> None:
    solver = ChiralSolver("models/chiral.mph")
    obj = ChiralReflectanceObjective(solver)

    ref = obj.evaluate(REFERENCE)
    r2 = ref.aux["abs_r_RR_sq"]
    print(f"reference: |r_RR|^2 = {r2:.4f}  score = {ref.score:.4f}")

    alt = obj.evaluate(ALT)
    print(f"alt:       |r_RR|^2 = {alt.aux['abs_r_RR_sq']:.4f}  score = {alt.score:.4f}")

    ok_read = abs(r2 - BASELINE_R2) < 1e-3
    ok_params = abs(alt.aux["abs_r_RR_sq"] - r2) > 1e-6
    print(f"\nread path (baseline reproduced): {'OK' if ok_read else 'FAIL'}")
    print(f"param path (alt differs):        {'OK' if ok_params else 'FAIL'}")
    if not (ok_read and ok_params):
        raise SystemExit("smoke test FAILED")
    print("\nsmoke test PASSED")


if __name__ == "__main__":
    main()
