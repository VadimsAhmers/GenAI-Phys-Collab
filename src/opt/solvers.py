"""Solver abstraction: geometry -> multipole coefficients.

This is the seam that lets the SAME optimizer drive different physics backends.
Today: the COMSOL axisymmetric oracle (`ComsolSolver`). Tomorrow: Andrei/Maksim's
chiral (non-axisymmetric) solver -- just another implementation of `Solver`.
`FakeSolver` lets the whole loop run with no COMSOL and no API (dry tests).

A Solver maps a radii vector (nm) to coefficients {"a"|"b": {n: {m: complex}}}.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


Coeffs = dict  # {"a"|"b": {n: {m: complex}}}


class Solver(Protocol):
    M: int

    def __call__(self, radii: np.ndarray) -> Coeffs:
        ...


class FakeSolver:
    """Deterministic synthetic coefficients for dry-running the loop without COMSOL.

    NOT physical: magnitudes decay with order n and depend on the radii so that the
    score varies with the candidate (caching/feedback behave realistically). Use
    only to exercise plumbing; never for results.
    """

    def __init__(self, M: int = 4) -> None:
        self.M = M

    def __call__(self, radii: np.ndarray) -> Coeffs:
        r = np.asarray(radii, dtype=float)
        seed = abs(hash(tuple(np.round(r, 3).tolist()))) % (2**32)
        rng = np.random.default_rng(seed)
        rbar = float(r.mean())
        coeffs: Coeffs = {"a": {}, "b": {}}
        for kind in ("a", "b"):
            for n in range(1, self.M + 1):
                coeffs[kind][n] = {}
                base = np.exp(-n) * (1.0 + 0.5 * np.sin(rbar / 100.0 * n))
                for m in range(-n, n + 1):
                    coeffs[kind][n][m] = complex(
                        base * (rng.normal() + 1j * rng.normal())
                    )
        return coeffs


class ComsolSolver:
    """Wraps the COMSOL axisymmetric oracle (`src.objective.ScatteringObjective`).

    Lazily starts COMSOL on first call. Reuses the existing oracle; the loss it
    computes internally is ignored here -- we only take the coefficients.
    """

    M: int = 4

    def __init__(self, model_path: str = "models/model.mph") -> None:
        self.model_path = model_path
        self._oracle = None

    def __call__(self, radii: np.ndarray) -> Coeffs:
        if self._oracle is None:
            from src.objective import make_objective, relative_l2

            # target/loss only satisfy the oracle's ctor; we ignore the loss value.
            self._oracle = make_objective(
                loss_fn=relative_l2, model_path=self.model_path
            )
        coeffs, _vec, _loss = self._oracle.evaluate(np.asarray(radii, dtype=float))
        return coeffs

    def close(self) -> None:
        if self._oracle is not None:
            self._oracle.close()
