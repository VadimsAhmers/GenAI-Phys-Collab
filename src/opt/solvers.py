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


class ChiralSolver:
    """Chiral-mirror oracle: 4 geometry params (mm) -> |r_RR|^2 via COMSOL.

    Unlike ComsolSolver (which returns multipole coefficients), this returns the
    scalar power reflection |r_RR|^2 of the target circular polarization, read from
    the Global Evaluation node `r_RR` in Maksim's model. The chiral task is scalar
    (1 - |r_RR|^2 -> min); multipoles are deliberately not used here, so this does
    NOT implement the `Solver` (radii -> Coeffs) protocol.

    Lazily starts COMSOL and loads the model on first call. The model must be opened
    in COMSOL >= its authoring version (6.4) and ships without a stored solution, so
    each call solves the port-sweep study (~25 s) before reading r_RR.
    """

    # our ChiralParametrization key -> COMSOL model parameter name
    PARAM_MAP = {
        "r_mm": "r_disk",
        "h_mm": "h_disk",
        "y_cut_mm": "y_cut",
        "r_cut_mm": "r_cut",
    }
    STUDY = "TE01 excitation"
    EVAL_NODE = "r_RR"

    def __init__(self, model_path: str = "models/chiral.mph") -> None:
        self.model_path = model_path
        self._client = None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import mph

            self._client = mph.start()
            self._model = self._client.load(self.model_path)
        return self._model

    def __call__(self, params: dict) -> float:
        model = self._ensure_model()
        for key, comsol_name in self.PARAM_MAP.items():
            model.parameter(comsol_name, f"{float(params[key])}[mm]")
        model.solve(self.STUDY)
        # `r_RR` node already applies abs(...)^2 -> real |r_RR|^2 in [0, 1]; the
        # port sweep yields one (identical) value per excitation, so take [0][0].
        node = model / "evaluations" / self.EVAL_NODE
        return float(node.java.getReal()[0][0])

    def close(self) -> None:
        self._model = None
        self._client = None
