"""Shared oracle: geometry -> (multipole coefficients, loss).

Every optimizer (CMA-ES baseline, OPRO-like, LLM-driven evolution) plugs into
this same expensive forward evaluation so comparisons stay apples-to-apples.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import mph

from src.cylinder import SingleCylinder, build_cylinders
from src.ab_multipoles import MULTIPOLES_TE, eval_ab, M
from src.prepare_model import N_CYL, CYL_HEIGHT

Coeffs = dict  # {"a"|"b": {n: {m: complex}}}


# --- coefficient (de)vectorization in a single canonical order ---------------

def coeff_labels() -> list[str]:
    """Canonical flat order of the 48 coefficients (matches coeffs_to_vector)."""
    return [
        f"{kind}{n},{m}"
        for kind in ("a", "b")
        for n in range(1, M + 1)
        for m in range(-n, n + 1)
    ]


def coeffs_to_vector(coeffs: Coeffs) -> np.ndarray:
    """Flatten {"a"/"b": {n: {m: complex}}} into a fixed-order complex vector."""
    return np.array(
        [
            coeffs[kind][n][m]
            for kind in ("a", "b")
            for n in range(1, M + 1)
            for m in range(-n, n + 1)
        ],
        dtype=complex,
    )


def load_target(path: str | Path = "target.pkl") -> np.ndarray:
    with open(path, "rb") as f:
        return coeffs_to_vector(pickle.load(f))


# --- loss functions ----------------------------------------------------------
# The 48 target magnitudes span ~5 orders (5.6e-3 .. 7.7e-8). A plain L2 is
# dominated by the few largest (dipole/quadrupole) terms and ignores the small
# high-order ones, so we offer a magnitude-balanced loss alongside the naive one.

def relative_l2(vec: np.ndarray, target: np.ndarray) -> float:
    """Naive global relative L2: ||c - t||^2 / ||t||^2. Magnitude-dominated."""
    denom = np.sum(np.abs(target) ** 2)
    return float(np.sum(np.abs(vec - target) ** 2) / denom)


def weighted_relative(
    vec: np.ndarray, target: np.ndarray, floor_rel: float = 1e-3
) -> float:
    """Per-coefficient relative error with a floor, so every multipole order
    contributes comparably regardless of its magnitude.

    floor_rel sets the smallest |target| (as a fraction of max|target|) that is
    still weighted relatively; below it, coefficients (incl. the exact zeros)
    are weighted at the floor — penalizing spurious excitation without letting
    the tiniest terms dominate.
    """
    floor = floor_rel * np.max(np.abs(target))
    w = 1.0 / (np.abs(target) ** 2 + floor ** 2)
    return float(np.mean(w * np.abs(vec - target) ** 2))


# --- geometry helper ---------------------------------------------------------

def radii_to_cylinders(
    radii: np.ndarray, total_height: float = CYL_HEIGHT
) -> list[SingleCylinder]:
    """Map a radius vector to stacked cylinders with equal heights H/N
    (the paper's parametrization: independent radii, single shared total height).
    """
    radii = np.asarray(radii, dtype=float)
    h = total_height / len(radii)
    return [SingleCylinder(h=h, r=float(r)) for r in radii]


# --- the expensive oracle ----------------------------------------------------

@dataclass
class ScatteringObjective:
    """Loads the prepared COMSOL model once, then maps geometry -> (coeffs, loss).

    Use evaluate() when you want the rich multipole feedback (for LLM context),
    or call the instance directly to get just the scalar loss (for classic
    optimizers).
    """

    target: np.ndarray
    loss_fn: Callable[[np.ndarray, np.ndarray], float]
    model_path: str = "models/model.mph"
    n_calls: int = 0

    def __post_init__(self) -> None:
        self._client = mph.start()
        self._model = self._client.load(self.model_path)

    def evaluate(
        self, radii: np.ndarray, total_height: float = CYL_HEIGHT
    ) -> tuple[Coeffs, np.ndarray, float]:
        """Build geometry, solve, return (coeffs dict, coeff vector, loss)."""
        cylinders = radii_to_cylinders(radii, total_height)
        build_cylinders(self._model, cylinders)
        self._model.solve()
        coeffs = eval_ab(self._model, pol="te", multipoles=MULTIPOLES_TE)
        vec = coeffs_to_vector(coeffs)
        self.n_calls += 1
        return coeffs, vec, self.loss_fn(vec, self.target)

    def __call__(self, radii: np.ndarray, total_height: float = CYL_HEIGHT) -> float:
        _, _, loss = self.evaluate(radii, total_height)
        return loss

    def close(self) -> None:
        self._client.remove(self._model)


def make_objective(
    loss_fn: Callable[[np.ndarray, np.ndarray], float] = weighted_relative,
    target_path: str | Path = "target.pkl",
    model_path: str = "models/model.mph",
) -> ScatteringObjective:
    return ScatteringObjective(
        target=load_target(target_path), loss_fn=loss_fn, model_path=model_path
    )


if __name__ == "__main__":
    # Smoke test: the seed=24 geometry is exactly how target.pkl was generated,
    # so its loss against the target must be ~0. Requires COMSOL.
    rng = np.random.default_rng(seed=24)
    from src.prepare_model import CYL_DIAMETER

    radii = (CYL_DIAMETER - CYL_DIAMETER / 3 * (rng.random(N_CYL) - 0.5)) / 2

    obj = make_objective(loss_fn=weighted_relative)
    coeffs, vec, loss = obj.evaluate(radii)
    print(f"weighted_relative loss at target geometry: {loss:.3e} (expected ~0)")
    print(f"relative_l2 loss: {relative_l2(vec, obj.target):.3e}")
    obj.close()
