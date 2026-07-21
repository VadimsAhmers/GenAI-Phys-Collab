"""llm_opt bindings for the multipole-channel inverse-design task.

Implements the two interfaces the generic optimizer needs:
  * RadiiParametrization  -- the search space (N cylinder radii, fixed total height).
  * MultipoleChannelObjective -- scores a geometry by how well its scattered power
    concentrates in the requested multipole channels (uses an injected Solver).

The physics backend is injected as a `Solver`, so COMSOL / chiral / fake are
interchangeable. See src/opt/solvers.py.
"""
from __future__ import annotations

import numpy as np

from llm_opt import Objective, Parametrization, EvalResult

from src.prepare_model import N_CYL, CYL_HEIGHT, CYL_DIAMETER
from src.opt.solvers import Solver, ChiralSolveError
from src.opt.multipole import multipole_loss, feedback_string, validate_target


# Layer size constraint: diameter <= 800 nm -> radius <= 400 nm (README).
R_MIN_NM = 20.0
R_MAX_NM = 400.0


class RadiiParametrization(Parametrization):
    """Search space: `n` independent cylinder radii (nm); layer heights are fixed
    at total_height / n (the paper's parametrization)."""

    def __init__(
        self,
        n: int = N_CYL,
        r_min: float = R_MIN_NM,
        r_max: float = R_MAX_NM,
    ) -> None:
        self.n = n
        self.r_min = r_min
        self.r_max = r_max

    def schema_hint(self) -> dict:
        mid = round((self.r_min + self.r_max) / 2, 1)
        return {"radii_nm": [mid] * self.n}

    def bounds_description(self) -> str:
        return (
            f"- radii_nm: list of {self.n} cylinder radii in nm, each in "
            f"[{self.r_min}, {self.r_max}]. Layers are stacked along the axis with "
            f"fixed equal heights ({CYL_HEIGHT}/{self.n} nm each); only the radii "
            f"are free. The structure is axially symmetric."
        )

    def decode_llm(self, payload: dict) -> dict:
        if "radii_nm" not in payload or not isinstance(payload["radii_nm"], (list, tuple)):
            raise ValueError("missing list field 'radii_nm'")
        try:
            return {"radii_nm": [float(v) for v in payload["radii_nm"]]}
        except (TypeError, ValueError):
            raise ValueError("'radii_nm' must be a list of numbers")

    def validate(self, params: dict) -> list[str]:
        r = params.get("radii_nm", [])
        errs: list[str] = []
        if len(r) != self.n:
            errs.append(f"radii_nm has {len(r)} entries, expected {self.n}")
        for i, v in enumerate(r):
            if not (self.r_min <= v <= self.r_max):
                errs.append(f"radii_nm[{i}]={v} out of [{self.r_min}, {self.r_max}]")
        return errs

    def initial_params(self) -> dict:
        return {"radii_nm": [round(CYL_DIAMETER / 2, 1)] * self.n}

    def random_params(self, rng) -> dict:
        return {
            "radii_nm": [
                round(float(v), 1)
                for v in rng.uniform(self.r_min, self.r_max, self.n)
            ]
        }

    def format_params(self, params: dict) -> str:
        return "[" + ", ".join(f"{v:.0f}" for v in params["radii_nm"]) + "]"

    def describe_change(self, prev: dict, cur: dict) -> str:
        ch = [
            f"r{i}:{prev['radii_nm'][i]:.0f}->{cur['radii_nm'][i]:.0f}"
            for i in range(len(cur["radii_nm"]))
            if abs(cur["radii_nm"][i] - prev["radii_nm"][i]) > 1e-6
        ]
        return ", ".join(ch) if ch else "no change"

    def dedup_key(self, params: dict):
        return tuple(round(v, 1) for v in params["radii_nm"])


# --- Chiral meta-atom (chiral mirror) -------------------------------------------
# Non-axisymmetric task (Andrei B. / Maksim T.). Microwave regime: f = 5.8 GHz
# (lambda ~ 51.7 mm); geometry is in MILLIMETRES, 4 free dimensions. Objective is
# reflectance of ONE circular polarization (1 - |r|^2 -> min); the COMSOL/CST
# solver is injected separately.
CHIRAL_FREQ_GHZ = 5.8

# name -> (min_mm, max_mm, human description)
# Bounds widened ~20-30% at Maksim T.'s request (2026-07-07) to push |r_RR|^2 -> 1
# (the resonant optimum sat against the old r/h ceilings). r's upper is CAPPED by
# the waveguide: the disk sits transverse in a 32x32 mm square guide, whose
# inscribed circle has radius 16 mm, so r=15.5 keeps ~0.5 mm clearance to the walls
# (larger and the disk touches/exceeds the guide). h is along propagation (L_wg =
# 150 mm) so it can grow freely. h's upper raised 8 -> 20 with Maksim T.'s approval
# (2026-07-21) after the eps run pinned h to its ceiling; L_wg=150 mm leaves ~65 mm
# clearance to the ports at h=20. NB: a much thicker slab is less a "thin meta-atom"
# and more a dielectric plug -- flagged to Maksim, who is fine exploring it.
CHIRAL_PARAMS = {
    "r_mm": (4.0, 15.5, "particle radius"),
    "h_mm": (2.0, 20.0, "thickness"),
    "y_cut_mm": (0.0, 17.5, "cut vertical offset (0 = centred in the geometry)"),
    "r_cut_mm": (0.0, 17.5, "cut radius"),
}

# Optional 5th DOF: the dielectric relative permittivity. Maksim T. (2026-07-20)
# asked to make eps_r a search variable in [5, 10] (a manufacturable low-K
# dielectric), same objective, jointly with the geometry. It is DIMENSIONLESS, so
# it needs no mm unit -- see CHIRAL_UNITS and ChiralSolver.PARAM_UNITS.
CHIRAL_EPS_PARAM = {"eps_r": (5.0, 10.0, "relative permittivity (dimensionless)")}

# Per-parameter unit for prompts and COMSOL parameter strings ('' = dimensionless).
CHIRAL_UNITS = {
    "r_mm": "mm",
    "h_mm": "mm",
    "y_cut_mm": "mm",
    "r_cut_mm": "mm",
    "eps_r": "",
}

# Combined catalog (bounds + descriptions) for looking up any param's metadata.
_CHIRAL_CATALOG = {**CHIRAL_PARAMS, **CHIRAL_EPS_PARAM}


class ChiralParametrization(Parametrization):
    """Search space for the chiral mirror: 4 scalar geometry params (mm).

    Unlike RadiiParametrization (axisymmetric cylinder stack), this structure is
    chiral / non-axisymmetric. Bounds and frequency are Maksim T.'s 5.8 GHz
    example; pass `bounds` to retarget another frequency.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        freq_ghz: float = CHIRAL_FREQ_GHZ,
        seed: int | None = None,
        random_init: bool = False,
        optimize_eps: bool = False,
    ) -> None:
        if bounds is not None:
            self.bounds = dict(bounds)
        else:
            base = dict(CHIRAL_PARAMS)
            if optimize_eps:
                base.update(CHIRAL_EPS_PARAM)
            self.bounds = {k: (lo, hi) for k, (lo, hi, _desc) in base.items()}
        # Descriptions/units pulled from the combined catalog so eps_r (and any
        # custom `bounds` key present there) is described correctly.
        self.descriptions = {
            k: _CHIRAL_CATALOG.get(k, (None, None, ""))[2] for k in self.bounds
        }
        self.optimize_eps = optimize_eps
        self.keys = list(self.bounds)
        self.freq_ghz = freq_ghz
        # For multistart: when `random_init` is set, step 0 is a random in-bounds
        # point drawn from this (seeded) rng instead of the fixed midpoint, so
        # separate runs (different `seed`) explore different basins. `seed` is the
        # same value passed to the optimizer's Config, keeping a run reproducible.
        self.random_init = random_init
        self._rng = np.random.default_rng(seed)

    def schema_hint(self) -> dict:
        return {k: round((lo + hi) / 2, 1) for k, (lo, hi) in self.bounds.items()}

    def bounds_description(self) -> str:
        lines = []
        for k, (lo, hi) in self.bounds.items():
            unit = CHIRAL_UNITS.get(k, "mm")
            suffix = f" {unit}" if unit else ""
            lines.append(f"- {k}: {self.descriptions.get(k, '')} in [{lo}, {hi}]{suffix}")
        eps_note = (
            " eps_r is the dielectric's relative permittivity (dimensionless), a "
            "free variable here." if self.optimize_eps else ""
        )
        return (
            f"All lengths in mm; design frequency {self.freq_ghz} GHz. The structure "
            f"is chiral (non-axisymmetric).{eps_note}\n" + "\n".join(lines)
        )

    def decode_llm(self, payload: dict) -> dict:
        out: dict = {}
        for k in self.keys:
            if k not in payload:
                raise ValueError(f"missing field '{k}'")
            try:
                out[k] = float(payload[k])
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' must be a number")
        return out

    def validate(self, params: dict) -> list[str]:
        errs: list[str] = []
        for k, (lo, hi) in self.bounds.items():
            if k not in params:
                errs.append(f"missing '{k}'")
                continue
            v = params[k]
            if not (lo <= v <= hi):
                unit = CHIRAL_UNITS.get(k, "mm")
                suffix = f" {unit}" if unit else ""
                errs.append(f"{k}={v} out of [{lo}, {hi}]{suffix}")
        # NOTE: Maksim gave independent box bounds. Whether a cut larger than the
        # body (e.g. r_cut/y_cut vs r) is geometrically degenerate is still open --
        # add a coupling constraint here once confirmed.
        return errs

    def initial_params(self) -> dict:
        # Random seeded start for multistart; otherwise the deterministic midpoint.
        if self.random_init:
            return self.random_params(self._rng)
        return {k: round((lo + hi) / 2, 1) for k, (lo, hi) in self.bounds.items()}

    def random_params(self, rng) -> dict:
        return {
            k: round(float(rng.uniform(lo, hi)), 1)
            for k, (lo, hi) in self.bounds.items()
        }

    def format_params(self, params: dict) -> str:
        return "{" + ", ".join(f"{k}={params[k]:.1f}" for k in self.keys) + "}"

    def describe_change(self, prev: dict, cur: dict) -> str:
        ch = [
            f"{k}:{prev[k]:.1f}->{cur[k]:.1f}"
            for k in self.keys
            if abs(cur[k] - prev[k]) > 1e-6
        ]
        return ", ".join(ch) if ch else "no change"

    def dedup_key(self, params: dict):
        return tuple(round(params[k], 1) for k in self.keys)


class ChiralReflectanceObjective(Objective):
    """Scores a chiral-mirror geometry by 1 - |r_RR|^2 (0 = perfect one-handed mirror).

    Consumes a `ChiralSolver` directly (scalar power reflection of the target
    circular polarization) rather than the multipole `Solver` protocol -- the chiral
    task is scalar and deliberately not reformulated through multipoles.
    """

    def __init__(self, solver, max_consecutive_failures: int = 10) -> None:
        self.solver = solver
        # Circuit breaker: abort the run after this many *consecutive* solve
        # failures (0 disables). A dropped license / dead COMSOL server makes
        # every solve fail; without this, the run would keep scoring 1.0 forever
        # even though `_is_geometry_failure` mis-slots the message as recoverable.
        # A healthy run resets the counter on any successful solve, so legitimate
        # exploration that hits scattered bad geometries won't trip it.
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0

    def evaluate(self, params: dict) -> EvalResult:
        try:
            r2 = self.solver(params)  # |r_RR|^2 in [0, 1]
        except ChiralSolveError as exc:
            self._consecutive_failures += 1
            if (
                self.max_consecutive_failures
                and self._consecutive_failures >= self.max_consecutive_failures
            ):
                raise RuntimeError(
                    f"{self._consecutive_failures} consecutive COMSOL solve "
                    f"failures -- aborting (likely a broken solver/license, not "
                    f"just bad geometry). Last message: {exc}"
                ) from exc
            # Degenerate geometry COMSOL couldn't mesh/solve. Score it as the
            # worst possible point (|r_RR|^2 = 0) so the optimizer avoids this
            # region and keeps running, and tell the LLM why it was rejected.
            return EvalResult(
                score=1.0,
                feedback=(
                    "INVALID geometry: COMSOL failed to build/mesh this design "
                    "(likely the cut intersecting the disk boundary). Treated as "
                    "|r_RR|^2 = 0 (worst). Keep r_cut/y_cut so the cut stays inside "
                    f"the disk body. Solver message: {exc}"
                ),
                aux={"abs_r_RR_sq": 0.0, "solve_failed": True},
            )
        self._consecutive_failures = 0
        score = 1.0 - r2
        return EvalResult(
            score=score,
            feedback=(
                f"|r_RR|^2 = {r2:.4f} -- reflected power of the target circular "
                f"polarization (want 1.0). Score 1-|r_RR|^2 = {score:.4f}, aim for 0."
            ),
            aux={"abs_r_RR_sq": r2},
        )

    def target_description(self) -> str:
        return (
            "Maximize the reflected power |r_RR|^2 of the target (right-handed) "
            "circular polarization -- a chiral mirror. Score = 1 - |r_RR|^2; 0 is a "
            "perfect one-handed mirror."
        )

    def problem_preamble(self) -> str:
        return (
            "You are designing a chiral (non-axisymmetric) meta-atom that reflects a "
            "single circular polarization at microwave frequency. A cylindrical "
            "dielectric disk (radius r, thickness h) has an off-centre cylindrical "
            "cut (vertical offset y_cut, radius r_cut) that breaks mirror symmetry "
            "and makes the reflection handedness-selective. A full-wave simulation "
            "returns the S-matrix, from which the right->right circular reflection "
            "coefficient r_RR is computed."
        )


class MultipoleChannelObjective(Objective):
    """Scores a geometry by the channel-distribution loss (formulation 2).

    target: desired channel shares, e.g. {"ED": 1.0} or {"ED": .5, "MD": .5}.
    """

    def __init__(
        self,
        solver: Solver,
        target: dict[str, float],
        beta: float = 0.0,
        P_min: float = 1e-4,
    ) -> None:
        validate_target(target)
        self.solver = solver
        self.target = target
        self.beta = beta
        self.P_min = P_min
        self.M = getattr(solver, "M", 4)

    def evaluate(self, params: dict) -> EvalResult:
        radii = np.asarray(params["radii_nm"], dtype=float)
        coeffs = self.solver(radii)
        loss, info = multipole_loss(
            coeffs, self.target, M=self.M, beta=self.beta, P_min=self.P_min
        )
        return EvalResult(
            score=loss,
            feedback=feedback_string(info, self.target),
            aux={"P": info["P"], "q": info["q"], "P_total": info["P_total"]},
        )

    def target_description(self) -> str:
        tgt = {k: v / sum(self.target.values()) for k, v in self.target.items()}
        wanted = ", ".join(f"{k}={v:.2f}" for k, v in tgt.items())
        return (
            "Concentrate the scattered power into these multipole channels "
            f"(as fractions of total scattered power): {wanted}. All other channels "
            "should be ~0. Score = sum over channels of (share - target)^2; 0 is "
            "perfect."
        )

    def problem_preamble(self) -> str:
        return (
            "You are designing an axially symmetric optical scatterer: a vertical "
            "stack of dielectric cylindrical layers of equal height and free radii. "
            "Light scattered by the structure is decomposed into multipole channels "
            "-- electric ED, EQ, EO, EH and magnetic MD, MQ, MO, MH (orders n=1..4). "
            "Changing the radii reshapes how scattered power distributes across these "
            "channels. Low-order channels (ED, MD) are easy to excite; high-order "
            "ones (EO, EH, ...) are weaker and harder to make dominant."
        )
