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


class FakeChiralSolver:
    """Synthetic |r_RR|^2 for dry-running the chiral loop without COMSOL or an API.

    NOT physical: a smooth deterministic bump over the 4 params, in (0, 1], so the
    score varies with the candidate (dedup/feedback behave realistically). Plumbing
    only; never for results.
    """

    KEYS = ("r_mm", "h_mm", "y_cut_mm", "r_cut_mm")

    # Synthetic resonance for frequency sweeps: the bump above is the peak HEIGHT,
    # and `freq_ghz` (when asked for) multiplies it by a Lorentzian whose centre
    # scales as 1/size, mimicking the real 1/a scaling. Lets the frequency
    # machinery and `spectrum_summary` be exercised with no COMSOL.
    F0_GHZ = 5.8
    REF_R_MM = 6.0
    FWHM_GHZ = 0.15

    def __call__(self, params: dict) -> float:
        v = np.array([float(params[k]) for k in self.KEYS])
        # off the parametrization midpoint (9.5, 4, 7, 7) so a dry run actually iterates
        center = np.array([7.0, 3.0, 4.0, 10.0])
        scale = np.array([9.0, 4.0, 14.0, 14.0])
        d2 = float(np.sum(((v - center) / scale) ** 2))
        # Optional 5th DOF: fold eps_r into the synthetic bump (optimum near 8.5) so
        # an `--optimize-eps` dry run varies with it too.
        if "eps_r" in params:
            d2 += ((float(params["eps_r"]) - 8.5) / 2.5) ** 2
        height = float(np.exp(-2.0 * d2))
        if "freq_ghz" not in params:
            return height
        f_res = self.F0_GHZ * self.REF_R_MM / max(float(params["r_mm"]), 1e-9)
        detune = (float(params["freq_ghz"]) - f_res) / (self.FWHM_GHZ / 2.0)
        return height / (1.0 + detune**2)

    def polarizations(self, params: dict) -> dict[str, float]:
        """Synthetic, deliberately different channels; not physical results."""
        rr = self(params)
        shifted = dict(params)
        if "freq_ghz" in shifted:
            shifted["freq_ghz"] -= 0.02
        return {"abs_r_RR_sq": rr, "abs_r_LL_sq": 0.9 * self(shifted)}


class ChiralSolveError(Exception):
    """A COMSOL solve failed for THIS geometry (bad mesh / no convergence).

    Raised instead of letting the raw Java/mph exception propagate, so a single
    invalid geometry proposed by the optimizer is scored as a bad point rather
    than crashing the whole run. Carries the COMSOL message for LLM feedback.

    Only per-geometry failures are wrapped this way (see GEOMETRY_FAIL_MARKERS);
    infrastructure failures (lost license, dropped COMSOL server, out of memory,
    mph/API bugs) are NOT -- they would recur on every solve, so they must abort
    the run loudly rather than silently poison it with bogus worst-case scores.
    """


# Substrings (matched case-insensitively) that identify a failure specific to the
# proposed geometry/physics point -- safe to score as a bad point and continue.
# Anything not matching these is treated as a genuine error and re-raised.
GEOMETRY_FAIL_MARKERS = (
    "mesh",                     # "Failed to generate mesh", "Free Tetrahedral 1"
    "intersecting",             # "Intersecting face elements"
    "geometry",                 # geometry build/finalize failures
    "multiphysics compilation", # degenerate domain -> compile error
    "failed to find a solution",
    "no solution",
    "singular",                 # singular matrix (ill-posed geometry)
    "undefined value",          # e.g. divide-by-zero from a collapsed domain
)


def _is_geometry_failure(message: str) -> bool:
    msg = message.lower()
    return any(marker in msg for marker in GEOMETRY_FAIL_MARKERS)


def _ll_expression_from_rr(expression: str) -> str:
    """Opposite circular basis for the expression in the shipped chiral model.

    RR = (S11-S22-i*(S21+S12))/2; LL changes the sign of i. This conjugates
    the basis coefficients, NOT the complex S parameters. Refuse an unknown
    model convention instead of silently inventing a second observable.
    """
    s11 = "withsol('sol4',(ewfd.S11),setval(freq,freq))"
    s22 = "withsol('sol5',(ewfd.S22),setval(freq,freq))"
    s21 = "withsol('sol4',(ewfd.S21),setval(freq,freq))"
    s12 = "withsol('sol5',(ewfd.S12),setval(freq,freq))"
    expected = f"abs(1/2*({s11}-{s22}-1i*({s21}+{s12})))^2"
    compact = "".join(expression.split())
    if compact != expected:
        raise ValueError(
            "Unrecognized r_RR expression: verify the circular-polarization "
            "convention before deriving r_LL for this model."
        )
    return compact.replace("-1i*", "+1i*")


class ChiralSolver:
    """Chiral-mirror oracle: 4 geometry params (mm) -> |r_RR|^2 via COMSOL.

    `polarizations(params)` additionally returns |r_LL|^2 from the same solve.

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
        "eps_r": "eps_r",  # optional 5th DOF (only set when present in params)
    }
    # COMSOL unit suffix per parameter ('' = dimensionless, written as a bare number).
    PARAM_UNITS = {
        "r_mm": "mm",
        "h_mm": "mm",
        "y_cut_mm": "mm",
        "r_cut_mm": "mm",
        "eps_r": "",
    }
    STUDY = "TE01 excitation"
    EVAL_NODE = "r_RR"

    # Optional key in `params`: solve at this frequency instead of the model's own
    # 5.800 GHz. Unlike the geometry, frequency is NOT a global model parameter --
    # Maksim's model hard-codes it in the study's Frequency Domain step -- so it is
    # applied by rewriting that step's list, not via model.parameter(). Absent key
    # => the study is left completely untouched (every existing 5.8 GHz run and
    # sweep behaves exactly as before).
    FREQ_KEY = "freq_ghz"

    def __init__(self, model_path: str = "models/chiral.mph") -> None:
        self.model_path = model_path
        self._client = None
        self._model = None
        self._freq_features = None
        self._freq_ghz = None

    def _ensure_model(self):
        if self._model is None:
            import mph

            self._client = mph.start()
            self._model = self._client.load(self.model_path)
        return self._model

    # Study-step type whose `plist` is a FREQUENCY list. Matched on the type and
    # not merely on "has a plist", because a Parametric Sweep step also carries a
    # value list -- and this model sweeps the PORT (PortName) to build the
    # S-matrix that r_RR is computed from. Writing a frequency into that list
    # would break the port sweep silently and poison every number it produces.
    # Matched as a case-insensitive substring of the step type ("Frequency" in
    # 6.3/6.4) so a renamed variant still matches, while "Parametric",
    # "Stationary" etc. never do.
    FREQ_STEP_TYPE = "freq"

    def _frequency_steps(self, model):
        """Study steps whose frequency list can be rewritten (discovered once).

        Discovered by type rather than by hard-coded tags, because the study/step
        tags are Maksim's and may change between model revisions. Every matching
        step in EVERY study is collected: the r_RR node reads two port-excitation
        solutions, and a frequency applied to only some of them would silently
        mix frequencies inside one number.
        """
        if self._freq_features is None:
            steps = []
            skipped = []
            java = model.java
            for study_tag in java.study().tags():
                study = java.study(study_tag)
                for feat_tag in study.feature().tags():
                    feature = study.feature(feat_tag)
                    kind = str(feature.getType())
                    has_plist = True
                    try:
                        feature.getString("plist")
                    except Exception:  # noqa: BLE001 - step carries no value list
                        has_plist = False
                    if self.FREQ_STEP_TYPE not in kind.lower() or not has_plist:
                        skipped.append(f"{study_tag}/{feat_tag} ({kind})")
                        continue
                    steps.append((str(study_tag), str(feat_tag), feature))
            if not steps:
                raise RuntimeError(
                    f"{self.model_path}: found no frequency study step (type "
                    f"containing '{self.FREQ_STEP_TYPE}' with a 'plist'), so the "
                    f"solve frequency cannot be set from here (saw: "
                    f"{', '.join(skipped) or 'no steps'}). Drop "
                    f"'{self.FREQ_KEY}' to solve at the model's own frequency."
                )
            # Printed once: a night of spectra hangs on this having matched the
            # right steps, so it belongs in the log, not just in a debugger.
            print(
                "ChiralSolver: frequency will be set on "
                + ", ".join(f"{s}/{f}" for s, f, _ in steps)
                + (f" (left alone: {', '.join(skipped)})" if skipped else "")
            )
            self._freq_features = steps
        return self._freq_features

    def set_frequency(self, freq_ghz: float) -> None:
        """Solve subsequent calls at `freq_ghz` (GHz). Not persisted to the file."""
        model = self._ensure_model()
        # Unit-carrying string: this is the form the model itself stores
        # ("5.800[GHz]"), so it is safe alongside the step's own `punit`.
        value = f"{float(freq_ghz)}[GHz]"
        for _study, _feat, feature in self._frequency_steps(model):
            feature.set("plist", value)
        self._freq_ghz = float(freq_ghz)

    def __call__(self, params: dict) -> float:
        model = self._ensure_model()
        # Outside the try: a frequency that cannot be applied is a broken model or
        # a broken session, not a bad geometry -- it must abort, not score 0.
        if self.FREQ_KEY in params:
            freq_ghz = float(params[self.FREQ_KEY])
            if freq_ghz != self._freq_ghz:
                self.set_frequency(freq_ghz)
        try:
            # Setting a parameter can itself fail on a degenerate geometry: COMSOL
            # re-evaluates selections/visualization mesh eagerly, so a bad point can
            # blow up here (e.g. "Failed to create visualization mesh") before solve.
            for key, comsol_name in self.PARAM_MAP.items():
                if key not in params:  # eps_r absent in a 4-DOF (geometry-only) run
                    continue
                unit = self.PARAM_UNITS.get(key, "mm")
                value = f"{float(params[key])}[{unit}]" if unit else f"{float(params[key])}"
                model.parameter(comsol_name, value)
            model.solve(self.STUDY)
        except Exception as exc:  # noqa: BLE001 - mph re-raises a Java exception
            # Only wrap failures tied to THIS geometry (bad mesh / no convergence)
            # so the caller can score it as a bad point and continue. Everything
            # else (lost license, dropped server, OOM, mph/API bug) would recur on
            # every solve, so let it propagate and abort the run loudly.
            if _is_geometry_failure(str(exc)):
                raise ChiralSolveError(str(exc)) from exc
            raise
        # `r_RR` node already applies abs(...)^2 -> real |r_RR|^2 in [0, 1]; the
        # port sweep yields one (identical) value per excitation, so take [0][0].
        node = model / "evaluations" / self.EVAL_NODE
        return float(node.java.getReal()[0][0])

    def polarizations(self, params: dict) -> dict[str, float]:
        """One port-sweep solve, then RR and LL power evaluations on that solution.

        Reuse the RR node's dataset and solution selections. Only its expression
        changes temporarily, and is restored even if LL evaluation fails. The
        model file is never saved. Evaluation/configuration errors propagate;
        they must not be reported as bad geometry or as zero LL reflection.
        """
        model = self._ensure_model()
        node = model / "evaluations" / self.EVAL_NODE
        original = node.property("expr")
        expressions = [original] if isinstance(original, str) else list(original)
        if len(expressions) != 1:
            raise ValueError("Expected a single power expression in r_RR.")
        ll_expression = _ll_expression_from_rr(str(expressions[0]))
        rr = self(params)
        try:
            node.property("expr", [ll_expression])
            ll = float(node.java.getReal()[0][0])
        finally:
            node.property("expr", original)
        return {"abs_r_RR_sq": rr, "abs_r_LL_sq": ll}

    def close(self) -> None:
        self._model = None
        self._client = None
        self._freq_features = None
        self._freq_ghz = None
