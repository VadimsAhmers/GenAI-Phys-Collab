"""Fixed 0.5 mm fillets in Maksim's supplied crescent model."""
from __future__ import annotations

import hashlib
from pathlib import Path

from llm_opt import EvalResult
from src.opt.bandwidth import ChiralBandwidthObjective
from src.opt.optimize import ChiralParametrization
from src.opt.solvers import ChiralSolver, ChiralSolveError, _is_geometry_failure


FILLET_MODEL_SHA256 = "077a0088784eb6a656771401c38606446a18df365de629313bf64a90fb4a99d1"
FILLET_START = dict(r_mm=13.0, h_mm=25.0, y_cut_mm=4.6, r_cut_mm=8.8, eps_r=7.3)


class FilletParametrization(ChiralParametrization):
    def validate(self, params):
        errors = super().validate(params)
        if not errors:
            r, cut, offset = (params[k] for k in ("r_mm", "r_cut_mm", "y_cut_mm"))
            if not abs(r - cut) < offset < r + cut:
                errors.append(
                    "The fillet model requires an open crescent: the disk and notch "
                    "circles must intersect at two points "
                    "(abs(r_mm-r_cut_mm) < y_cut_mm < r_mm+r_cut_mm)."
                )
        return errors

    def bounds_description(self):
        return (super().bounds_description()
                + " Fixed tip fillets of radius 0.5 mm are part of the COMSOL model; "
                "do not optimize their radius. Keep an open crescent: "
                "abs(r_mm-r_cut_mm) < y_cut_mm < r_mm+r_cut_mm. "
                "COMSOL must also be able to build these fillets.")


class FilletChiralSolver(ChiralSolver):
    def __init__(self, model_path):
        if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != FILLET_MODEL_SHA256:
            raise ValueError("Fillet mode requires the inspected model supplied by Maksim")
        super().__init__(model_path)
        self._prepared_geometry = None

    def prepare_geometry(self, params):
        geometry = {k: params[k] for k in self.PARAM_MAP}
        if geometry == self._prepared_geometry:
            return
        self._prepared_geometry = None
        model = self._ensure_model()
        try:
            for key, value in geometry.items():
                unit = self.PARAM_UNITS[key]
                model.parameter(self.PARAM_MAP[key], f"{float(value)}[{unit}]"
                                if unit else str(float(value)))
            model.build()
        except Exception as exc:
            message = str(exc)
            if _is_geometry_failure(message) or "failed to create fillet" in message.lower():
                raise ChiralSolveError(f"Fillet geometry build failed: {message}") from exc
            raise
        geom = model.java.geom("geom1")
        # Java String must first become a Python str (mph/JPype does not coerce it).
        radius = float(str(geom.feature("fil1").getString("radius")))
        if radius != 0.5:
            raise RuntimeError("Fixed fillet radius changed; refusing to evaluate")
        if int(geom.getNDomains()) != 2:
            raise ChiralSolveError("Fillet geometry must contain exactly air and disk domains")
        self._prepared_geometry = geometry
        print(f"Fillet geometry built (radius 0.5 mm): {geometry}", flush=True)

    def polarizations(self, params):
        self.prepare_geometry(params)
        return super().polarizations(params)

    def close(self):
        self._prepared_geometry = None
        super().close()


class FilletBandwidthObjective(ChiralBandwidthObjective):
    def evaluate(self, params):
        prepare = getattr(self.solver, "prepare_geometry", None)
        if prepare is not None:
            try:
                prepare(params)
            except ChiralSolveError as exc:
                self._consecutive_failures += 1
                if (self.max_consecutive_failures
                        and self._consecutive_failures >= self.max_consecutive_failures):
                    raise RuntimeError("Consecutive fillet geometry failures; aborting") from exc
                return EvalResult(
                    score=2.0, feedback=f"INVALID fillet geometry: {exc}",
                    aux={"solve_failed": True, "geometry_build_failed": True,
                         "solver_calls_new": 0, "spectrum": []},
                )
        return super().evaluate(params)

    def problem_preamble(self):
        return (super().problem_preamble()
                + " The open crescent has fixed 0.5 mm tip fillets, already included "
                "in every new COMSOL evaluation. Optimize the rounded geometry. "
                "Results for sharp tips must not be treated as scores for this model.")
