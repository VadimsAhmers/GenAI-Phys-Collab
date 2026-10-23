"""Rounded-geometry checks and seeded bandwidth search without COMSOL/API."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import check_fillet_bandwidth, run_fillet_search
from src.opt import run_chiral
from src.opt.fillet import (
    FILLET_START, FilletChiralSolver, FilletParametrization, FilletBandwidthObjective,
)
from src.opt.solvers import ChiralSolver, ChiralSolveError
from tests.test_bandwidth_search import cli_root


def test_closed_hole_and_tangency_rejected_before_solver():
    param = FilletParametrization(bounds=run_fillet_search.BOUNDS, optimize_eps=True)
    assert not param.validate(FILLET_START)
    for offset in (2, 7):
        assert "open crescent" in " ".join(param.validate(
            {**FILLET_START, "r_cut_mm": 6, "y_cut_mm": offset}))
    assert "0.5 mm" in param.bounds_description()


def test_solver_rebuilds_once_per_geometry_before_any_frequency_solve(monkeypatch):
    events = []
    class JavaString:
        def __str__(self):
            return "0.5"

    geom = SimpleNamespace(getNDomains=lambda: 2,
                           feature=lambda tag: SimpleNamespace(getString=lambda key: JavaString()))
    model = SimpleNamespace(parameter=lambda *args: None,
                            build=lambda: events.append("build"),
                            java=SimpleNamespace(geom=lambda tag: geom))
    solver = object.__new__(FilletChiralSolver)
    solver._prepared_geometry = None
    monkeypatch.setattr(solver, "_ensure_model", lambda: model)
    monkeypatch.setattr(ChiralSolver, "polarizations", lambda self, params: events.append("solve"))
    for freq in (5.5, 5.52):
        solver.polarizations({**FILLET_START, "freq_ghz": freq})
    solver.polarizations({**FILLET_START, "h_mm": 25.1, "freq_ghz": 5.5})
    assert events == ["build", "solve", "solve", "build", "solve"]

    def broken():
        raise RuntimeError("Failed to create fillet")
    model.build = broken
    with pytest.raises(ChiralSolveError, match="geometry build failed"):
        solver.polarizations({**FILLET_START, "h_mm": 25.2, "freq_ghz": 5.5})
    assert events == ["build", "solve", "solve", "build", "solve"]
    assert solver._prepared_geometry is None


def test_geometry_failure_costs_zero_frequency_solves_and_infrastructure_aborts():
    class Broken:
        error = ChiralSolveError("fillet cannot be built")
        def prepare_geometry(self, params):
            raise self.error

        def polarizations(self, params):
            pytest.fail("No spectrum may start after a failed geometry build")

    solver = Broken()
    objective = FilletBandwidthObjective(solver, max_consecutive_failures=3)
    for _ in range(2):
        result = objective.evaluate(FILLET_START)
        assert result.aux["solver_calls_new"] == 0
        assert result.aux["geometry_build_failed"]
        assert "common_band" not in result.aux
    with pytest.raises(RuntimeError, match="Consecutive fillet"):
        objective.evaluate(FILLET_START)
    solver.error = RuntimeError("License server unavailable")
    with pytest.raises(RuntimeError, match="License"):
        objective.evaluate(FILLET_START)


def test_custom_start_reuses_baseline_and_rejects_changed_resume(cli_root, monkeypatch):
    source = cli_root / "runs/baseline"
    original = check_fillet_bandwidth.evaluate(source, "fake", resume=False)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    class NoSolves:
        def polarizations(self, params):
            pytest.fail("The initial spectrum should be reused completely")

    monkeypatch.setattr(run_chiral, "FakeBandwidthSolver", NoSolves)
    args = ["run_chiral", "--objective", "bandwidth", "--fillet", "--max-iters", "1",
            "--start-params", *map(str, FILLET_START.values()), "--run-name", "seeded",
            "--baseline-from", str(source), "--bandwidth-step", "0.02"]
    monkeypatch.setattr(sys, "argv", args)
    run_chiral.main()
    step = json.loads((cli_root / "runs/seeded/steps/step_000.json").read_text())
    assert step["params"] == FILLET_START
    assert step["aux"]["solver_calls_new"] == 0
    assert step["score"] == pytest.approx(original.score)
    assert {p: p.read_bytes() for p in before} == before
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    run_chiral.main()
    # Reusing the same output without fillet checks is not a valid continuation.
    monkeypatch.setattr(sys, "argv", [x for x in args if x != "--fillet"] + ["--resume"])
    with pytest.raises(SystemExit, match="differ"):
        run_chiral.main()


def test_expanded_wrapper_uses_finalist_and_counts_total_proposals(cli_root, monkeypatch):
    source = cli_root / "runs/baseline"
    check_fillet_bandwidth.evaluate(source, "fake", resume=False)
    monkeypatch.setattr(run_fillet_search, "ROOT", cli_root)
    args = ["run_fillet_search", "--solver", "fake", "--proposer", "random",
            "--baseline-from", str(source), "--run-name", "pilot"]
    monkeypatch.setattr(sys, "argv", args)
    run_fillet_search.main()
    run = cli_root / "runs/pilot"
    steps = sorted((run / "steps").glob("step_*.json"))
    assert len(steps) == 6
    assert json.loads(steps[0].read_text())["params"] == FILLET_START
    import yaml
    config = yaml.safe_load((run / "config.yaml").read_text())
    assert config["extra"]["chiral"]["bounds"] == {k: list(v) for k, v in run_fillet_search.BOUNDS.items()}
    monkeypatch.setattr(sys, "argv", args + ["--resume", "--proposals", "7"])
    run_fillet_search.main()
    assert len(list((run / "steps").glob("step_*.json"))) == 8


@pytest.mark.parametrize("extra,match", [
    (["--start-params", "13", "25", "4.6", "8.8", "12"], "Bounds exclude"),
    (["--start-params", "13", "25.001", "4.6", "8.8", "7.3"], "decimal places"),
    (["--start-params", "13", "25", "4.6", "8.8", "7.3", "--random-init"], "another start"),
])
def test_start_is_not_silently_clipped_or_rounded(cli_root, monkeypatch, extra, match):
    monkeypatch.setattr(sys, "argv", ["run_chiral", "--objective", "bandwidth", *extra])
    with pytest.raises(SystemExit, match=match):
        run_chiral.main()
