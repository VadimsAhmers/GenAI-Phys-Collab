"""Threshold geometry, partial-solve recovery and CLI compatibility without COMSOL/API."""
import json
import math
import sys

import pytest

from src.opt.bandwidth import (
    BandwidthSettings, ChiralBandwidthObjective, MAKSIM_2026_START, MAKSIM_2026_BOUNDS,
)
from src.opt.optimize import ChiralParametrization
from src.opt.solvers import ChiralSolveError, FakeBandwidthSolver
from src.opt.spectrum import threshold_band_summary
from src.opt import run_chiral


def test_disjoint_bands_are_not_summed_and_not_tied_to_peak():
    # Taller first peak, but widest band is the second component.
    result = threshold_band_summary([0, 1, 2, 3, 4, 5, 6],
                                    [0, 1, 0, 0.9, 0.9, 0.9, 0])
    assert len(result["intervals_ghz"]) == 2
    assert result["width_ghz"] == pytest.approx(2 + 2 / 9)
    assert result["widest_interval_ghz"][0] > 2


def test_interpolate_channels_before_intersecting_not_min_samples():
    # Neither endpoint passes BOTH, but the two interpolated channels overlap.
    result = threshold_band_summary([5.5, 6.0], [0.7, 1], [1, 0.7])
    assert result["widest_interval_ghz"] == pytest.approx([5.5 + 0.5 / 3, 6 - 0.5 / 3])
    assert result["width_ghz"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("values,width,left,right", [
    ([0.8, 0.8, 0.8], 0.5, True, True),
    ([0.9, 0.9, 0.7], 0.375, True, False),
    ([0.7, 0.9, 0.9], 0.375, False, True),
    ([0.7, 0.8, 0.7], 0.0, False, False),
    ([0.7, 0.7, 0.7], 0.0, False, False),
])
def test_threshold_equality_and_window_edges(values, width, left, right):
    result = threshold_band_summary([5.5, 5.75, 6], values)
    assert result["width_ghz"] == pytest.approx(width)
    assert result["touches_window_left"] == left
    assert result["touches_window_right"] == right


@pytest.mark.parametrize("fs,values", [
    ([5.5], [0.9]), ([5.5, 5.5], [0.9, 0.9]),
    ([5.5, 6], [0.9]), ([5.5, 6], [math.nan, 0.9]),
])
def test_bad_spectra_raise(fs, values):
    with pytest.raises(ValueError):
        threshold_band_summary(fs, values)


class LinearBands:
    """Known intersections: RR in [5.6,5.9], LL in [5.7,6.0]."""
    def __init__(self, fail_at=None, error=RuntimeError):
        self.calls = []
        self.fail_at = fail_at
        self.error = error

    def polarizations(self, params):
        f = params["freq_ghz"]
        self.calls.append(f)
        if len(self.calls) == self.fail_at:
            raise self.error("injected failure")
        return {"abs_r_RR_sq": 0.95 - abs(f - 5.75),
                "abs_r_LL_sq": 0.95 - abs(f - 5.85)}


def settings():
    return BandwidthSettings(step_ghz=0.05, edge_tolerance_ghz=0.0025)


def test_objective_width_score_feedback_counts_and_persistent_resume(tmp_path):
    solver = LinearBands()
    result = ChiralBandwidthObjective(solver, settings(), tmp_path).evaluate(MAKSIM_2026_START)
    assert result.aux["common_band"]["width_ghz"] == pytest.approx(0.2)
    assert result.score == pytest.approx(0.6)
    assert len(set(solver.calls)) == len(solver.calls) == result.aux["solver_calls_new"]
    assert result.aux["channels"]["abs_r_RR_sq"]["band"]["width_ghz"] == pytest.approx(0.3)
    assert "sampled peak" in result.feedback
    points = result.aux["spectrum"]
    for a, b in zip(points, points[1:]):
        if any((a[c] >= 0.8) != (b[c] >= 0.8) for c in ("abs_r_RR_sq", "abs_r_LL_sq")):
            assert b["freq_ghz"] - a["freq_ghz"] <= 0.0025 + 1e-12
    fresh = LinearBands()
    resumed = ChiralBandwidthObjective(fresh, settings(), tmp_path).evaluate(MAKSIM_2026_START)
    assert fresh.calls == []
    assert resumed.score == result.score
    assert resumed.aux["spectrum"] == points


def test_resume_partial_geometry_reuses_completed_frequencies(tmp_path):
    broken = LinearBands(fail_at=4)
    with pytest.raises(RuntimeError, match="injected"):
        ChiralBandwidthObjective(broken, settings(), tmp_path).evaluate(MAKSIM_2026_START)
    assert len(list(tmp_path.glob("*/point_*.json"))) == 3
    solver = LinearBands()
    result = ChiralBandwidthObjective(solver, settings(), tmp_path).evaluate(MAKSIM_2026_START)
    assert not set(broken.calls[:3]) & set(solver.calls)
    assert result.score == pytest.approx(0.6)
    assert len(list(tmp_path.glob("*/failure_*.json"))) == 1


def test_geometry_failure_has_no_band_and_does_not_bridge_partial_spectrum(tmp_path):
    objective = ChiralBandwidthObjective(
        LinearBands(fail_at=5, error=ChiralSolveError), settings(), tmp_path)
    result = objective.evaluate(MAKSIM_2026_START)
    assert result.score > 1
    assert result.aux["solve_failed"]
    assert "common_band" not in result.aux
    assert len(result.aux["spectrum"]) == 4
    always_fails = ChiralBandwidthObjective(
        LinearBands(fail_at=1, error=ChiralSolveError), max_consecutive_failures=1)
    with pytest.raises(RuntimeError, match="Consecutive"):
        always_fails.evaluate(MAKSIM_2026_START)


@pytest.mark.parametrize("value", [float("nan"), -0.1])
def test_invalid_solver_values_do_not_become_zero_width(value):
    class Broken:
        def polarizations(self, params):
            return {"abs_r_RR_sq": 0.9, "abs_r_LL_sq": value}
    with pytest.raises(ValueError):
        ChiralBandwidthObjective(Broken()).evaluate(MAKSIM_2026_START)


def test_october_start_bounds_and_synthetic_positive_band():
    param = ChiralParametrization(bounds=MAKSIM_2026_BOUNDS, optimize_eps=True,
                                  start_params=MAKSIM_2026_START, frequency_window=(5.5, 6))
    assert param.initial_params() == MAKSIM_2026_START
    assert param.validate(param.initial_params()) == []
    assert "design frequency" not in param.bounds_description()
    assert "5.5–6" in param.bounds_description()
    result = ChiralBandwidthObjective(FakeBandwidthSolver()).evaluate(param.initial_params())
    assert 0 < result.score < 1
    assert result.aux["channels"]["abs_r_RR_sq"] != result.aux["channels"]["abs_r_LL_sq"]


def test_no_band_is_valid_score_one_and_no_hidden_peak_objective():
    class Low:
        def polarizations(self, params):
            return {"abs_r_RR_sq": 0.79, "abs_r_LL_sq": 0.1}
    result = ChiralBandwidthObjective(Low()).evaluate(MAKSIM_2026_START)
    assert result.score == 1
    assert result.aux["common_band"]["width_ghz"] == 0
    assert not result.aux["solve_failed"]


def test_cli_fake_random_resume_finalize_and_mismatch_guard(tmp_path, monkeypatch):
    # Keep all run artifacts outside the real project. Source copies are only
    # for the CLI's provenance hashes; imports still use the actual code.
    root = run_chiral.REPO_ROOT
    for relative in run_chiral.BANDWIDTH_CODE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    monkeypatch.setattr(run_chiral, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    base = ["run_chiral", "--objective", "bandwidth", "--solver", "fake",
            "--proposer", "random", "--run-name", "band_test", "--max-iters", "3"]
    monkeypatch.setattr(sys, "argv", base)
    run_chiral.main()
    run = tmp_path / "runs/band_test"
    first = json.loads((run / "steps/step_000.json").read_text())
    assert first["params"] == MAKSIM_2026_START
    assert first["aux"]["common_band"]["width_ghz"] > 0
    assert (run / "bandwidth.csv").exists()
    assert (run / "plots/bandwidth_peaks.png").exists()
    original = {p: p.read_bytes() for p in (run / "spectra").glob("*/point_*.json")}
    monkeypatch.setattr(sys, "argv", base + ["--resume"])
    run_chiral.main()
    assert {p: p.read_bytes() for p in original} == original
    monkeypatch.setattr(sys, "argv", base + ["--resume", "--reflection-level", "0.9"])
    with pytest.raises(SystemExit, match="differ"):
        run_chiral.main()
    monkeypatch.setattr(sys, "argv", ["run_chiral", "--run-name", "band_test", "--finalize"])
    run_chiral.main()
    assert {p: p.read_bytes() for p in original} == original
    assert json.loads((run / "solver_usage.json").read_text())["completed_solver_calls"] == len(original)
    assert (run / "best/spectrum.csv").exists()


def test_cli_can_resume_interrupted_first_spectrum_without_any_steps(tmp_path, monkeypatch):
    root = run_chiral.REPO_ROOT
    for relative in run_chiral.BANDWIDTH_CODE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    monkeypatch.setattr(run_chiral, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    broken = LinearBands(fail_at=4)
    monkeypatch.setattr(run_chiral, "FakeBandwidthSolver", lambda: broken)
    args = ["run_chiral", "--objective", "bandwidth", "--run-name", "partial",
            "--max-iters", "1"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(RuntimeError, match="injected"):
        run_chiral.main()
    run = tmp_path / "runs/partial"
    assert not list((run / "steps").glob("*.json"))
    saved_config = (run / "config.yaml").read_bytes()
    # Even the legacy/default point mode may not overwrite this partial run.
    monkeypatch.setattr(sys, "argv", ["run_chiral", "--run-name", "partial"])
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        run_chiral.main()
    assert (run / "config.yaml").read_bytes() == saved_config
    fresh = LinearBands()
    monkeypatch.setattr(run_chiral, "FakeBandwidthSolver", lambda: fresh)
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    run_chiral.main()
    assert not set(fresh.calls) & set(broken.calls[:3])
    rec = json.loads((run / "steps/step_000.json").read_text())
    assert rec["score"] == pytest.approx(0.6)
    usage = json.loads((run / "solver_usage.json").read_text())
    assert usage["completed_solver_calls"] == rec["aux"]["n_frequency_points"] + 1


@pytest.mark.parametrize("extra", [
    ["--bound", "r_mm", "4", "14"],
    ["--bound", "eps_r", "5", "11"],
    ["--bound", "h_mm", "2", "20"],
    ["--start-reference"],
])
def test_cli_rejects_hardware_violation_or_silent_start_clipping(extra, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_chiral", "--objective", "bandwidth"] + extra)
    with pytest.raises(SystemExit):
        run_chiral.main()
