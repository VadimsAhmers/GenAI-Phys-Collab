"""Launch/recovery checks using synthetic spectra; no COMSOL or API calls."""
import json

import pytest

from scripts import check_fillet_bandwidth as check
from src.opt.bandwidth_reuse import BaselineSpectrum
from src.opt.solvers import FakeBandwidthSolver


def test_interrupted_run_resumes_and_can_supply_a_future_baseline(tmp_path, monkeypatch):
    class Interrupted(FakeBandwidthSolver):
        def __init__(self):
            self.calls = 0

        def polarizations(self, params):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("simulated interruption")
            return super().polarizations(params)

    monkeypatch.setattr(check, "FakeBandwidthSolver", Interrupted)
    assert check.main(["--solver", "fake", "--out", str(tmp_path)]) == 1
    assert (tmp_path / "exit_code.txt").read_text().strip() == "1"
    assert len(list(tmp_path.glob("spectra/*/point_*.json"))) == 3
    assert not (tmp_path / "summary.json").exists()

    monkeypatch.setattr(check, "FakeBandwidthSolver", FakeBandwidthSolver)
    assert check.main(["--solver", "fake", "--out", str(tmp_path), "--resume"]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["solver_calls_new"] == summary["n_frequency_points"] - 3
    assert summary["common_band"]["width_ghz"] > 0
    assert (tmp_path / "exit_code.txt").read_text().strip() == "0"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    baseline = BaselineSpectrum(tmp_path, manifest, check.PARAMS)
    assert len(baseline.points) == summary["n_frequency_points"]
    # A completed resume must not spend any additional solver calls.
    result = check.evaluate(tmp_path, "fake", resume=True)
    assert result.aux["solver_calls_new"] == 0
    assert result.score == summary["score"]


def test_resume_rejects_changed_inputs_and_new_run_refuses_overwrite(tmp_path, monkeypatch):
    check.evaluate(tmp_path, "fake", resume=False)
    before = (tmp_path / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        check.evaluate(tmp_path, "fake", resume=False)
    monkeypatch.setattr(check, "PARAMS", {**check.PARAMS, "eps_r": 7.4})
    with pytest.raises(ValueError, match="inputs changed"):
        check.evaluate(tmp_path, "fake", resume=True)
    assert (tmp_path / "summary.json").read_bytes() == before


def test_wrong_model_is_rejected_before_comsol_or_output_creation(tmp_path, monkeypatch):
    wrong_model = tmp_path / "wrong.mph"
    wrong_model.write_bytes(b"not the inspected model")
    monkeypatch.setattr(check, "MODEL", wrong_model)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="Model differs"):
        check.evaluate(out, "comsol", resume=False)
    assert not out.exists()
