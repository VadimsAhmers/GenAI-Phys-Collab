"""Cross-run provenance, real prompt delivery and resume, without COMSOL/API."""
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from llm_opt import Config
from src.opt import run_chiral
from src.opt.bandwidth import BandwidthSettings, ChiralBandwidthObjective, MAKSIM_2026_START
from src.opt.bandwidth_reuse import BaselineSpectrum, sha256
from tests.test_bandwidth import LinearBands


@pytest.fixture
def cli_root(tmp_path, monkeypatch):
    root = run_chiral.REPO_ROOT
    for name in run_chiral.BANDWIDTH_CODE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    monkeypatch.setattr(run_chiral, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_chiral, "load_dotenv", lambda: None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_source(path):
    settings = BandwidthSettings()
    result = ChiralBandwidthObjective(LinearBands(), settings, path / "spectra").evaluate(
        MAKSIM_2026_START)
    config = Config(run_name=path.name)
    manifest = {
        "objective": "bandwidth", "both_polarizations": True,
        "solver": "fake", "model_sha256": None,
        "code_sha256": {"src/opt/solvers.py": "test-solver-hash"},
        "settings": asdict(settings),
    }
    config.extra["chiral"] = manifest
    import yaml
    (path / "config.yaml").write_text(yaml.safe_dump(config.to_dict()))
    return manifest, result


def test_reuse_rescores_new_grid_preserves_source_and_exact_geometry(tmp_path):
    source = tmp_path / "source"
    manifest, original = make_source(source)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    baseline = BaselineSpectrum(source, manifest, MAKSIM_2026_START)
    solver = LinearBands()
    objective = ChiralBandwidthObjective(solver, BandwidthSettings(step_ghz=0.02),
                                         tmp_path / "destination", baseline=baseline)
    result = objective.evaluate(MAKSIM_2026_START)
    assert not solver.calls
    assert result.score == pytest.approx(original.score)
    assert result.aux["solver_calls_new"] == 0
    assert result.aux["reused_frequency_points"] == result.aux["n_frequency_points"]
    assert result.aux["n_frequency_points"] < original.aux["n_frequency_points"]
    assert all(r["elapsed_s"] == 0 and r["reused_from"]["sha256"]
               for r in result.aux["spectrum"])
    # Near-identical geometry must not borrow a single physical point.
    changed = {**MAKSIM_2026_START, "h_mm": 24.81}
    second = objective.evaluate(changed)
    assert second.aux["reused_frequency_points"] == 0
    assert len(solver.calls) == second.aux["n_frequency_points"]
    assert {p: p.read_bytes() for p in before} == before
    resumed = objective.evaluate(MAKSIM_2026_START)
    assert resumed.aux["solver_calls_new"] == 0
    assert resumed.aux["spectrum"] == result.aux["spectrum"]


@pytest.mark.parametrize("field,value", [
    ("solver", "comsol"), ("model_sha256", "different-model"),
    ("code_sha256", {"src/opt/solvers.py": "changed-solver"}),
])
def test_reuse_rejects_incompatible_physics(tmp_path, field, value):
    manifest, _ = make_source(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        BaselineSpectrum(tmp_path, {**manifest, field: value}, MAKSIM_2026_START)


def test_reuse_missing_frequency_is_computed_not_interpolated(tmp_path):
    manifest, _ = make_source(tmp_path / "source")
    baseline = BaselineSpectrum(tmp_path / "source", manifest, MAKSIM_2026_START)
    solver = LinearBands()
    result = ChiralBandwidthObjective(
        solver, BandwidthSettings(step_ghz=0.005), baseline=baseline,
    ).evaluate(MAKSIM_2026_START)
    assert solver.calls
    assert not set(solver.calls) & set(baseline.points)
    assert result.aux["solver_calls_new"] == len(solver.calls)
    assert result.aux["reused_frequency_points"] + len(solver.calls) == result.aux["n_frequency_points"]


def test_reuse_detects_corrupted_result_and_model_checksum(tmp_path):
    manifest, _ = make_source(tmp_path)
    path = next(tmp_path.glob("spectra/*/point_*.json"))
    original = path.read_bytes()
    row = json.loads(original)
    row["abs_r_LL_sq"] += 0.1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="values disagree"):
        BaselineSpectrum(tmp_path, manifest, MAKSIM_2026_START)
    path.write_bytes(original)
    (tmp_path / "model_after.sha256").write_text("changed-model model.mph\n")
    with pytest.raises(ValueError, match="disagrees"):
        BaselineSpectrum(tmp_path, manifest, MAKSIM_2026_START)


def test_cli_reuse_usage_resume_and_source_change_guard(cli_root, monkeypatch):
    monkeypatch.setattr(run_chiral, "FakeBandwidthSolver", LinearBands)
    base = ["run_chiral", "--objective", "bandwidth", "--max-iters", "1"]
    monkeypatch.setattr(sys, "argv", base + ["--run-name", "source"])
    run_chiral.main()
    source = cli_root / "runs/source"
    source_hash = sha256(source / "config.yaml")

    class NoSolves:
        def polarizations(self, params):
            pytest.fail("Baseline should be entirely reused")

    monkeypatch.setattr(run_chiral, "FakeBandwidthSolver", NoSolves)
    args = base + ["--run-name", "reused", "--bandwidth-step", "0.02",
                   "--baseline-from", str(source)]
    monkeypatch.setattr(sys, "argv", args)
    run_chiral.main()
    destination = cli_root / "runs/reused"
    usage = json.loads((destination / "solver_usage.json").read_text())
    assert usage["completed_solver_calls"] == usage["solver_elapsed_s"] == 0
    assert usage["reused_frequency_points"] > 0
    assert sha256(source / "config.yaml") == source_hash
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    run_chiral.main()
    assert json.loads((destination / "solver_usage.json").read_text()) == usage
    monkeypatch.setattr(sys, "argv", ["run_chiral", "--run-name", "reused", "--finalize"])
    run_chiral.main()
    assert json.loads((destination / "solver_usage.json").read_text()) == usage
    # Even a source metadata change must not silently alter an existing run.
    with (source / "config.yaml").open("a") as f:
        f.write("\n# changed since import\n")
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    with pytest.raises(SystemExit, match="differ"):
        run_chiral.main()


def test_recent_failed_spectra_reach_actual_llm_request_after_resume(cli_root, monkeypatch):
    import llm_opt.llm

    class VariedPeaks(LinearBands):
        def polarizations(self, params):
            if params["h_mm"] == 24.8:
                return super().polarizations(params)
            peak = 0.17 if params["h_mm"] == 10 else 0.63
            return {"abs_r_RR_sq": peak, "abs_r_LL_sq": peak - 0.01}

    proposals = iter([{**MAKSIM_2026_START, "h_mm": h} for h in (10, 26, 25)])
    requests = []

    def create(**kwargs):
        requests.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(proposals))))],
            usage=None,
        )

    monkeypatch.setattr(llm_opt.llm, "OpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    monkeypatch.setattr(run_chiral, "FakeBandwidthSolver", VariedPeaks)
    args = ["run_chiral", "--objective", "bandwidth", "--proposer", "llm",
            "--run-name", "prompt_test", "--bandwidth-step", "0.02"]
    monkeypatch.setattr(sys, "argv", args + ["--max-iters", "3"])
    run_chiral.main()
    monkeypatch.setattr(sys, "argv", args + ["--resume", "--max-iters", "4"])
    run_chiral.main()
    prompt = requests[-1][-1]["content"]
    assert "sampled peak=0.1700" in prompt and "sampled peak=0.6300" in prompt
    assert "sampled peak=0.1600" in prompt and "sampled peak=0.6200" in prompt
    assert "no effect" not in prompt
    assert "not extra objective terms" in prompt
    for step in (1, 2):
        record = json.loads((cli_root / f"runs/prompt_test/steps/step_{step:03d}.json").read_text())
        assert record["score"] == 1.0
    saved = json.loads((cli_root / "runs/prompt_test/steps/step_003.json").read_text())
    assert saved["llm"]["messages"] == requests[-1]
