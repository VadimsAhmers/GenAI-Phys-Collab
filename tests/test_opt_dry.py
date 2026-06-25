"""Dry tests for the multipole-channel optimizer bindings (no COMSOL, no API)."""
import numpy as np
import pytest

from src.opt import (
    FakeSolver,
    MultipoleChannelObjective,
    RadiiParametrization,
    channel_powers,
    multipole_loss,
    validate_target,
)


def _perfect_coeffs(channel="ED"):
    """Coefficients with all power in a single channel."""
    c = {kind: {n: {m: 0j for m in range(-n, n + 1)} for n in range(1, 5)}
         for kind in ("a", "b")}
    kind, n = ("a", 1) if channel == "ED" else ("b", 1)
    c[kind][n][0] = 1 + 0j
    return c


def test_multipole_loss_is_zero_for_pure_channel():
    loss, info = multipole_loss(_perfect_coeffs("ED"), {"ED": 1.0})
    assert loss < 1e-12
    assert info["q"]["ED"] > 0.999


def test_channel_powers_sum_matches():
    P = channel_powers(_perfect_coeffs("ED"))
    assert set(P) == {"ED", "EQ", "EO", "EH", "MD", "MQ", "MO", "MH"}
    assert abs(sum(P.values()) - 1.0) < 1e-12


def test_parametrization_roundtrip_and_validation():
    param = RadiiParametrization(n=10)
    payload = param.schema_hint()
    decoded = param.decode_llm(payload)
    assert len(decoded["radii_nm"]) == 10
    assert param.validate(decoded) == []
    # out-of-bounds rejected
    bad = {"radii_nm": [1000.0] * 10}
    assert param.validate(bad)
    # wrong length rejected
    assert param.validate({"radii_nm": [200.0] * 3})


def test_objective_with_fake_solver_runs():
    obj = MultipoleChannelObjective(FakeSolver(), target={"ED": 1.0})
    param = RadiiParametrization()
    res = obj.evaluate(param.initial_params())
    assert res.score >= 0.0
    assert "ED" in res.aux["q"]
    assert "power shares" in res.feedback


def test_validate_target_rejects_bad_input():
    validate_target({"ED": 1.0})  # ok
    with pytest.raises(ValueError):
        validate_target({})  # empty
    with pytest.raises(ValueError):
        validate_target({"Ed": 1.0})  # wrong case / unknown channel
    with pytest.raises(ValueError):
        validate_target({"ED": 0.0})  # non-positive sum


def test_objective_rejects_unknown_channel():
    with pytest.raises(ValueError):
        MultipoleChannelObjective(FakeSolver(), target={"XX": 1.0})


def test_fake_solver_is_deterministic():
    solver = FakeSolver()
    r = np.array([200.0] * 10)
    assert solver(r)["a"][1][0] == solver(r)["a"][1][0]
