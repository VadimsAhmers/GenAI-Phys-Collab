"""Spectrum summary + the frequency path of the fake chiral solver (no COMSOL)."""
import numpy as np

from src.opt import FakeChiralSolver, spectrum_summary, format_summary, CHIRAL_REFERENCE


def lorentzian(f, f0=5.8, fwhm=0.15, height=0.9):
    return height / (1.0 + ((f - f0) / (fwhm / 2.0)) ** 2)


def test_summary_recovers_peak_and_width():
    freqs = np.arange(5.3, 6.31, 0.025)
    s = spectrum_summary(freqs, [lorentzian(f) for f in freqs])
    assert abs(s["peak_ghz"] - 5.8) < 0.005
    assert abs(s["peak_value"] - 0.9) < 0.01
    assert abs(s["fwhm_ghz"] - 0.15) < 0.01
    assert abs(s["q"] - 5.8 / 0.15) < 3
    assert not s["at_edge"]


def test_summary_flags_a_peak_outside_the_band():
    # Band ends before the resonance: report no width rather than invent one.
    freqs = np.arange(5.0, 5.51, 0.05)
    s = spectrum_summary(freqs, [lorentzian(f) for f in freqs])
    assert s["at_edge"]
    assert s["fwhm_ghz"] is None
    assert s["q"] is None
    assert "not bracketed" in format_summary(s)


def test_summary_is_order_independent():
    freqs = [5.9, 5.7, 5.8, 6.0, 5.6]
    s = spectrum_summary(freqs, [lorentzian(f) for f in freqs])
    assert abs(s["peak_ghz"] - 5.8) < 0.02


def test_fake_solver_frequency_response_is_a_resonance():
    solver = FakeChiralSolver()
    geom = {k: v for k, v in CHIRAL_REFERENCE.items() if k != "eps_r"}
    freqs = np.arange(5.4, 6.21, 0.02)
    values = [solver({**geom, "freq_ghz": f}) for f in freqs]
    s = spectrum_summary(freqs, values)
    assert abs(s["peak_ghz"] - 5.8) < 0.02
    # Smaller particle -> higher resonance frequency (the 1/size scaling the
    # real sweep is meant to test).
    small = [solver({**geom, "r_mm": 5.0, "freq_ghz": f}) for f in freqs]
    assert spectrum_summary(freqs, small)["peak_ghz"] > s["peak_ghz"]
    # No freq key -> unchanged single-frequency behaviour.
    assert solver(geom) == solver({**geom})
