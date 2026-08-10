"""Summarize a sampled |r_RR|^2(f) spectrum into peak / linewidth numbers.

Kept separate from the sweep script because the SAME summary is what the LLM
will eventually be fed per candidate geometry: Maksim T. (2026-07-30) proposed
optimizing over a frequency band instead of the single 5.8 GHz point, and the
value of that is not "catch the peak somewhere" but that the peak FREQUENCY
gives the optimizer a direction (resonance at 6.4 GHz, want 5.8 -> grow the
structure ~10%), which one scalar cannot.

Everything here is pure arithmetic on sampled points -- no COMSOL, no physics
model. A sampled spectrum can only bound the true peak: with a linewidth of
~150 MHz and a 50 MHz grid, `peak_ghz` is good to a few MHz but `fwhm_ghz` is
only as good as the two half-max brackets, so treat it as an estimate and
report `at_edge` / a missing `fwhm_ghz` honestly rather than extrapolating.
"""
from __future__ import annotations


def spectrum_summary(freqs_ghz, values) -> dict:
    """Peak height/position and FWHM of a sampled spectrum.

    Returns a dict with:
      peak_value  -- the largest SAMPLED value (never interpolated upward)
      peak_ghz    -- peak position, parabola-interpolated from the top 3 points
      peak_index  -- index of the largest sampled value
      fwhm_ghz    -- full width at half the peak value, or None if unbracketed
      q           -- peak_ghz / fwhm_ghz, or None
      at_edge     -- True when the maximum sits on the first/last sample, i.e.
                     the real peak is probably outside the scanned band
    """
    points = sorted(zip((float(f) for f in freqs_ghz), (float(v) for v in values)))
    if not points:
        raise ValueError("empty spectrum")
    fs = [f for f, _ in points]
    vs = [v for _, v in points]

    imax = max(range(len(vs)), key=vs.__getitem__)
    peak_value = vs[imax]
    peak_ghz = fs[imax]

    # Parabola through the three top samples -- on a smooth resonance this is
    # accurate to a small fraction of the grid step. Guarded against a flat or
    # inverted triple (denominator ~ 0) and clamped to the bracketing interval
    # so noise can never throw the vertex outside the sampled points.
    if 0 < imax < len(fs) - 1:
        f0, f1, f2 = fs[imax - 1], fs[imax], fs[imax + 1]
        v0, v1, v2 = vs[imax - 1], vs[imax], vs[imax + 1]
        denom = (f0 - f1) * (f0 - f2) * (f1 - f2)
        if abs(denom) > 1e-12:
            a = (f2 * (v1 - v0) + f1 * (v0 - v2) + f0 * (v2 - v1)) / denom
            b = (
                f2 * f2 * (v0 - v1) + f1 * f1 * (v2 - v0) + f0 * f0 * (v1 - v2)
            ) / denom
            if a < 0:  # a real maximum, not a minimum or a straight line
                vertex = -b / (2 * a)
                if f0 <= vertex <= f2:
                    peak_ghz = vertex

    half = peak_value / 2.0
    left = _cross(fs, vs, imax, half, step=-1)
    right = _cross(fs, vs, imax, half, step=+1)
    fwhm = None if left is None or right is None else right - left
    q = peak_ghz / fwhm if fwhm else None

    return {
        "peak_value": peak_value,
        "peak_ghz": peak_ghz,
        "peak_index": imax,
        "fwhm_ghz": fwhm,
        "q": q,
        "at_edge": imax in (0, len(fs) - 1),
        "f_min": fs[0],
        "f_max": fs[-1],
        "n_points": len(fs),
    }


def _cross(fs: list[float], vs: list[float], imax: int, level: float, step: int):
    """Frequency where the spectrum first drops through `level` away from the peak.

    Walks outward from the maximum and linearly interpolates the crossing.
    Returns None when the spectrum never falls below `level` on that side --
    the band was too narrow to bracket the line, and a width taken from the
    band edge would be a fabricated number.
    """
    i = imax
    while 0 <= i + step < len(vs):
        j = i + step
        if vs[j] <= level:
            if vs[i] == vs[j]:
                return fs[j]
            t = (vs[i] - level) / (vs[i] - vs[j])
            return fs[i] + t * (fs[j] - fs[i])
        i = j
    return None


def format_summary(summary: dict) -> str:
    """One-line human/LLM-readable rendering of `spectrum_summary`."""
    parts = [
        f"peak |r_RR|^2 = {summary['peak_value']:.4f}",
        f"at {summary['peak_ghz']:.4f} GHz",
    ]
    if summary["fwhm_ghz"] is not None:
        parts.append(
            f"FWHM {summary['fwhm_ghz'] * 1000:.0f} MHz (Q ~ {summary['q']:.0f})"
        )
    else:
        parts.append("FWHM not bracketed by the scanned band")
    if summary["at_edge"]:
        parts.append(
            f"WARNING: maximum sits on the band edge "
            f"({summary['f_min']:g}-{summary['f_max']:g} GHz)"
        )
    return "; ".join(parts)
