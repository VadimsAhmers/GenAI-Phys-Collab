"""Half-width analysis of the reference-point sweep.

Estimates each axis's width at half the peak |r_RR|^2 by linear interpolation
between the sampled points.

These are ONE-DIMENSIONAL widths through one point. They must not be multiplied
into a 5-D "volume of the good region": that assumes the parameters are
independent, and the run's own history refutes it -- see the correction block in
notes/chiral-sweep-2026-07-29.md.

Reads a CSV written by `sweep_chiral.py --out`; no COMSOL needed.

Usage:
    uv run python scripts/analyze_sweep.py [sweep.csv]
"""
import csv
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "notes/sweep_reference.csv"

axes = defaultdict(list)
base = {}
with open(CSV) as fh:
    for row in csv.DictReader(fh):
        if row["status"] != "ok":
            continue
        v, y = float(row["value"]), float(row["abs_r_RR_sq"])
        axes[row["param"]].append((v, y))
        if row["is_base"] == "True":
            base[row["param"]] = (v, y)

PEAK = base["r_mm"][1]
HALF = PEAK / 2


def crossing(points, x0, direction):
    """Where |r_RR|^2 falls through half-max walking away from the base."""
    seq = [p for p in points if (p[0] > x0 if direction > 0 else p[0] < x0)]
    seq.sort(key=lambda p: abs(p[0] - x0))
    prev = (x0, PEAK)
    for x, y in seq:
        if y < HALF <= prev[1]:
            t = (prev[1] - HALF) / (prev[1] - y)
            return prev[0] + t * (x - prev[0])
        prev = (x, y)
    return None


print(f"peak |r_RR|^2 = {PEAK:.4f}   half-max = {HALF:.4f}\n")
print(f"{'param':>9} {'base':>7} {'lo':>7} {'hi':>7} {'width':>7} {'rel':>7}")
for p, (x0, _y0) in base.items():
    lo = crossing(axes[p], x0, -1)
    hi = crossing(axes[p], x0, +1)
    fwhm = hi - lo
    print(f"{p:>9} {x0:7.3f} {lo:7.3f} {hi:7.3f} {fwhm:7.3f} {fwhm / x0:6.1%}")

# Bandwidth proxy: at fixed geometry the resonance tracks 1/sqrt(eps), so a
# fractional eps width maps to half that fractional frequency width.
eps_fwhm_rel = (crossing(axes["eps_r"], 80.0, +1)
                - crossing(axes["eps_r"], 80.0, -1)) / 80.0
f0 = 5.8
print(f"\neps FWHM = {eps_fwhm_rel:.1%} -> df/f ~ {eps_fwhm_rel / 2:.1%} "
      f"-> ~{f0 * eps_fwhm_rel / 2 * 1000:.0f} MHz at {f0} GHz, Q ~ "
      f"{1 / (eps_fwhm_rel / 2):.0f}")

print("\noff-peak landscape (highest non-base points per axis):")
for p in axes:
    off = sorted((y, v) for v, y in axes[p] if abs(v - base[p][0]) > 1e-3)
    print(f"  {p:>9}: " + ", ".join(f"{v:g}->{y:.3f}" for y, v in off[-3:][::-1]))
