"""Figure: the three things the overnight frequency campaign measured.

Reads the CSVs written by `sweep_freq.py --out` (no COMSOL needed).

Three panels, shared y in |r_RR|^2, because the three questions behind the band
pivot are answered by three different comparisons and one axis cannot carry them:

  A  the reference lineshape at 5 MHz -- is it tuned, and how narrow really?
  B  the size family -- does f_res go as 1/a, and does the peak HEIGHT follow?
  C  reference vs the eps<=10 crescent -- who is mistuned, and by how much?

Usage:
    uv run python scripts/plot_freq.py [RUNDIR] [-o out.png]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated palette (see scripts/plot_sweep.py): categorical slots 1-2 for the
# two named geometries in panel C, the blue sequential ramp for the size family
# in panel B -- size is a magnitude, so it gets a ramp, not six identities.
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"
RAMP = ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"

DESIGN_FREQ = 5.8  # GHz -- the single point the objective currently scores


def load(path: Path) -> dict[tuple, list[tuple[float, float]]]:
    """{(geom label, scale): [(f, |r_RR|^2), ...]}, each spectrum sorted by f."""
    out: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            key = (row["geom"], row["scale"])
            out[key].append((float(row["freq_ghz"]), float(row["abs_r_RR_sq"])))
    for points in out.values():
        points.sort()
    return out


def peak(points):
    """Sub-grid peak (f, value) by a parabola through the maximum sample."""
    i = max(range(len(points)), key=lambda j: points[j][1])
    if i in (0, len(points) - 1):
        return points[i]
    (x0, y0), (x1, y1), (x2, y2) = points[i - 1:i + 2]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return x1, y1
    return x1 + 0.5 * (y0 - y2) / denom * (x1 - x0), y1 - 0.125 * (y0 - y2) ** 2 / denom


def fwhm(points, f_peak, v_peak):
    """Width in MHz between the half-peak crossings bracketing f_peak, or None."""
    half, left, right = v_peak / 2, None, None
    for (fa, ya), (fb, yb) in zip(points, points[1:]):
        if ya < half <= yb and fb <= f_peak:
            left = fa + (half - ya) / (yb - ya) * (fb - fa)
        if ya >= half > yb and fa >= f_peak:
            right = fa + (half - ya) / (yb - ya) * (fb - fa)
            break
    return None if left is None or right is None else (right - left) * 1000


def style(ax, xlabel=True):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 1.0)
    ax.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    if xlabel:
        ax.set_xlabel("частота, ГГц", color=TEXT_SECONDARY, fontsize=9.5)


def mark_design_freq(ax):
    ax.axvline(DESIGN_FREQ, color=TEXT_MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
               zorder=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", nargs="?", default="runs/freq")
    ap.add_argument("-o", "--out", default="notes/freq_campaign.png")
    args = ap.parse_args()

    d = Path(args.rundir)
    ref = load(d / "freq_reference.csv")[("base", "1.0")]
    scales = load(d / "freq_scale.csv")
    legacy = load(d / "freq_legacy.csv")[("base", "1.0")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.14, wspace=0.16)

    # --- A: the reference lineshape --------------------------------------
    ax = axes[0]
    style(ax)
    mark_design_freq(ax)
    ax.plot([f for f, _ in ref], [y for _, y in ref], color=SERIES, linewidth=2,
            solid_capstyle="round")
    fp, vp = peak(ref)
    w = fwhm(ref, fp, vp)
    ax.plot([fp - w / 2000, fp + w / 2000], [vp / 2, vp / 2], color=TEXT_MUTED,
            linewidth=1.5, solid_capstyle="butt")
    ax.annotate(f"FWHM {w:.0f} МГц\nQ ≈ {fp * 1000 / w:.0f}",
                xy=(fp + w / 2000, vp / 2), xytext=(10, -4), textcoords="offset points",
                color=TEXT_SECONDARY, fontsize=9.5, va="center")
    ax.annotate(f"пик {vp:.3f} @ {fp:.4f} ГГц", xy=(fp, vp), xytext=(12, -2),
                textcoords="offset points", color=TEXT_PRIMARY, fontsize=9.5)
    ax.set_ylabel("|r$_{RR}$|²", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title("A · опорная точка настроена точно на 5.800 ГГц",
                 color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)

    # --- B: the size family ----------------------------------------------
    ax = axes[1]
    style(ax)
    mark_design_freq(ax)
    family = sorted(((float(s), pts) for (_, s), pts in scales.items()),
                    key=lambda t: t[0])
    family = sorted(family + [(1.0, ref)])
    for color, (a, pts) in zip(RAMP, family):
        # The reference spectrum spans a full GHz; clip it to the same +-300 MHz
        # window the scaled points were measured on, so the six lines compare.
        fp_a, _ = peak(pts)
        win = [(f, y) for f, y in pts if abs(f - fp_a) <= 0.3]
        ax.plot([f for f, _ in win], [y for _, y in win], color=color, linewidth=2,
                solid_capstyle="round")
        fp_a, vp_a = peak(win)
        ax.annotate(f"×{a:g}", xy=(fp_a, vp_a), xytext=(0, 6),
                    textcoords="offset points", color=color, fontsize=9,
                    ha="center", fontweight="bold")
    ax.set_title("B · f$_{рез}$ ∝ 1/a с точностью 0.08 %, но высота пика — нет",
                 color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)

    # --- C: reference vs the eps<=10 crescent ------------------------------
    ax = axes[2]
    style(ax)
    mark_design_freq(ax)
    ax.plot([f for f, _ in ref], [y for _, y in ref], color=SERIES, linewidth=2,
            solid_capstyle="round")
    ax.plot([f for f, _ in legacy], [y for _, y in legacy], color=SERIES_2,
            linewidth=2, solid_capstyle="round")
    lp, lv = peak(legacy)
    at_design = next(y for f, y in legacy if abs(f - DESIGN_FREQ) < 1e-6)
    ax.annotate(f"ε=10: пик {lv:.3f}\n@ {lp:.3f} ГГц", xy=(lp, lv), xytext=(6, 8),
                textcoords="offset points", color=SERIES_2, fontsize=9.5)
    ax.annotate(f"ε=80\n{vp:.3f}", xy=(fp, vp), xytext=(-34, -2),
                textcoords="offset points", color=SERIES, fontsize=9.5)
    ax.annotate("", xy=(DESIGN_FREQ, lv), xytext=(DESIGN_FREQ, at_design),
                arrowprops=dict(arrowstyle="<->", color=TEXT_SECONDARY, linewidth=1.2))
    ax.annotate(f"+{lv - at_design:.3f} даром", xy=(DESIGN_FREQ, (lv + at_design) / 2),
                xytext=(-8, 0), textcoords="offset points", color=TEXT_PRIMARY,
                fontsize=9.5, ha="right", va="center")
    ax.set_title("C · низкий ε расстроен: он шире и выше, но не на 5.8",
                 color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)

    fig.suptitle("Частотный отклик хирального зеркала: 543 решения COMSOL, ночь 10–11.08.2026",
                 color=TEXT_PRIMARY, fontsize=13, x=0.05, ha="left", y=0.955)
    fig.text(0.05, 0.885,
             "пунктир — рабочая точка 5.800 ГГц, на которой считается текущая целевая функция",
             color=TEXT_MUTED, fontsize=9.5, ha="left")

    out = Path(args.out)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
