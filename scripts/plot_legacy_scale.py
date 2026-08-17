"""Figure: what happens to the eps<=10 crescent when its size is scaled.

Reads `freq_legacy.csv` (the a = 1.0 anchor, from the overnight campaign) and
`freq_legacy_scale.csv` (the follow-up). No COMSOL needed.

Two panels, because the follow-up answered its own question with a no and then
raised a better one:

  A  the five spectra -- the x1.024 rescale was supposed to put the 0.899 peak on
     5.8 GHz and did not, because f_res does not go as 1/a for this structure.
  B  peak height against how much of the guide cross-section the disk fills,
     with the eps=80 family beside it -- the trend that produced 0.980, and the
     reason it cannot be reached at 5.8 GHz with the guide fixed at 32 mm.

Usage:
    uv run python scripts/plot_legacy_scale.py [RUNDIR] [-o out.png]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = "#2a78d6"       # eps = 10 (this run)
SERIES_2 = "#eb6834"     # eps = 80 (the reference family, for contrast)
# Blue sequential ramp, steps 250-700: 250 is the lightest that still clears
# 2:1 against the light surface -- anything lighter disappears on the page.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"

DESIGN_FREQ = 5.8
A_WG_MM = 32.0
R_LEGACY_MM = 14.7       # a = 1.0 disk radius
R_REF_MM = 6.0           # the compact eps = 80 reference

# Peak heights of the eps=80 size family, from freq_scale.csv + freq_reference.csv
# (scripts/plot_freq.py panel B). Hard-coded rather than re-derived: this panel
# only needs the trend line, and the numbers are in notes/chiral-freq-2026-08-11.md.
EPS80 = {0.93: 0.8049, 0.96: 0.8465, 1.00: 0.8861, 1.04: 0.8946, 1.08: 0.8783,
         1.12: 0.8438}


def load(path: Path) -> dict[float, list[tuple[float, float]]]:
    """{scale: [(f, |r_RR|^2), ...]}, sorted by frequency."""
    out: dict[float, list[tuple[float, float]]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            scale = float(row["scale"]) if row["scale"] else 1.0
            out[scale].append((float(row["freq_ghz"]), float(row["abs_r_RR_sq"])))
    for points in out.values():
        points.sort()
    return out


def peak(points):
    i = max(range(len(points)), key=lambda j: points[j][1])
    if i in (0, len(points) - 1):
        return points[i]
    (x0, y0), (x1, y1), (x2, y2) = points[i - 1:i + 2]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return x1, y1
    return x1 + 0.5 * (y0 - y2) / denom * (x1 - x0), y1 - 0.125 * (y0 - y2) ** 2 / denom


def at_freq(points, f):
    """Linear interpolation of |r_RR|^2 at f. Points are sorted by frequency."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= f <= x1:
            return y0 + (y1 - y0) * (f - x0) / (x1 - x0) if x1 > x0 else y0
    return None


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", nargs="?", default="runs/freq")
    ap.add_argument("-o", "--out", default="notes/freq_legacy_scale.png")
    args = ap.parse_args()

    d = Path(args.rundir)
    # freq_legacy_scale*.csv: the first follow-up plus the small-a extension.
    family: dict[float, list[tuple[float, float]]] = {}
    for path in sorted(d.glob("freq_legacy_scale*.csv")):
        family.update(load(path))
    family[1.0] = load(d / "freq_legacy.csv")[1.0]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.13, wspace=0.22)

    # --- A: the five spectra ---------------------------------------------
    ax = axes[0]
    style(ax)
    ax.set_ylim(0, 1.12)
    ax.axvline(DESIGN_FREQ, color=TEXT_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.axhline(1.0, color="#d5d4cf", linewidth=1)
    for i, (color, a) in enumerate(zip(RAMP, sorted(family))):
        points = family[a]
        ax.plot([f for f, _ in points], [y for _, y in points], color=color,
                linewidth=2, solid_capstyle="round")
        fp, vp = peak(points)
        # The x0.9 and x0.92 peaks are 96 MHz apart and both at ~1.0, so their
        # labels overlap on one line -- stagger every other one upward.
        ax.annotate(f"×{a:g}", xy=(fp, vp), xytext=(0, 7 + 12 * (i % 2)),
                    textcoords="offset points", color=color, fontsize=9,
                    ha="center", fontweight="bold")
        # The whole argument of the panel: as the peak climbs, its crossing of
        # the 5.8 GHz line drops. Mark the crossings so the eye reads both.
        # The x0.9 and x0.92 sweeps start at 5.9 GHz and have no crossing to mark.
        v = at_freq(points, DESIGN_FREQ)
        if v is not None:
            ax.plot([DESIGN_FREQ], [v], marker="o", markersize=7.5, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
            if a == 1.0:
                ax.annotate(f"{v:.3f} — наша точка", xy=(DESIGN_FREQ, v),
                            xytext=(-8, -2), textcoords="offset points",
                            color=color, fontsize=9, ha="right",
                            fontweight="bold")
    ax.annotate("идеальное отражение", xy=(5.35, 1.0), xytext=(0, -15),
                textcoords="offset points", color=TEXT_MUTED, fontsize=9)
    ax.annotate("точки на пунктире — значение в рабочей\nточке 5.8 ГГц: чем выше пик, тем они ниже",
                xy=(6.62, 0.06), color=TEXT_MUTED, fontsize=9, ha="right")
    ax.set_xlabel("частота, ГГц", color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel("|r$_{RR}$|²", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title("A · чем лучше пик, тем хуже на 5.8 ГГц", color=TEXT_PRIMARY,
                 fontsize=11, loc="left", pad=10)

    # --- B: peak height vs how much of the guide the disk fills ----------
    ax = axes[1]
    style(ax)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(1.0, color="#d5d4cf", linewidth=1)
    fill_lo = [2 * R_LEGACY_MM * a / A_WG_MM * 100 for a in sorted(family)]
    peak_lo = [peak(family[a])[1] for a in sorted(family)]
    # x1.05 sits against the r <= 15.5 mm bound with 0.57 mm of wall clearance and
    # its line is visibly deformed -- plotted, but marked as not trustworthy.
    ax.plot(fill_lo[:-1], peak_lo[:-1], color=SERIES, linewidth=2, marker="o",
            markersize=8, solid_capstyle="round")
    ax.plot(fill_lo[-2:], peak_lo[-2:], color=SERIES, linewidth=2,
            linestyle=(0, (3, 3)))
    ax.plot(fill_lo[-1:], peak_lo[-1:], color=SERIES, marker="o", markersize=8,
            markerfacecolor=SURFACE, markeredgewidth=2)
    ax.annotate("ε = 10", xy=(fill_lo[0], peak_lo[0]), xytext=(6, 6),
                textcoords="offset points", color=SERIES, fontsize=10,
                fontweight="bold")
    ax.annotate(f"{peak_lo[0]:.3f}", xy=(fill_lo[0], peak_lo[0]), xytext=(6, -13),
                textcoords="offset points", color=TEXT_PRIMARY, fontsize=9.5)
    ax.annotate("диск почти\nкасается стенок", xy=(fill_lo[-1], peak_lo[-1]),
                xytext=(-10, 14), textcoords="offset points", color=TEXT_MUTED,
                fontsize=9, ha="right")

    fill_hi = [2 * R_REF_MM * a / A_WG_MM * 100 for a in sorted(EPS80)]
    ax.plot(fill_hi, [EPS80[a] for a in sorted(EPS80)], color=SERIES_2, linewidth=2,
            marker="o", markersize=8, solid_capstyle="round")
    ax.annotate("ε = 80", xy=(fill_hi[-1], EPS80[1.12]), xytext=(8, -2),
                textcoords="offset points", color=SERIES_2, fontsize=10,
                fontweight="bold")

    ax.set_xlabel("доля сечения волновода, занятая диском (2r / 32 мм), %",
                  color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel("высота пика |r$_{RR}$|²", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title("B · высота выходит на 0.998 при заполнении 83–85 %",
                 color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)

    fig.suptitle("Масштабирование полумесяца ε = 10: 107 решений COMSOL, 11.08.2026",
                 color=TEXT_PRIMARY, fontsize=13, x=0.06, ha="left", y=0.945)
    fig.text(0.06, 0.875,
             "|r|² = 0.998 достижимо при том же мета-атоме — не хватает только "
             "волновода пошире: 35.5 мм вместо 32 (закон 1/a здесь не работает, показатель −0.79)",
             color=TEXT_MUTED, fontsize=9.5, ha="left")

    out = Path(args.out)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
