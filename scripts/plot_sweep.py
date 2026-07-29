"""Figure: how sharply |r_RR|^2 falls off around the reference geometry.

Small multiples -- one panel per swept parameter, a shared x axis in *percent
deviation from the reference* so the five widths are directly comparable, and a
shared y axis in |r_RR|^2. The half-peak line and the measured width of each
resonance are drawn in, which is the point of the figure: every axis loses half
the reflectance within a few percent.

Reads the CSV written by `sweep_chiral.py --out` (no COMSOL needed).

Usage:
    uv run python scripts/plot_sweep.py [sweep.csv] [-o out.png]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated single-hue palette (categorical slot 1, light surface) plus the
# text/surface tokens -- marks carry the data, text stays in ink colors.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"

# Panel titles use the model's own parameter names -- spelled-out Russian names
# made the five titles wide enough to collide.
LABELS = {
    "r_mm": ("r", "мм"),
    "h_mm": ("h", "мм"),
    "y_cut_mm": ("y_cut", "мм"),
    "r_cut_mm": ("r_cut", "мм"),
    "eps_r": ("eps_r", ""),
}


def load(path: str):
    axes, base = defaultdict(list), {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok":
                continue
            value, y = float(row["value"]), float(row["abs_r_RR_sq"])
            axes[row["param"]].append((value, y))
            if row["is_base"] == "True":
                base[row["param"]] = (value, y)
    for points in axes.values():
        points.sort()
    return axes, base


def half_width(points, x0, peak):
    """Width in native units between the two half-peak crossings around x0."""
    half = peak / 2
    edges = []
    for direction in (-1, +1):
        seq = [p for p in points if (p[0] > x0 if direction > 0 else p[0] < x0)]
        seq.sort(key=lambda p: abs(p[0] - x0))
        prev = (x0, peak)
        for x, y in seq:
            if y < half <= prev[1]:
                t = (prev[1] - half) / (prev[1] - y)
                edges.append(prev[0] + t * (x - prev[0]))
                break
            prev = (x, y)
        else:
            edges.append(None)
    lo, hi = edges
    return None if lo is None or hi is None else hi - lo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default="notes/sweep_reference.csv")
    ap.add_argument("-o", "--out", default="notes/sweep_reference.png")
    args = ap.parse_args()

    axes, base = load(args.csv)
    order = [k for k in LABELS if k in axes]
    peak = base[order[0]][1]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })
    fig, panels = plt.subplots(
        1, len(order), figsize=(14.5, 3.6), sharey=True,
        gridspec_kw={"wspace": 0.12})

    for ax, param in zip(panels, order):
        x0, _ = base[param]
        points = axes[param]
        # Percent deviation puts all five panels on one comparable scale.
        xs = [(v - x0) / x0 * 100 for v, _ in points]
        ys = [y for _, y in points]

        ax.axhline(peak / 2, color=TEXT_MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.plot(xs, ys, color=SERIES, lw=2, zorder=3)
        ax.plot(xs, ys, "o", color=SERIES, ms=4, zorder=3)
        ax.plot([0], [peak], "o", color=SERIES, ms=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

        name, unit = LABELS[param]
        width = half_width(points, x0, peak)
        caption = f"{width:.2f} {unit}".strip() if width else ""
        rel = f"{width / x0 * 100:.0f}%" if width else ""
        # Reference value goes in the title; the width annotation sits in the
        # top-left corner, which the peak (always at x=0) never reaches.
        ax.set_title(f"{name} = {x0:g} {unit}".strip(),
                     color=TEXT_PRIMARY, fontsize=11, pad=9)
        ax.text(0.03, 0.93, f"ширина {caption}", transform=ax.transAxes,
                ha="left", fontsize=9.5, color=TEXT_PRIMARY)
        ax.text(0.03, 0.85, f"({rel} от значения)", transform=ax.transAxes,
                ha="left", fontsize=9.5, color=TEXT_SECONDARY)

        ax.set_xlim(-55, 55)
        ax.set_xticks([-50, -25, 0, 25, 50])
        # Headroom so the peak marker clears the corner annotation.
        ax.set_ylim(0, peak * 1.28)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d6d5cf")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    panels[0].set_ylabel("$|r_{RR}|^2$", color=TEXT_PRIMARY, fontsize=11)
    panels[0].text(-52, peak / 2 + 0.025, "половина пика", fontsize=9,
                   color=TEXT_MUTED)

    fig.suptitle(
        "Резонанс вокруг референсной точки: срез по каждому параметру отдельно",
        x=0.5, y=1.06, fontsize=13, color=TEXT_PRIMARY)
    fig.text(0.5, 0.985,
             f"пик $|r_{{RR}}|^2$ = {peak:.3f}; по горизонтали — отклонение "
             f"параметра от референса, %; остальные параметры фиксированы",
             ha="center", fontsize=10, color=TEXT_SECONDARY)
    fig.text(0.5, -0.06, "отклонение параметра от референсного значения, %",
             ha="center", fontsize=10.5, color=TEXT_PRIMARY)

    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
