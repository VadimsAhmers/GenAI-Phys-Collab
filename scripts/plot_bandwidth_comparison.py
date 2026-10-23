"""Plot the October baseline and verified 5 MHz finalist without new solves.

MPLCONFIGDIR=/tmp/genai-phys-mpl .venv/bin/python -m scripts.plot_bandwidth_comparison
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.opt.spectrum import threshold_band_summary


def read_spectrum(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if any(r.get("status", "ok") != "ok" for r in rows):
        raise ValueError(f"Failed points in {path}")
    rows.sort(key=lambda r: float(r["freq_ghz"]))
    return np.array([[float(r[k]) for k in
                      ("freq_ghz", "abs_r_RR_sq", "abs_r_LL_sq")] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=Path("runs/bandwidth_local_20261021"))
    parser.add_argument("--validation", type=Path, default=Path("runs/freq/best12_grid5_20261022"))
    args = parser.parse_args()
    old = json.loads((args.search / "steps/step_000.json").read_text())
    report = json.loads((args.validation / "verification.json").read_text())
    assert report["status"] == "complete_verified"
    assert hashlib.sha256((args.validation / "spectrum.csv").read_bytes()).hexdigest() == report["spectrum_sha256"]
    original = read_spectrum(args.search / "first_spectrum.csv")
    refined = read_spectrum(args.validation / "spectrum.csv")
    series = [(original, old["params"], "Исходный образец", "#2879b9"),
              (refined, report["params"], "Найденный вариант", "#d75c28")]
    bands = [threshold_band_summary(a[:, 0], a[:, 1], a[:, 2], level=.8)
             for a, *_ in series]
    assert np.isclose(bands[0]["width_ghz"], old["aux"]["common_band"]["width_ghz"])
    assert np.isclose(bands[1]["width_ghz"], report["common_band"]["width_ghz"])
    gain = 100 * (bands[1]["width_ghz"] / bands[0]["width_ghz"] - 1)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    ax = fig.add_axes([.1, .39, .85, .43])
    for (a, params, name, color), band in zip(series, bands):
        # The lower channel at each sample directly displays the joint criterion.
        ax.plot(a[:, 0], np.minimum(a[:, 1], a[:, 2]), color=color,
                marker="o", ms=2.6, lw=1.8,
                label=f"{name}: {band['width_ghz'] * 1000:.1f} МГц")
    ax.axhline(.8, color="#596675", ls="--", lw=1)
    ax.text(5.505, .815, "порог 0.8 для обоих каналов", color="#596675", fontsize=10)
    ax.axvline(5.8, color="#b6bec6", ls=":", lw=1, zorder=0)
    ax.set(xlim=(5.5, 6), ylim=(0, 1.04),
           ylabel="Меньшее отражение из RR и LL", xlabel="Частота, ГГц")
    ax.set_xticks(np.arange(5.5, 6.001, .05))
    ax.grid(alpha=.17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    bars = fig.add_axes([.1, .23, .85, .085], sharex=ax)
    for i, ((_, _, name, color), band) in enumerate(zip(series, bands)):
        y = 1-i
        lo, hi = band["widest_interval_ghz"]
        bars.plot([lo, hi], [y, y], color=color, lw=10, solid_capstyle="butt")
        bars.text(lo-.007, y, f"{lo:.4f}", ha="right", va="center", color=color, fontsize=10)
        bars.text(hi+.007, y, f"{hi:.4f}", ha="left", va="center", color=color, fontsize=10)
        bars.text((lo+hi)/2, y+.25, f"{name} · {band['width_ghz']*1000:.1f} МГц",
                  ha="center", color=color, fontsize=10)
    bars.set_ylim(-.4, 1.65)
    bars.axis("off")
    fig.suptitle(f"Общая полоса отражения расширена на {gain:.0f}%", y=.96,
                 fontsize=20, weight="bold", color="#253449")
    fig.text(.5, .902, f"{bands[0]['width_ghz']*1000:.1f} → {bands[1]['width_ghz']*1000:.1f} МГц  ·  +{gain:.1f}%  ·  RR и LL ≥ 0.8",
             ha="center", fontsize=14, color="#34465b")
    fig.text(.1, .17, "Параметры (r, h, y_cut, r_cut), мм:", fontsize=10, color="#596675")
    for i, (_, params, name, color) in enumerate(series):
        values = ", ".join(f"{params[k]:g}" for k in ("r_mm", "h_mm", "y_cut_mm", "r_cut_mm"))
        fig.text(.1, .142-i*.029, f"{name}: ({values}), ε = {params['eps_r']:g}", color=color, fontsize=11)
    fig.text(.1, .067, "COMSOL · волновод 32 × 32 мм · исходный спектр: шаг 20 МГц; найденный: 5 МГц.",
             fontsize=9, color="#596675")
    fig.text(.1, .043, "Точки — расчёты; линии соединяют отсчёты. Границы полос — интерполяция, скобки ≤2.5 МГц.",
             fontsize=9, color="#596675")
    out = args.validation / "plots" / "baseline_vs_best_grid5"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix("."+ext), dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
