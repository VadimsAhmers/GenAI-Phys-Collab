"""Draw the COMSOL cylinder-minus-offset-cylinder geometry from a saved step.

MPLCONFIGDIR=/tmp/genai-phys-mpl .venv/bin/python -m scripts.plot_chiral_geometry \
  --step runs/bandwidth_local_20261021/steps/step_012.json
No COMSOL solve; shape follows chiral.mph geom1/cyl1, cyl2, dif1.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    step = json.loads(args.step.read_text())
    p = step["params"]
    r, h, d, c = (p[k] for k in ("r_mm", "h_mm", "y_cut_mm", "r_cut_mm"))
    # This annular parameterization applies when the offset cutter contains
    # the disk centre; it includes both a closed hole and an open notch.
    if not 0 < d < c or d >= r or c >= r + d:
        raise ValueError("This renderer requires 0 < offset < min(cut radius, disk radius) and overlapping disks")
    k = (r*r-c*c-d*d)/(2*d*c)
    if k >= 1:
        theta = np.linspace(0, 2*np.pi, 361)
    else:
        alpha = np.arcsin(k)
        theta = np.linspace(np.pi-alpha, 2*np.pi+alpha, 361)
    outer = -d*np.sin(theta) + np.sqrt(r*r-d*d*np.cos(theta)**2)
    a = np.column_stack((c*np.cos(theta), d+c*np.sin(theta)))
    b = np.column_stack((outer*np.cos(theta), d+outer*np.sin(theta)))
    outline = np.vstack((b, a[::-1]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(13, 7), facecolor="white")
    ax = fig.add_axes([.045, .18, .43, .68])
    ax.add_patch(Rectangle((-16, -16), 32, 32, facecolor="#f3f6fa", edgecolor="#7d8999", lw=2))
    ax.add_patch(Polygon(outline, facecolor="#63b4cf", edgecolor="#165a75", lw=1.6))
    ax.plot([0, 0], [-14, 15], color="#9aa8b5", lw=.8, ls="--")
    ax.plot([-14, 14], [0, 0], color="#9aa8b5", lw=.8, ls="--")
    ax.scatter([0, 0], [0, d], s=22, color="#253953", zorder=5)
    arrow = dict(arrowstyle="<->", color="#253953", shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(0, d), xytext=(0, 0), arrowprops=arrow)
    ax.text(.7, d/2, f"смещение {d:g} мм", va="center", fontsize=10)
    ax.annotate("", xy=(-r, 0), xytext=(0, 0), arrowprops=arrow)
    ax.text(-7.8, -1.6, f"R = {r:g} мм", ha="center", fontsize=10)
    ax.annotate("", xy=(c, d), xytext=(0, d), arrowprops=arrow)
    ax.text(4.5, d+1, f"R выреза = {c:g} мм", ha="center", fontsize=10)
    ax.annotate("", xy=(16, -18), xytext=(-16, -18), arrowprops=arrow)
    ax.text(0, -19.7, "волновод 32 × 32 мм", ha="center")
    ax.set(xlim=(-19, 19), ylim=(-21, 19), aspect="equal")
    ax.set_title("Вид вдоль оси волновода (z)", pad=12, weight="bold")
    ax.axis("off")

    ax3 = fig.add_axes([.51, .15, .46, .73], projection="3d")
    faces, colors = [], []
    for i in range(len(theta)-1):
        for z in [-h/2, h/2]:
            faces.append([(a[i,0],a[i,1],z), (b[i,0],b[i,1],z),
                          (b[i+1,0],b[i+1,1],z), (a[i+1,0],a[i+1,1],z)])
            colors.append("#70bed4" if z > 0 else "#397f9c")
        for curve, color in [(a,"#28728e"), (b,"#4b9db9")]:
            faces.append([(curve[i,0],curve[i,1],-h/2), (curve[i+1,0],curve[i+1,1],-h/2),
                          (curve[i+1,0],curve[i+1,1],h/2), (curve[i,0],curve[i,1],h/2)])
            colors.append(color)
    ax3.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors="none", linewidths=0))
    for curve in [a,b]:
        for z in [-h/2,h/2]:
            ax3.plot(curve[:,0],curve[:,1],z,color="#165a75",lw=.7)
    ax3.plot([16,16],[-9,-9],[-h/2,h/2],color="#253953",lw=1.4)
    for z in [-h/2,h/2]:
        ax3.plot([15,17],[-9,-9],[z,z],color="#253953")
    ax3.text(17,-9,0,f"h = {h:g} мм", fontsize=11)
    ax3.set(xlim=(-15,19),ylim=(-16,16),zlim=(-15,15))
    ax3.set_box_aspect((34,32,30))
    ax3.view_init(elev=29,azim=65)
    ax3.set_proj_type("ortho")
    ax3.set_axis_off()
    ax3.set_title("Объёмный вид · сквозной вырез", pad=0, weight="bold")
    fig.suptitle(f"Лучшая геометрия ночного прогона · шаг {step['step']}", y=.97, fontsize=19, weight="bold", color="#20354c")
    fig.text(.5,.09,f"Диэлектрик εᵣ = {p['eps_r']:g}   •   Ø {2*r:g} мм   •   толщина {h:g} мм",ha="center",fontsize=13)
    fig.text(.5,.045,"Построено по параметрам шага и операциям COMSOL-модели; размеры в масштабе.\nОбъёмный вид показан без стенок волновода.",ha="center",fontsize=10,color="#59697b")
    out = args.out or args.step.parent.parent / "plots" / f"geometry_step_{step['step']:03d}"
    out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out.with_suffix('.'+ext),dpi=190,facecolor="white")
    plt.close(fig)
    print(out.with_suffix('.png'))


if __name__ == "__main__":
    main()
