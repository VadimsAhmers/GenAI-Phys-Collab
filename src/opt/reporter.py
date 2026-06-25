"""Domain artifacts for a multipole-channel run.

On finalize, writes two plots into the run's plots/ dir:
  * best_geometry.png   -- the axisymmetric cross-section of the best structure.
  * channel_distribution.png -- best power shares q vs the target shares t.

Generic score-history is already produced by the core; this Reporter adds the
problem-specific views (cf. Ivan's per-task figures).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)  # headless-safe; we only save files
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from llm_opt import Reporter
from src.prepare_model import CYL_HEIGHT
from src.opt.multipole import CHANNEL_ORDER


class MultipoleReporter(Reporter):
    def __init__(self, target: dict[str, float], total_height: float = CYL_HEIGHT) -> None:
        self.target = target
        self.total_height = total_height

    def on_finalize(self, store, state) -> None:
        if state.best is None:
            return
        radii = state.best.params.get("radii_nm")
        if radii:
            self._plot_geometry(radii, store.plots_dir / "best_geometry.png")
        if state.best_result is not None:
            q = state.best_result.aux.get("q")
            if q:
                self._plot_channels(q, store.plots_dir / "channel_distribution.png")

    # --- plots --------------------------------------------------------------
    def _plot_geometry(self, radii, path) -> None:
        n = len(radii)
        h = self.total_height / n
        fig, ax = plt.subplots(figsize=(4, 6))
        for i, r in enumerate(radii):
            z0 = -self.total_height / 2 + i * h
            ax.add_patch(
                Rectangle((-r, z0), 2 * r, h, facecolor="#7fb3d5",
                          edgecolor="#21618c", linewidth=1.0)
            )
        rmax = max(radii)
        ax.set_xlim(-rmax * 1.15, rmax * 1.15)
        ax.set_ylim(-self.total_height / 2 * 1.05, self.total_height / 2 * 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("r, nm")
        ax.set_ylabel("z, nm")
        ax.set_title("Best geometry (cross-section)")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def _plot_channels(self, q: dict, path) -> None:
        labels = CHANNEL_ORDER
        best = [q.get(k, 0.0) for k in labels]
        tgt_total = sum(self.target.values())
        target = [self.target.get(k, 0.0) / tgt_total for k in labels]
        x = range(len(labels))
        w = 0.4
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - w / 2 for i in x], best, w, label="best q", color="#2e86c1")
        ax.bar([i + w / 2 for i in x], target, w, label="target t", color="#c0392b")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("power share")
        ax.set_title("Channel distribution: best vs target")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
