"""Two-polarization bandwidth objective; no changes to the generic llm_opt repo."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_opt import Config, EvalResult, Objective, Reporter

from src.opt.solvers import ChiralSolveError
from src.opt.spectrum import spectrum_summary, threshold_band_summary


MAKSIM_2026_START = dict(r_mm=13.0, h_mm=24.8, y_cut_mm=5.1, r_cut_mm=8.3, eps_r=7.1)
# h=40 is an exploratory search ceiling, NOT a manufacturing/hardware limit.
# Lower radius/thickness and cut bounds retain the previous search choices.
MAKSIM_2026_BOUNDS = dict(
    r_mm=(4.0, 13.0), h_mm=(2.0, 40.0),
    y_cut_mm=(0.0, 17.5), r_cut_mm=(0.0, 17.5), eps_r=(6.0, 11.0),
)
CHANNELS = ("abs_r_RR_sq", "abs_r_LL_sq")


@dataclass(frozen=True)
class BandwidthSettings:
    f_min: float = 5.5
    f_max: float = 6.0
    level: float = 0.8
    step_ghz: float = 0.01
    edge_tolerance_ghz: float = 0.0025

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in asdict(self).values())
                or not 0 < self.f_min < self.f_max
                or not 0 < self.level <= 1
                or not 1e-6 <= self.step_ghz <= self.f_max - self.f_min
                or not 1e-6 <= self.edge_tolerance_ghz <= self.step_ghz):
            raise ValueError("Invalid bandwidth window, level, grid step or edge tolerance")

    def grid(self):
        n = math.ceil(round((self.f_max - self.f_min) / self.step_ghz, 10))
        return [round(self.f_min + (self.f_max - self.f_min) * i / n, 12)
                for i in range(n + 1)]


def write_json(path: Path, data):
    """Replace only our own artifact; never expose a half-written cache point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_spectrum(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["freq_ghz", *CHANNELS, "elapsed_s"])
        writer.writeheader()
        writer.writerows(rows)


class ChiralBandwidthObjective(Objective):
    def __init__(self, solver, settings=None, cache_dir=None,
                 max_consecutive_failures=10):
        self.solver = solver
        self.settings = settings or BandwidthSettings()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0

    def evaluate(self, params):
        settings = self.settings
        if "freq_ghz" in params:
            raise ValueError("Bandwidth objective controls frequency; remove freq_ghz")
        identity = {"params": params, "settings": asdict(settings)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = self.cache_dir / key if self.cache_dir is not None else None
        if directory:
            write_json(directory / "geometry.json", identity)
        points = {}
        calls = 0
        started = time.monotonic()

        def sample(f):
            nonlocal calls
            if f in points:
                return
            path = directory / f"point_{f:.12f}.json" if directory else None
            if path and path.exists():
                row = json.loads(path.read_text())
                if row["freq_ghz"] != f:
                    raise ValueError(f"Frequency mismatch in cache {path}")
            else:
                calls += 1
                begin = time.monotonic()
                print(f"  spectrum f={f:.6f} GHz (new solve {calls})", flush=True)
                try:
                    powers = self.solver.polarizations({**params, "freq_ghz": f})
                    row = {"freq_ghz": f, **{c: float(powers[c]) for c in CHANNELS},
                           "elapsed_s": time.monotonic() - begin}
                    if not all(math.isfinite(row[c]) and row[c] >= 0 for c in CHANNELS):
                        raise ValueError("Non-finite or negative reflection returned by solver")
                except Exception as exc:
                    if directory:
                        write_json(directory / f"failure_{time.time_ns()}.json",
                                   {"freq_ghz": f, "error": str(exc),
                                    "elapsed_s": time.monotonic() - begin})
                    raise
                if path:
                    write_json(path, row)
            if not all(math.isfinite(row[c]) and row[c] >= 0 for c in CHANNELS):
                raise ValueError("Invalid reflection in spectrum cache")
            points[f] = row

        try:
            for f in settings.grid():
                sample(f)
            # Bisect every observed channel threshold bracket, including bands
            # other than the widest one. Both channels share each physical solve.
            while True:
                fs = sorted(points)
                mids = []
                for a, b in zip(fs, fs[1:]):
                    crossing = any((points[a][c] >= settings.level)
                                   != (points[b][c] >= settings.level) for c in CHANNELS)
                    if crossing and b - a > settings.edge_tolerance_ghz + 1e-12:
                        mids.append(round((a + b) / 2, 12))
                if not mids:
                    break
                for f in mids:
                    sample(f)
        except ChiralSolveError as exc:
            self._consecutive_failures += 1
            if (self.max_consecutive_failures
                    and self._consecutive_failures >= self.max_consecutive_failures):
                raise RuntimeError("Consecutive COMSOL geometry failures; aborting") from exc
            # A partial spectrum has no valid bandwidth. Never bridge a missing
            # point, fabricate zero reflection, or score a partial interval.
            return EvalResult(
                score=2.0, feedback=f"INVALID spectrum: geometry solve failed: {exc}",
                aux={"solve_failed": True, "solver_calls_new": calls,
                     "spectrum": [points[f] for f in sorted(points)]},
            )
        self._consecutive_failures = 0
        rows = [points[f] for f in sorted(points)]
        fs = [r["freq_ghz"] for r in rows]
        values = [[r[c] for r in rows] for c in CHANNELS]
        common = threshold_band_summary(fs, *values, level=settings.level)
        channel_info = {
            c: {"band": threshold_band_summary(fs, v, level=settings.level),
                "peak": spectrum_summary(fs, v)}
            for c, v in zip(CHANNELS, values)
        }
        score = 1.0 - common["width_ghz"] / (settings.f_max - settings.f_min)
        aux = {
            "objective": "common_bandwidth", "settings": asdict(settings),
            "common_band": common, "channels": channel_info, "spectrum": rows,
            "n_frequency_points": len(rows), "solver_calls_new": calls,
            "evaluation_elapsed_s": time.monotonic() - started,
            "sampled_common_peak": max(min(r[c] for c in CHANNELS) for r in rows),
            "solve_failed": False,
        }
        if directory:
            write_json(directory / "result.json", {"score": score, **aux})
            write_spectrum(directory / "spectrum.csv", rows)
        details = []
        for c in CHANNELS:
            info = channel_info[c]
            peak = info["peak"]
            sampled_f = fs[peak["peak_index"]]
            details.append(f"{c}: sampled peak={peak['peak_value']:.4f} at "
                           f"{sampled_f:.6f} GHz; individual width="
                           f"{info['band']['width_ghz'] * 1000:.2f} MHz")
        feedback = (
            f"Common RR/LL >= {settings.level:g} band: "
            f"{common['width_ghz'] * 1000:.2f} MHz, interval "
            f"{common['widest_interval_ghz']} GHz in {settings.f_min:g}–{settings.f_max:g} GHz. "
            f"Score=1-width/window={score:.6f}; lower is better. "
            + "; ".join(details)
            + f". {len(rows)} sampled frequencies; crossings linearly interpolated "
            f"with brackets <= {settings.edge_tolerance_ghz * 1000:g} MHz. "
            "Unresolved dips/peaks between samples are possible; validate finalists on a finer grid. "
            "A window-edge interval is useful in-window width, not the full physical linewidth. "
            "Peak heights are diagnostics, not an additional score term."
        )
        return EvalResult(score=score, feedback=feedback, aux=aux)

    def target_description(self):
        s = self.settings
        return (
            f"Maximize the widest continuous interval inside {s.f_min:g}–{s.f_max:g} GHz "
            f"where BOTH |r_RR|^2 and |r_LL|^2 >= {s.level:g}. "
            "Do not sum disconnected intervals; no requirement to include 5.8 GHz. "
            "Score = 1 - common bandwidth / window width. All zero-width candidates "
            "score 1; use spectral feedback to find an above-threshold band."
        )

    def problem_preamble(self):
        return (
            "Design a dielectric disk with an off-centre circular cut in the existing "
            "32x32 mm microwave waveguide. Geometry and permittivity change; the "
            "waveguide is fixed. Full-wave spectra provide the two circular "
            "co-polarized reflection powers RR and LL. High reflection in both "
            "channels does not establish polarization selectivity. Optimize their common bandwidth."
        )


class BandwidthReporter(Reporter):
    """Export every evaluated design and a width/peak plot from saved steps."""
    def on_finalize(self, store, state):
        import matplotlib.pyplot as plt

        config = Config.from_yaml(store.root / "config.yaml")
        synthetic = config.extra.get("chiral", {}).get("solver") == "fake"
        records = [json.loads(p.read_text()) for p in sorted(store.steps_dir.glob("step_*.json"))]
        rows, successful = [], []
        for rec in records:
            aux = rec.get("aux", {})
            if not rec["valid"] or rec["duplicate"] or aux.get("solve_failed", True):
                continue
            row = {"step": rec["step"], **rec["params"], "score": rec["score"],
                   "width_mhz": 1000 * aux["common_band"]["width_ghz"],
                   "rr_peak": aux["channels"][CHANNELS[0]]["peak"]["peak_value"],
                   "ll_peak": aux["channels"][CHANNELS[1]]["peak"]["peak_value"],
                   "n_frequency_points": aux["n_frequency_points"]}
            rows.append(row)
            successful.append(rec)
        if rows:
            with (store.root / "bandwidth.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            fig, ax = plt.subplots(figsize=(6, 4))
            for column, label in (("rr_peak", "RR"), ("ll_peak", "LL")):
                ax.scatter([r["width_mhz"] for r in rows],
                           [r[column] for r in rows], label=label)
            ax.set(xlabel="Common bandwidth (MHz)", ylabel="Sampled peak reflectance")
            if synthetic:
                ax.set_title("Synthetic solver — software test only")
            ax.legend()
            fig.tight_layout()
            fig.savefig(store.plots_dir / "bandwidth_peaks.png", dpi=160)
            fig.savefig(store.plots_dir / "bandwidth_peaks.pdf")
            plt.close(fig)
            best = min(successful, key=lambda r: r["score"])
            write_spectrum(store.best_dir / "spectrum.csv", best["aux"]["spectrum"])
            write_spectrum(store.root / "first_spectrum.csv", successful[0]["aux"]["spectrum"])
            fig, ax = plt.subplots(figsize=(7, 4))
            comparisons = ([(best, "First / best")] if best is successful[0]
                           else [(successful[0], "First evaluated"), (best, "Best")])
            for rec, label in comparisons:
                samples = rec["aux"]["spectrum"]
                for c, style in zip(CHANNELS, ("-", "--")):
                    ax.plot([r["freq_ghz"] for r in samples], [r[c] for r in samples],
                            style, label=f"{label}: {c[6:8]}")
            ax.axhline(best["aux"]["settings"]["level"], color="gray", linestyle=":")
            ax.set(xlabel="Frequency (GHz)", ylabel="Reflectance")
            if synthetic:
                ax.set_title("Synthetic solver — software test only")
            ax.legend()
            fig.tight_layout()
            fig.savefig(store.plots_dir / "bandwidth_spectra.png", dpi=160)
            fig.savefig(store.plots_dir / "bandwidth_spectra.pdf")
            plt.close(fig)
        # Read the durable cache, not n_evals (which counts geometries). Includes
        # points from interrupted evaluations; an interrupted in-flight solve
        # cannot be counted as a completed call.
        cache = store.root / "spectra"
        calls = list(cache.glob("*/point_*.json")) + list(cache.glob("*/failure_*.json"))
        write_json(store.root / "solver_usage.json", {
            "completed_solver_calls": len(calls),
            "failed_solver_calls": len(list(cache.glob("*/failure_*.json"))),
            "solver_elapsed_s": sum(json.loads(p.read_text())["elapsed_s"] for p in calls),
            "note": "Completed calls only; excludes any interrupted in-flight call. Fake is not physics.",
        })
