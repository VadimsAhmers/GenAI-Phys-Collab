"""Re-evaluate the October finalist using Maksim's model with 0.5 mm fillets.

Run from the repository root:
    .venv/bin/python -u -m scripts.check_fillet_bandwidth

One geometry, both polarizations, 5.5–6 GHz, 10 MHz grid with threshold
refinement to 2.5 MHz. No optimizer or LLM calls. The source model is never
saved. Use --solver fake to exercise the workflow without COMSOL or an API.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import traceback

import yaml

from src.opt.bandwidth import (
    BandwidthSettings, ChiralBandwidthObjective, write_json, write_spectrum,
)
from src.opt.solvers import ChiralSolver, FakeBandwidthSolver


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/chiral_fillet_0p5.mph"
MODEL_SHA256 = "077a0088784eb6a656771401c38606446a18df365de629313bf64a90fb4a99d1"
PARAMS = dict(r_mm=13.0, h_mm=25.0, y_cut_mm=4.6, r_cut_mm=8.8, eps_r=7.3)
SETTINGS = BandwidthSettings(step_ghz=0.01, edge_tolerance_ghz=0.0025)
CODE_FILES = (
    "scripts/check_fillet_bandwidth.py", "src/opt/solvers.py",
    "src/opt/bandwidth.py", "src/opt/spectrum.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(solver):
    """Rebuild the finalist and record selections, without solving or saving."""
    model = solver._ensure_model()
    for key, value in PARAMS.items():
        unit = solver.PARAM_UNITS[key]
        model.parameter(solver.PARAM_MAP[key], f"{value}[{unit}]" if unit else str(value))
    model.build()
    geom = model.java.geom("geom1")
    radius = float(str(geom.feature("fil1").getString("radius")))
    if radius != 0.5 or geom.getNDomains() != 2:
        raise ValueError("Expected 0.5 mm fillets and two domains (air and disk)")
    return {
        "status": "geometry_built", "fillet_radius_mm": radius,
        "domains": int(geom.getNDomains()), "boundaries": int(geom.getNBoundaries()),
        "geometry_bounds_mm": [float(v) for v in geom.getBoundingBox()],
        "materials": {
            str(tag): [int(v) for v in model.java.material(tag).selection().entities()]
            for tag in model.java.material().tags()
        },
        "ports": {
            str(tag): [int(v) for v in model.java.physics("ewfd").feature(tag).selection().entities()]
            for tag in model.java.physics("ewfd").feature().tags()
            if str(model.java.physics("ewfd").feature(tag).getType()) == "Port"
        },
        "limitations": "Rebuild and selection IDs only; no visual or mesh convergence check",
    }


def evaluate(out: Path, backend: str, resume: bool):
    """Keep partial frequency records; refuse cache reuse with changed inputs."""
    model_hash = sha256(MODEL) if backend == "comsol" else None
    if backend == "comsol" and model_hash != MODEL_SHA256:
        raise ValueError("Model differs from the inspected file supplied by Maksim")
    manifest = {
        "solver": backend, "objective": "bandwidth", "both_polarizations": True,
        "model": str(MODEL) if backend == "comsol" else None,
        "model_sha256": model_hash, "params": PARAMS, "settings": asdict(SETTINGS),
        "code_sha256": {p: sha256(ROOT / p) for p in CODE_FILES},
        "fillet_radius_mm": 0.5 if backend == "comsol" else None,
        "purpose": "Single fixed geometry; no optimizer or LLM calls",
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "manifest.json"
        if path.exists():
            if not resume:
                raise ValueError("Run already exists; use --resume or a different --out")
            if json.loads(path.read_text()) != manifest:
                raise ValueError("Resume inputs changed; refusing to reuse frequency records")
        else:
            if resume or (out / "spectra").exists():
                raise ValueError("Missing manifest; cannot establish cache provenance")
            write_json(path, manifest)
            # Compatible with the existing BaselineSpectrum importer.
            (out / "config.yaml").write_text(yaml.safe_dump({
                "run_name": out.name, "proposer": "random",
                "stopping": {"max_iters": 1}, "extra": {"chiral": manifest},
            }, sort_keys=False))
        solver = None
        exit_code = 1
        (out / "started_at.txt").write_text(now() + "\n")
        for name in ("exit_code.txt", "finished_at.txt"):
            (out / name).unlink(missing_ok=True)
        write_json(out / "status.json", {"status": "running", "started_at": now()})
        try:
            if backend == "comsol":
                (out / "model_before.sha256").write_text(f"{model_hash}  {MODEL}\n")
                solver = ChiralSolver(str(MODEL))
                write_json(out / "preflight.json", preflight(solver))
            else:
                solver = FakeBandwidthSolver()
            print(f"Backend: {backend}; parameters: {PARAMS}", flush=True)
            print("51 grid points plus threshold refinement; both RR and LL.", flush=True)
            result = ChiralBandwidthObjective(
                solver, SETTINGS, out / "spectra", max_consecutive_failures=1,
            ).evaluate(PARAMS)
            if result.aux["solve_failed"]:
                raise RuntimeError(result.feedback)
            if backend == "comsol" and sha256(MODEL) != model_hash:
                raise RuntimeError("Source model checksum changed during the run")
            write_spectrum(out / "spectrum.csv", result.aux["spectrum"])
            write_json(out / "summary.json", {
                "status": "complete", "solver": backend, "params": PARAMS,
                "score": result.score, "feedback": result.feedback, **result.aux,
            })
            write_json(out / "status.json", {"status": "complete", "finished_at": now()})
            print(result.feedback, flush=True)
            exit_code = 0
            return result
        except BaseException as exc:
            write_json(out / "status.json", {
                "status": "failed", "error": str(exc), "finished_at": now(),
            })
            raise
        finally:
            try:
                if isinstance(solver, ChiralSolver):
                    solver.close()
            finally:
                if backend == "comsol":
                    (out / "model_after.sha256").write_text(f"{sha256(MODEL)}  {MODEL}\n")
                (out / "finished_at.txt").write_text(now() + "\n")
                (out / "exit_code.txt").write_text(str(exit_code) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("comsol", "fake"), default="comsol")
    parser.add_argument("--out", type=Path, help="Output directory; defaults to runs/fillet_check_SOLVER")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    out = args.out or ROOT / "runs" / f"fillet_check_{args.solver}"
    try:
        evaluate(out, args.solver, args.resume)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
