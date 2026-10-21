"""Read-only import of measured frequency points, never an imported score.

Grid/threshold/reporting code may change: the new objective requests individual
frequencies and recomputes its own band. Solver implementation, model, backend
and exact geometry must match. Source runs are never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from llm_opt import Config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BaselineSpectrum:
    def __init__(self, source_run, current_manifest, params):
        self.root = Path(source_run).resolve()
        config_path = self.root / "config.yaml"
        source = Config.from_yaml(config_path).extra.get("chiral", {})
        if source.get("objective") != "bandwidth" or not source.get("both_polarizations"):
            raise ValueError("Baseline must be a bandwidth run with both polarizations")
        for field in ("solver", "model_sha256"):
            if field not in source or source[field] != current_manifest[field]:
                raise ValueError(f"Baseline {field} differs from this run")
        solver_file = "src/opt/solvers.py"
        solver_hash = source.get("code_sha256", {}).get(solver_file)
        if not solver_hash or solver_hash != current_manifest["code_sha256"][solver_file]:
            raise ValueError("Baseline solver code differs from this run")
        if source["solver"] == "comsol" and not source["model_sha256"]:
            raise ValueError("Baseline lacks model provenance")
        inputs = {"config.yaml": sha256(config_path)}
        # Older launchers also recorded model checksums before/after execution.
        for name in ("model_before.sha256", "model_after.sha256"):
            path = self.root / name
            if path.exists():
                if path.read_text().split()[0] != source["model_sha256"]:
                    raise ValueError(f"Baseline {name} disagrees with saved model")
                inputs[name] = sha256(path)

        self.params = dict(params)
        matches = []
        for path in sorted((self.root / "spectra").glob("*/geometry.json")):
            identity = json.loads(path.read_text())
            if identity.get("params") == self.params:
                matches.append((path, identity))
        if len(matches) != 1:
            raise ValueError("Baseline needs exactly one spectrum of the exact starting geometry")
        geometry_path, identity = matches[0]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if geometry_path.parent.name != key or identity["settings"] != source["settings"]:
            raise ValueError("Baseline geometry identity/settings mismatch")
        result_path = geometry_path.with_name("result.json")
        result = json.loads(result_path.read_text())
        if result.get("solve_failed", True):
            raise ValueError("Baseline spectrum did not finish successfully")
        for path in (geometry_path, result_path):
            inputs[str(path.relative_to(self.root))] = sha256(path)
        self.points = {}
        for path in sorted(geometry_path.parent.glob("point_*.json")):
            row = json.loads(path.read_text())
            f = row["freq_ghz"]
            values = [row[k] for k in ("freq_ghz", "abs_r_RR_sq", "abs_r_LL_sq", "elapsed_s")]
            if (not all(math.isfinite(v) and v >= 0 for v in values)
                    or path.name != f"point_{f:.12f}.json" or f in self.points
                    or not source["settings"]["f_min"] <= f <= source["settings"]["f_max"]):
                raise ValueError(f"Invalid baseline frequency point: {path}")
            relative = str(path.relative_to(self.root))
            digest = sha256(path)
            inputs[relative] = digest
            self.points[f] = {
                "freq_ghz": f, "abs_r_RR_sq": row["abs_r_RR_sq"],
                "abs_r_LL_sq": row["abs_r_LL_sq"], "elapsed_s": 0.0,
                "reused_from": {"run": str(self.root), "point": relative,
                                "sha256": digest, "source_elapsed_s": row["elapsed_s"]},
            }
        # Compare the finished spectrum against its durable point records.
        recorded = result.get("spectrum", [])
        if not recorded or {r["freq_ghz"] for r in recorded} != set(self.points):
            raise ValueError("Baseline result and point cache disagree")
        for row in recorded:
            if any(row[c] != self.points[row["freq_ghz"]][c]
                   for c in ("abs_r_RR_sq", "abs_r_LL_sq")):
                raise ValueError("Baseline result and point values disagree")
        self.manifest = {
            "source_run": str(self.root), "available_points": len(self.points),
            "inputs_sha256": hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        }

    def point(self, params, frequency):
        if params != self.params:
            return None
        return self.points.get(frequency)
