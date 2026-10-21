"""Domain-specific spectral context without modifying the nested llm_opt repo."""
from __future__ import annotations

import json
from pathlib import Path

from llm_opt import Optimizer
from llm_opt.llm import LLMProposer


class SpectralLLMProposer(LLMProposer):
    def __init__(self, *args, steps_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_dir = Path(steps_dir)

    def user_prompt(self, top_k, recent, best, best_feedback,
                    trajectory=None, supervisor_note=None):
        # The generic "no effect" verdict for equal scalar scores is misleading
        # here: two zero-width designs can have very different spectra.
        base = super().user_prompt(top_k, recent, best, best_feedback,
                                   trajectory=None, supervisor_note=supervisor_note)
        records = {}
        for path in sorted(self.steps_dir.glob("step_*.json")):
            record = json.loads(path.read_text())
            if record.get("valid") and record.get("feedback"):
                records[json.dumps(record["params"], sort_keys=True)] = record
        selected = [*top_k, *recent, *(trajectory or [])[-5:]]
        seen, lines = set(), []
        for attempt in selected:
            key = json.dumps(attempt.params, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            record = records.get(key)
            if record:
                lines.append(
                    f"Saved step {record['step']}: {self.param.format_params(record['params'])}\n"
                    f"score={record['score']:.6f}; {record['feedback']}")
        if lines:
            base += (
                "\n\nSPECTRAL RESULTS for best/recent evaluated candidates:\n"
                + "\n\n".join(lines)
                + "\nEqual scores mean equal measured common bandwidth, not equal spectra. "
                "Peak heights and positions are diagnostics, not extra objective terms. "
                "Use them to plan the next geometry while maximizing common bandwidth. "
                "Changes to several parameters at once do not identify a single cause."
            )
        return base


class BandwidthOptimizer(Optimizer):
    def _make_proposer(self):
        if self.cfg.proposer != "llm":
            return super()._make_proposer()
        c = self.cfg.llm
        return SpectralLLMProposer(
            param=self.param, objective=self.objective, api_key=self._api_key(),
            steps_dir=self.store.steps_dir,
            model=c.model, base_url=c.base_url, temperature=c.temperature,
            max_parse_retries=c.max_parse_retries, aggressiveness=c.aggressiveness,
            max_retries=c.max_retries, request_timeout=c.request_timeout,
            providers=c.providers, allow_fallbacks=c.allow_fallbacks,
            reasoning_effort=c.reasoning_effort,
        )
