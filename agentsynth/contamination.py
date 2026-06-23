"""Contamination audit for a pack — is the benchmark already in the training data?

Public agent-benchmark scores inflate by 5–15 points when the benchmark has leaked into
pretraining or fine-tuning (2026 agent-benchmark surveys). Three defenses, all here:

- **canary strings**: a unique token per scenario you embed in the pack and later grep
  for in a model's output or a training corpus — if it comes back, the pack was memorized
- **corpus overlap**: shingle-Jaccard between each scenario's task and a candidate corpus,
  flagging the ones a model may already have seen
- **held-out siblings**: `perturb_scenario` rewrites the labels so a memorizing model
  can't match, giving a contamination-resistant variant to bench against

    from agentsynth.contamination import contamination_report
    print(contamination_report(scenarios, corpus=my_training_texts).summary_md())
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field


def canary_for(scenario_id: str, salt: str = "agentsynth") -> str:
    """A stable, unguessable token unique to a scenario.

    Embed it in the pack (or the task text) and search a model's outputs or a training
    corpus for it; a hit means the pack was memorized, not solved.
    """
    digest = hashlib.sha256(f"{salt}:{scenario_id}".encode("utf-8")).hexdigest()[:12]
    return f"agentsynth-canary-{digest}"


def corpus_overlap(scenario: Any, corpus: Sequence[str], k: int = 3) -> float:
    """Highest shingle-Jaccard between a scenario's task and any document in the corpus."""
    from .dedup import _shingles, jaccard

    if not corpus:
        return 0.0
    task = _shingles(scenario.task, k)
    return round(max(jaccard(task, _shingles(doc, k)) for doc in corpus), 4)


def held_out_pack(scenarios: Sequence[Any], seed: int = 0) -> List[Any]:
    """Isomorphic siblings of every scenario — a contamination-resistant variant.

    Single-table worlds are relabelled; multi-table ones (data in the schema's INSERTs)
    come back unchanged, so check the ids if you need to know which were transformed.
    """
    from .robustness import perturb_scenario

    return [perturb_scenario(scenario, seed) for scenario in scenarios]


class ScenarioContamination(BaseModel):
    id: str
    canary: str
    max_overlap: Optional[float] = None
    contaminated: bool = False


class ContaminationReport(BaseModel):
    n: int
    flagged: int
    threshold: float
    has_corpus: bool
    rows: List[ScenarioContamination] = Field(default_factory=list)

    def summary_md(self) -> str:
        lines: List[str] = []
        if self.has_corpus:
            lines.append(
                f"Corpus overlap: {self.flagged}/{self.n} scenarios sit above "
                f"{self.threshold:.0%} similarity to a corpus document — likely seen in "
                "training, so their scores are suspect."
            )
            for row in self.rows:
                if row.contaminated:
                    lines.append(f"- {row.id}: overlap {row.max_overlap:.0%}")
        else:
            lines.append(
                f"No corpus given — generated a canary for each of the {self.n} scenarios. "
                "Embed them and grep model outputs / training data; a hit means memorized."
            )
        lines.append("")
        lines.append("Canaries (embed in the pack, then search for leaks):")
        for row in self.rows[:5]:
            lines.append(f"- {row.id}: {row.canary}")
        if len(self.rows) > 5:
            lines.append(f"- … and {len(self.rows) - 5} more")
        lines.append("")
        lines.append(
            "Bench the held-out siblings (`--held-out`) for a contamination-resistant number."
        )
        return "\n".join(lines)


def contamination_report(
    scenarios: Sequence[Any],
    corpus: Optional[Sequence[str]] = None,
    threshold: float = 0.8,
) -> ContaminationReport:
    """Score each scenario for contamination risk and mint its canary."""
    rows: List[ScenarioContamination] = []
    flagged = 0
    for scenario in scenarios:
        overlap = corpus_overlap(scenario, corpus) if corpus else None
        contaminated = overlap is not None and overlap >= threshold
        if contaminated:
            flagged += 1
        rows.append(
            ScenarioContamination(
                id=scenario.id,
                canary=canary_for(scenario.id),
                max_overlap=overlap,
                contaminated=contaminated,
            )
        )
    return ContaminationReport(
        n=len(scenarios),
        flagged=flagged,
        threshold=threshold,
        has_corpus=corpus is not None,
        rows=rows,
    )


__all__ = [
    "canary_for",
    "corpus_overlap",
    "held_out_pack",
    "ScenarioContamination",
    "ContaminationReport",
    "contamination_report",
]
