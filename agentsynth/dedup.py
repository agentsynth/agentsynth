"""Near-duplicate removal and benchmark decontamination.

Both use Jaccard similarity over word-shingles of the query — no extra
dependencies, fully deterministic. Dedup keeps the first of a near-duplicate
group; decontamination drops trajectories whose query looks like a held-out
benchmark item.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

from .schemas import Trajectory

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _content(trajectory: Trajectory) -> str:
    """Default dedup key: the whole example, not just the prompt — so the diverse
    variations a single query produces survive and only real duplicates are dropped."""
    return f"{trajectory.query} {trajectory.tool_signature()} {trajectory.final_answer}"


def _shingles(text: str, k: int = 3) -> Set[str]:
    tokens = _tokens(text)
    if len(tokens) < k:
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class DedupResult:
    kept: List[Trajectory] = field(default_factory=list)
    removed: List[Trajectory] = field(default_factory=list)
    # (removed_id, kept_id, similarity) for each dropped trajectory
    duplicates: List[Tuple[str, str, float]] = field(default_factory=list)


def dedup_trajectories(
    trajectories: List[Trajectory],
    threshold: float = 0.85,
    shingle_k: int = 3,
    key: Optional[Callable[[Trajectory], str]] = None,
) -> DedupResult:
    """Drop trajectories that are near-identical to one already kept.

    Similarity is Jaccard over shingles of `key(trajectory)`, which defaults to the
    full example (query + tool sequence + answer). Pass `key=lambda t: t.query` for
    prompt-level dedup instead.
    """
    key = key or _content
    result = DedupResult()
    kept_shingles: List[Tuple[Trajectory, Set[str]]] = []
    for traj in trajectories:
        shingles = _shingles(key(traj), shingle_k)
        best_sim = 0.0
        best_kept: Optional[Trajectory] = None
        for kept_traj, kept_sh in kept_shingles:
            sim = jaccard(shingles, kept_sh)
            if sim >= threshold and sim > best_sim:
                best_sim, best_kept = sim, kept_traj
        if best_kept is not None:
            result.removed.append(traj)
            result.duplicates.append((traj.id, best_kept.id, round(best_sim, 3)))
        else:
            result.kept.append(traj)
            kept_shingles.append((traj, shingles))
    return result


def decontaminate(
    trajectories: List[Trajectory],
    contaminants: List[str],
    threshold: float = 0.8,
    shingle_k: int = 3,
) -> Tuple[List[Trajectory], List[Trajectory]]:
    """Split trajectories into (clean, flagged) by similarity to benchmark text."""
    contaminant_shingles = [_shingles(c, shingle_k) for c in contaminants]
    clean: List[Trajectory] = []
    flagged: List[Trajectory] = []
    for traj in trajectories:
        shingles = _shingles(traj.query, shingle_k)
        if any(jaccard(shingles, cs) >= threshold for cs in contaminant_shingles):
            flagged.append(traj)
        else:
            clean.append(traj)
    return clean, flagged


__all__ = ["DedupResult", "dedup_trajectories", "decontaminate", "jaccard"]
