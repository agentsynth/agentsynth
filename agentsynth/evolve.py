"""Grow a query set with controlled variations.

Template mutations by default — deterministic, free, good enough to widen a seed
set. Pass an `llm_client` to paraphrase instead; any empty completion falls back
to the template, so the result is always `k` usable queries.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .utils import stable_seed

_MUTATIONS = (
    "{q}",
    "{q} Keep it under three steps.",
    "{q} Use exact figures in the answer.",
    "Urgent: {q}",
    "{q} You're the on-call engineer for this.",
    "{q} Explain the result in one sentence.",
    "{q} Double-check the inputs before acting.",
)

_PARAPHRASE_PROMPT = (
    "Rewrite this task in different words. Keep the same intent, tools, and any "
    "names or numbers. Reply with the rewritten task only.\n\nTask: {q}"
)


def evolve_queries(
    queries: Sequence[str],
    k: int = 20,
    seed: int = 7,
    llm_client: Optional[Any] = None,
) -> List[str]:
    """`k` variations over `queries`, visiting the sources round-robin."""
    sources = [q for q in queries if q and q.strip()]
    out: List[str] = []
    seen = set()
    round_no = 0
    use_llm = llm_client is not None and getattr(llm_client, "available", False)

    while sources and len(out) < k:
        for i, query in enumerate(sources):
            if round_no == 0:  # the originals pass through first
                variant = query
            else:
                variant = ""
                if use_llm and llm_client is not None:
                    variant = llm_client.complete(
                        [{"role": "user", "content": _PARAPHRASE_PROMPT.format(q=query)}]
                    ).strip()
                if not variant:
                    template = _MUTATIONS[stable_seed(seed, f"{i}#{round_no}") % len(_MUTATIONS)]
                    variant = template.format(q=query)
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
            if len(out) >= k:
                break
        round_no += 1
        if round_no > k:
            break
    return out


__all__ = ["evolve_queries"]
