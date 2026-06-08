"""A seed bank of agent tasks across domains.

Feed these into the generator to get a batch that spans domains and modes instead
of variations on a single query. `sample_tasks` is deterministic given a seed.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from ..utils import stable_seed


class SeedTask(BaseModel):
    id: str
    domain: str
    query: str
    mode: str = "single_agent"


SEED_TASKS: List[SeedTask] = [
    SeedTask(
        id="data_analysis_1",
        domain="data_analysis",
        query="Which region had the highest total revenue, and by how much?",
        mode="multi_agent",
    ),
    SeedTask(
        id="data_analysis_2",
        domain="data_analysis",
        query="Break down units sold by product and flag the weakest one.",
        mode="single_agent",
    ),
    SeedTask(
        id="data_analysis_3",
        domain="data_analysis",
        query="Compare Q3 and Q4 revenue by region and summarize the trend.",
        mode="multi_agent",
    ),
    SeedTask(
        id="coding_1",
        domain="coding",
        query="Compute the mean and standard deviation of 12, 19, 7, 22, 31, 44.",
        mode="code_execution",
    ),
    SeedTask(
        id="coding_2",
        domain="coding",
        query="Find the 12th Fibonacci number and explain the recurrence.",
        mode="code_execution",
    ),
    SeedTask(
        id="coding_3",
        domain="coding",
        query="Check whether 561 is a Carmichael number.",
        mode="code_execution",
    ),
    SeedTask(
        id="weather_math_1",
        domain="weather_math",
        query="What's the weather in Tokyo and what's an 18% tip on a $54 bill?",
        mode="single_agent",
    ),
    SeedTask(
        id="weather_math_2",
        domain="weather_math",
        query="Is it warmer in Madrid or Berlin right now?",
        mode="single_agent",
    ),
    SeedTask(
        id="file_qa_1",
        domain="file_qa",
        query="Read report.csv and tell me the total revenue and the best region.",
        mode="single_agent",
    ),
    SeedTask(
        id="file_qa_2",
        domain="file_qa",
        query="Open notes.md and summarize the action items.",
        mode="single_agent",
    ),
    SeedTask(
        id="web_research_1",
        domain="web_research",
        query="Find three recent sources on retrieval-augmented generation and summarize them.",
        mode="multi_agent",
    ),
    SeedTask(
        id="web_research_2",
        domain="web_research",
        query="What is the current population of Vietnam? Include a source.",
        mode="single_agent",
    ),
    SeedTask(
        id="support_1",
        domain="customer_support",
        query="A customer can't reset their password. Diagnose the likely cause and draft a reply.",
        mode="multi_agent",
    ),
    SeedTask(
        id="support_2",
        domain="customer_support",
        query="Summarize this week's top support tickets by theme.",
        mode="multi_agent",
    ),
    SeedTask(
        id="devops_1",
        domain="devops",
        query="A service is returning 500s. Outline a triage plan and check the logs.",
        mode="multi_agent",
    ),
    SeedTask(
        id="devops_2",
        domain="devops",
        query="Parse these request latencies and report p50 and p95: 120 90 300 75 210 180.",
        mode="code_execution",
    ),
    SeedTask(
        id="finance_1",
        domain="finance",
        query="Convert 100 shares of a $192 stock into euros at a rate of 0.92.",
        mode="code_execution",
    ),
    SeedTask(
        id="finance_2",
        domain="finance",
        query="What's the compound interest on $5000 at 4% annually for 6 years?",
        mode="code_execution",
    ),
    SeedTask(
        id="travel_1",
        domain="travel",
        query="Plan a 3-day Kyoto itinerary on a $1200 budget.",
        mode="multi_agent",
    ),
    SeedTask(
        id="productivity_1",
        domain="productivity",
        query="Draft and send a status email to the team about the launch.",
        mode="single_agent",
    ),
    SeedTask(
        id="productivity_2",
        domain="productivity",
        query="Schedule a 30-minute sync next Tuesday afternoon.",
        mode="single_agent",
    ),
    SeedTask(
        id="research_1",
        domain="research",
        query="Explain the difference between SFT and DPO for training agents.",
        mode="single_agent",
    ),
    SeedTask(
        id="research_2",
        domain="research",
        query="Summarize the main risks of deploying autonomous agents.",
        mode="single_agent",
    ),
]


def domains() -> List[str]:
    """Distinct domains in the seed bank, in first-seen order."""
    out: List[str] = []
    for task in SEED_TASKS:
        if task.domain not in out:
            out.append(task.domain)
    return out


def tasks_for_domain(domain: str) -> List[SeedTask]:
    return [t for t in SEED_TASKS if t.domain == domain]


def sample_tasks(n: int, domains: Optional[List[str]] = None, seed: int = 0) -> List[SeedTask]:
    """Pick `n` tasks deterministically, optionally restricted to some domains.

    Cycles through the pool if `n` is larger than it.
    """
    wanted = set(domains) if domains else None
    pool = [t for t in SEED_TASKS if wanted is None or t.domain in wanted]
    if not pool or n <= 0:
        return []
    ordered = sorted(pool, key=lambda t: stable_seed(seed, t.id))
    if n <= len(ordered):
        return ordered[:n]
    return [ordered[i % len(ordered)] for i in range(n)]


__all__ = ["SeedTask", "SEED_TASKS", "domains", "tasks_for_domain", "sample_tasks"]
