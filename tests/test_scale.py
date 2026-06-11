"""Industrial-scale tests: cache, retries, budgets, resumable runs, MinHash dedup.

The LLM is a fake injected after construction, so everything runs hermetically.
"""

import json

import pytest

from agentsynth import (
    AgentTrajectoryGenerator,
    BudgetExceeded,
    CachingLLMClient,
    CostMeter,
    Recipe,
    run_resumable,
)
from agentsynth.dedup import dedup_trajectories

MESSAGES = [{"role": "user", "content": "say ok"}]


class _FakeLitellm:
    """Counts calls; fails the first `fail_times`; no pricing table."""

    def __init__(self, fail_times=0):
        self.calls = 0
        self.fail_times = fail_times

    def completion(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient upstream error")
        return {
            "choices": [{"message": {"content": f"ok-{self.calls}"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    def completion_cost(self, completion_response):
        raise ValueError("no pricing for the fake model")


def _client(tmp_path=None, **kwargs):
    kwargs.setdefault("backoff", 0.0)
    return CachingLLMClient(
        model="fake/model",
        cache_dir=str(tmp_path / "cache") if tmp_path else None,
        **kwargs,
    )


# --- cache ----------------------------------------------------------------------


def test_identical_prompts_hit_the_cache(tmp_path):
    client = _client(tmp_path)
    client._litellm = _FakeLitellm()
    assert client.complete(MESSAGES) == "ok-1"
    assert client.complete(MESSAGES) == "ok-1"  # served from disk
    assert client._litellm.calls == 1
    assert client.cache_hits == 1
    # a different request misses cleanly
    assert client.complete([{"role": "user", "content": "other"}]) == "ok-2"
    assert client._litellm.calls == 2


def test_cache_survives_a_new_client(tmp_path):
    first = _client(tmp_path)
    first._litellm = _FakeLitellm()
    first.complete(MESSAGES)
    second = _client(tmp_path)
    second._litellm = _FakeLitellm()
    assert second.complete(MESSAGES) == "ok-1"
    assert second._litellm.calls == 0  # never touched the provider


# --- retries --------------------------------------------------------------------


def test_transient_failures_are_retried():
    client = _client()
    client._litellm = _FakeLitellm(fail_times=2)
    assert client.complete(MESSAGES) == "ok-3"
    assert client._litellm.calls == 3


def test_exhausted_retries_keep_the_empty_string_contract():
    client = _client(max_retries=2)
    client._litellm = _FakeLitellm(fail_times=99)
    assert client.complete(MESSAGES) == ""
    assert "completion failed" in (client.last_error or "")
    assert client._litellm.calls == 2


# --- meter + budget ---------------------------------------------------------------


def test_meter_tracks_tokens_and_fallback_price():
    meter = CostMeter()
    client = _client(price_per_1k_tokens=0.01, meter=meter)
    client._litellm = _FakeLitellm()
    client.complete(MESSAGES)
    report = meter.report()
    assert report["calls"] == 1
    assert report["total_tokens"] == 120
    assert report["usd"] == pytest.approx(0.0012)


def test_budget_cap_stops_the_run():
    meter = CostMeter()
    client = _client(price_per_1k_tokens=0.01, budget_usd=0.001, meter=meter)
    client._litellm = _FakeLitellm()
    client.complete(MESSAGES)  # spends 0.0012 > 0.001
    with pytest.raises(BudgetExceeded):
        client.complete([{"role": "user", "content": "one more"}])


def test_rate_limit_spaces_calls_out():
    import time

    client = _client(min_interval=0.05)
    client._litellm = _FakeLitellm()
    start = time.monotonic()
    client.complete(MESSAGES)
    client.complete([{"role": "user", "content": "second"}])
    assert time.monotonic() - start >= 0.05


# --- resumable runs -----------------------------------------------------------------


def _lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_resumable_run_chunks_and_completes(tmp_path):
    recipe = Recipe(queries=[f"analyze dataset {i} by region" for i in range(6)], evaluate=False)
    out = str(tmp_path / "run")

    first = run_resumable(recipe, out, max_items=4)
    assert (first["done"], first["added"], first["total"]) == (4, 4, 6)
    assert len(_lines(first["path"])) == 4

    second = run_resumable(recipe, out)  # continues where it stopped
    assert (second["done"], second["added"]) == (6, 2)
    rows = _lines(second["path"])
    assert len(rows) == 6
    assert len({r["id"] for r in rows}) == 6

    third = run_resumable(recipe, out)  # idempotent once complete
    assert third["added"] == 0


def test_resumable_output_loads_back(tmp_path):
    from agentsynth import load_jsonl

    recipe = Recipe(queries=["weather in Hanoi today please"], evaluate=False)
    result = run_resumable(recipe, str(tmp_path / "r2"))
    trajs = load_jsonl(result["path"])
    assert len(trajs) == 1
    assert trajs[0].query == "weather in Hanoi today please"


# --- minhash dedup --------------------------------------------------------------------


def test_minhash_matches_pairwise_on_near_duplicates():
    gen = AgentTrajectoryGenerator(use_mock=True, seed=3)
    batch = []
    for _ in range(3):  # identical content -> guaranteed near-dups
        batch.append(gen.generate("total revenue by region this quarter"))
    batch.append(gen.generate("compute the mean of 4 8 15 16 23"))

    pairwise = dedup_trajectories(batch, threshold=0.85)
    minhash = dedup_trajectories(batch, threshold=0.85, method="minhash")
    assert len(minhash.removed) == len(pairwise.removed) >= 1
    assert {t.id for t in minhash.kept} == {t.id for t in pairwise.kept}


def test_minhash_keeps_distinct_items():
    gen = AgentTrajectoryGenerator(use_mock=True, seed=3)
    batch = [
        gen.generate("weather in Tokyo right now"),
        gen.generate("refund order 7 in the database"),
        gen.generate("compute the mean of 4 8 15 16 23"),
    ]
    result = dedup_trajectories(batch, threshold=0.85, method="minhash")
    assert len(result.kept) == 3 and not result.removed
