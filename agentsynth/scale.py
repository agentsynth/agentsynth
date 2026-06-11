"""Caching, retries, budgets, and resumable runs for real-LLM generation.

`CachingLLMClient` is a drop-in `LLMClient` (the generator's `llm_client=`) with a
disk cache keyed on the full request, exponential-backoff retries, a token/cost
meter, a hard budget cap, and an optional rate limit. `run_resumable` writes
trajectories incrementally with a state file, so an interrupted run continues
where it stopped. Local backends work through LiteLLM model strings
("ollama/llama3.1", "hosted_vllm/<model>").

    meter = CostMeter()
    client = CachingLLMClient("claude-haiku-...", cache_dir=".agentsynth_cache",
                              budget_usd=25.0, meter=meter)
    gen = AgentTrajectoryGenerator(llm_client=client)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from .pipelines import Recipe
from .utils import LLMClient


class BudgetExceeded(RuntimeError):
    """Raised before a call that would start past the configured budget."""


class CostMeter:
    """Thread-safe usage counter shared across clients and runs."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usd = 0.0
        self._lock = threading.Lock()

    def add(self, prompt_tokens: int, completion_tokens: int, usd: float) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            self.usd += float(usd)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def report(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usd": round(self.usd, 6),
        }


class CachingLLMClient(LLMClient):
    """An LLMClient with a disk cache, retries, a cost meter, and a budget cap.

    The cache key is the full request (model, messages, sampling params). Costs
    come from LiteLLM's pricing table when it knows the model, otherwise from
    `price_per_1k_tokens`; without either, `usd` stays 0, so set a price before
    relying on `budget_usd`.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1536,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        backoff: float = 0.5,
        budget_usd: Optional[float] = None,
        price_per_1k_tokens: Optional[float] = None,
        min_interval: float = 0.0,
        meter: Optional[CostMeter] = None,
    ) -> None:
        super().__init__(
            model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key
        )
        self.cache_dir = cache_dir
        self.max_retries = max(1, int(max_retries))
        self.backoff = backoff
        self.budget_usd = budget_usd
        self.price_per_1k_tokens = price_per_1k_tokens
        self.min_interval = min_interval
        self.meter = meter or CostMeter()
        self.cache_hits = 0
        self._gate = threading.Lock()
        self._last_call = 0.0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- cache -----------------------------------------------------------------

    def _cache_path(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int):
        if not self.cache_dir:
            return None
        payload = json.dumps(
            {"m": self.model, "msg": messages, "t": temperature, "mt": max_tokens},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    # -- the call --------------------------------------------------------------

    def complete(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        if not self.available:
            self.last_error = self.last_error or "LLM client not available"
            return ""
        temp = self.temperature if temperature is None else temperature
        toks = self.max_tokens if max_tokens is None else max_tokens

        path = self._cache_path(messages, temp, toks)
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.cache_hits += 1
                    return json.load(fh)["text"]
            except Exception:
                pass  # unreadable cache entry — fall through to a real call

        if self.budget_usd is not None and self.meter.usd >= self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.meter.usd:.4f} of the ${self.budget_usd:.2f} budget"
            )

        if self.min_interval > 0:
            with self._gate:
                wait = self.min_interval - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.monotonic()

        for attempt in range(self.max_retries):
            try:
                resp = self._litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=toks,
                    api_key=self.api_key,
                    **kwargs,
                )
                text = resp["choices"][0]["message"]["content"] or ""
                self._record_usage(resp)
                if path:
                    try:
                        with open(path, "w", encoding="utf-8") as fh:
                            json.dump({"text": text}, fh, ensure_ascii=False)
                    except Exception:
                        pass  # caching is best-effort
                return text
            except Exception as exc:
                self.last_error = f"completion failed: {exc}"
                if attempt + 1 < self.max_retries and self.backoff > 0:
                    time.sleep(self.backoff * (2**attempt))
        return ""

    def _record_usage(self, resp: Any) -> None:
        prompt_tokens = completion_tokens = 0
        try:
            usage = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
            if usage:
                get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
                prompt_tokens = int(get("prompt_tokens", 0) or 0)
                completion_tokens = int(get("completion_tokens", 0) or 0)
        except Exception:
            pass
        usd = 0.0
        try:
            usd = float(self._litellm.completion_cost(completion_response=resp))
        except Exception:
            if self.price_per_1k_tokens:
                usd = (prompt_tokens + completion_tokens) / 1000.0 * self.price_per_1k_tokens
        self.meter.add(prompt_tokens, completion_tokens, usd)


def run_resumable(
    recipe: Recipe,
    out_dir: str,
    llm_client: Optional[LLMClient] = None,
    max_items: Optional[int] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate a recipe with incremental output and a resume file.

    Trajectories append to `<out_dir>/trajectories.jsonl` one line at a time;
    `<out_dir>/state.json` records progress, and re-invoking with the same
    `out_dir` (and the same recipe) continues from there. `max_items` caps how
    many this invocation adds, for chunked or cron-driven runs. Returns
    `{total, done, added, path}`. Run evaluation/verification/dedup as a
    post-pass over `load_jsonl(path)` once `done == total`.
    """
    from .exporters import _traj_to_record
    from .generator import AgentTrajectoryGenerator
    from .pipelines.recipe import make_environment
    from .pipelines.runner import _plan

    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "state.json")
    data_path = os.path.join(out_dir, "trajectories.jsonl")

    plan = _plan(recipe)
    done = 0
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as fh:
            done = int(json.load(fh).get("done", 0))

    environment = make_environment(recipe.environment)
    generator = AgentTrajectoryGenerator(
        model=recipe.model,
        temperature=recipe.temperature,
        max_steps=recipe.max_steps,
        use_mock=recipe.use_mock,
        seed=recipe.seed,
        environment=environment,
        llm_client=llm_client,
    )

    added = 0
    try:
        for query, idx, mode in plan[done:]:
            if max_items is not None and added >= max_items:
                break
            trajectory = generator.generate(query, mode=mode, index=idx)
            with open(data_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_traj_to_record(trajectory), ensure_ascii=False) + "\n")
            done += 1
            added += 1
            tmp = state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"done": done, "total": len(plan)}, fh)
            os.replace(tmp, state_path)
            if callable(progress):
                try:
                    progress(done / (len(plan) or 1), desc=f"{done}/{len(plan)}")
                except Exception:
                    pass
    finally:
        if environment is not None:
            environment.close()

    return {"total": len(plan), "done": done, "added": added, "path": data_path}


__all__ = ["BudgetExceeded", "CostMeter", "CachingLLMClient", "run_resumable"]
