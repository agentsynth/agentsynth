"""Run τ-bench against a model.

Unlike BFCL, τ-bench is multi-turn and agentic — a user simulator plus a domain
database — so it can't be scored from single tool calls. This is a thin bridge to
the official `tau-bench` package: it builds the environment and the package's
tool-calling agent and reports the pass rate.

Point `model` at your fine-tuned model served behind an OpenAI-compatible endpoint
(vLLM, TGI, …). Install the harness with:

    pip install git+https://github.com/sierra-research/tau-bench
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def tau_bench_available() -> bool:
    try:
        import tau_bench  # noqa: F401

        return True
    except Exception:
        return False


def run_tau_bench(
    model: str,
    provider: str = "openai",
    env_name: str = "retail",
    user_model: str = "gpt-4o",
    user_provider: str = "openai",
    task_split: str = "test",
    task_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run τ-bench and return `{env, n, pass_rate, avg_reward}`.

    Requires the `tau-bench` package and API keys for the agent + user-simulator models.
    """
    try:
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        from tau_bench.envs import get_env
    except Exception as exc:
        raise ImportError(
            "tau-bench is not installed: "
            "pip install git+https://github.com/sierra-research/tau-bench"
        ) from exc

    env = get_env(
        env_name,
        user_strategy="llm",
        user_model=user_model,
        user_provider=user_provider,
        task_split=task_split,
    )
    agent = ToolCallingAgent(
        tools_info=env.tools_info, wiki=env.wiki, model=model, provider=provider
    )
    ids = task_ids if task_ids is not None else list(range(len(env.tasks)))
    rewards = [float(getattr(agent.solve(env=env, task_index=i), "reward", 0.0)) for i in ids]
    n = len(rewards)
    return {
        "env": env_name,
        "n": n,
        "pass_rate": round(sum(1 for r in rewards if r >= 0.999) / n, 4) if n else 0.0,
        "avg_reward": round(sum(rewards) / n, 4) if n else 0.0,
    }


__all__ = ["tau_bench_available", "run_tau_bench"]
