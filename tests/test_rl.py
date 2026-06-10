"""RL layer tests: AgentGym episodes, verified rewards, OpenEnv bridge.

Everything core runs hermetically against SQLEnvironment on any interpreter; the
OpenEnv bridge tests are skipped where openenv-core (Python 3.10+) isn't installed.
"""

import json

import pytest

from agentsynth import AgentGym, make_reward_fn
from agentsynth.environments import SQLEnvironment
from agentsynth.rl import ToolAction, episodes_to_grpo_jsonl, score_tool_completion
from agentsynth.rl.episode import _coerce_action

GOOD_SQL = "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region"


@pytest.fixture()
def gym():
    g = AgentGym(SQLEnvironment(), task="total revenue by region", seed=11)
    yield g
    g.close()


# --- episode mechanics -------------------------------------------------------


def test_reset_returns_task_and_requires_one(gym):
    assert gym.reset() == "total revenue by region"
    with pytest.raises(ValueError):
        AgentGym(SQLEnvironment()).reset()


def test_tool_step_executes_for_real(gym):
    gym.reset()
    out = gym.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
    assert not out.done
    assert out.reward == 0.0
    assert "EMEA" in out.observation  # the actual SQLite fixture answered


def test_error_observation_is_penalized(gym):
    gym.reset()
    out = gym.step({"tool_name": "sql_query", "arguments": {"query": "DROP TABLE sales"}})
    assert out.observation.startswith("SQLError")
    assert out.reward == -gym.error_penalty


def test_invalid_tool_is_penalized_and_not_recorded(gym):
    gym.reset()
    out = gym.step({"tool_name": "no_such_tool", "arguments": {}})
    assert out.info.get("invalid_action")
    assert out.reward == -gym.invalid_action_penalty
    final = gym.step({"answer": "done"})
    assert len(final.info["trajectory"].tool_calls()) == 0  # invalid attempt not in the data


def test_answer_finishes_with_verified_reward(gym):
    gym.reset()
    gym.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
    out = gym.step({"answer": "EMEA leads on revenue."})
    assert out.done
    assert 0.0 <= out.reward <= 1.0
    assert out.info["verification"]["verified"] is True
    assert "overall" in out.info["judge"]
    traj = out.info["trajectory"]
    assert traj.verification is not None
    assert traj.tool_names_used() == ["sql_query"]


def test_step_after_done_raises(gym):
    gym.reset()
    gym.step({"answer": "early exit"})
    with pytest.raises(RuntimeError):
        gym.step({"answer": "again"})


def test_max_steps_truncates_and_scores(gym):
    gym.max_steps = 2
    gym.reset()
    gym.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
    out = gym.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
    assert out.done
    assert out.info["truncated"] is True


def test_episodes_are_deterministic():
    def run():
        g = AgentGym(SQLEnvironment(), task="revenue by region", seed=5)
        g.reset()
        g.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
        out = g.step({"answer": "EMEA leads."})
        g.close()
        return out.reward, out.info["trajectory_id"]

    assert run() == run()


def test_coerce_action_accepts_raw_model_text():
    act = _coerce_action('{"tool": "sql_query", "args": {"query": "SELECT 1"}}')
    assert act.tool_name == "sql_query" and act.arguments == {"query": "SELECT 1"}
    assert _coerce_action("just an answer").answer == "just an answer"
    assert _coerce_action(ToolAction(answer="x")).answer == "x"
    with pytest.raises(TypeError):
        _coerce_action(42)


def test_rollout_with_scripted_policy(gym):
    script = iter(
        [
            {"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}},
            {"answer": "EMEA leads with the highest total revenue."},
        ]
    )
    episode = gym.rollout(lambda obs, g: next(script))
    assert episode.trajectory.final_answer.startswith("EMEA")
    assert len(episode.rewards) == 2
    assert episode.total_reward == round(sum(episode.rewards), 6)
    assert episode.info["verification"]["verified"] is True


class _FakeEnv:
    """Minimal Environment double returning a canned observation."""

    def __init__(self, text):
        self.text = text
        from agentsynth.schemas import ToolSpec

        self._tools = [
            ToolSpec(
                name="echo", description="echo", parameters={"type": "object", "properties": {}}
            )
        ]

    def tools(self):
        return self._tools

    def tool_names(self):
        return ["echo"]

    def execute(self, tool_name, args):
        return self.text

    def reset(self):
        pass

    def close(self):
        pass


def test_error_regex_requires_exception_format():
    # "Error rates dropped 5%" is data, not a failure — must not be penalized.
    g = AgentGym(_FakeEnv("Error rates dropped 5% this quarter"), task="report errors")
    g.reset()
    assert g.step({"tool_name": "echo", "arguments": {}}).reward == 0.0
    g2 = AgentGym(_FakeEnv("SQLError: no such table"), task="t")
    g2.reset()
    assert g2.step({"tool_name": "echo", "arguments": {}}).reward == -g2.error_penalty


def test_huge_observations_are_truncated():
    g = AgentGym(_FakeEnv("x" * 100_000), task="t", max_observation_chars=500)
    g.reset()
    out = g.step({"tool_name": "echo", "arguments": {}})
    assert len(out.observation) <= 500 and out.observation.endswith("…")


def test_no_tool_answer_earns_no_verification_credit(gym):
    gym.reset()
    lazy = gym.step({"answer": "A long plausible-sounding answer about revenue " * 3})
    assert lazy.done
    # judge credit only — capped well below a grounded episode's ceiling
    assert lazy.reward <= gym.judge_weight + 1e-9

    grounded_gym = AgentGym(SQLEnvironment(), task=gym.task, seed=11)
    grounded_gym.reset()
    grounded_gym.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
    grounded = grounded_gym.step({"answer": "EMEA leads on revenue."})
    assert grounded.reward > lazy.reward
    grounded_gym.close()


def test_truncation_does_not_double_count_rewards(gym):
    gym.max_steps = 2
    gym.reset()
    gym.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
    out = gym.step({"tool_name": "sql_query", "arguments": {"query": "DROP TABLE sales"}})
    assert out.done and out.info["truncated"]
    traj = out.info["trajectory"]
    assert traj.success is False and traj.final_answer == ""
    # one reward slot per action: [step1, terminal+step2] — the -error_penalty
    # from the truncating step is folded in exactly once
    assert len(gym._rewards) == 2
    assert gym._rewards[1] == out.reward


def test_answer_takes_precedence_over_tool_name(gym):
    gym.reset()
    out = gym.step({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}, "answer": "done"})
    assert out.done  # the answer ends the episode; the tool name is ignored


def test_rollouts_on_one_gym_get_unique_trajectory_ids(gym):
    ids = set()
    for _ in range(3):
        script = iter([{"answer": "quick answer"}])
        episode = gym.rollout(lambda obs, g: next(script))
        ids.add(episode.trajectory.id)
    assert len(ids) == 3


# --- reward functions ---------------------------------------------------------


def test_score_orders_completions_by_quality():
    env = SQLEnvironment()
    tools = env.tools()
    perfect = json.dumps({"tool": "sql_query", "args": {"query": GOOD_SQL}})
    exec_fail = json.dumps({"tool": "sql_query", "args": {"query": "DROP TABLE sales"}})
    missing_args = json.dumps({"tool": "sql_query", "args": {}})
    wrong_tool = json.dumps({"tool": "nope", "args": {"query": GOOD_SQL}})
    garbage = "not json at all"

    scores = [
        score_tool_completion(c, tools, environment=env)[0]
        for c in (perfect, exec_fail, missing_args, wrong_tool, garbage)
    ]
    assert scores[0] == 1.0
    assert scores[0] > scores[1] > scores[2] > scores[3] > scores[4] == 0.0
    env.close()


def test_score_without_execution_renormalizes():
    tools = SQLEnvironment().tools()
    perfect = json.dumps({"tool": "sql_query", "args": {"query": GOOD_SQL}})
    reward, detail = score_tool_completion(perfect, tools, environment=None)
    assert reward == 1.0  # parse + tool + args, renormalized to 1
    assert "execute" not in detail["checks"]


def test_reward_fn_matches_trl_contract():
    env = SQLEnvironment()
    fn = make_reward_fn(environment=env)
    assert fn.__name__ == "agentsynth_verified_reward"
    completions = [
        json.dumps({"tool": "sql_query", "args": {"query": GOOD_SQL}}),
        "garbage",
    ]
    # TRL calls with keyword args plus extra kwargs we must tolerate
    rewards = fn(
        prompts=["p1", "p2"],
        completions=completions,
        completion_ids=[[1], [2]],
        trainer_state=None,
    )
    assert rewards == [1.0, 0.0]
    env.close()


def test_reward_fn_handles_conversational_completions():
    env = SQLEnvironment()
    fn = make_reward_fn(environment=env)
    convo = [
        [
            {
                "role": "assistant",
                "content": json.dumps({"tool": "sql_query", "args": {"query": GOOD_SQL}}),
            }
        ]
    ]
    assert fn(prompts=["p"], completions=convo) == [1.0]
    env.close()


def test_reward_fn_requires_tools_or_env():
    with pytest.raises(ValueError):
        make_reward_fn()


def test_score_accepts_both_naming_conventions():
    env = SQLEnvironment()
    native = json.dumps({"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}})
    assert score_tool_completion(native, env.tools(), environment=env)[0] == 1.0
    env.close()


def test_export_rejects_empty_episode_list(tmp_path):
    with pytest.raises(ValueError):
        episodes_to_grpo_jsonl([], str(tmp_path / "empty.jsonl"))


def test_episodes_export_for_offline_rl(gym, tmp_path):
    script = iter(
        [
            {"tool_name": "sql_query", "arguments": {"query": GOOD_SQL}},
            {"answer": "EMEA leads."},
        ]
    )
    episode = gym.rollout(lambda obs, g: next(script))
    path = str(tmp_path / "episodes.jsonl")
    episodes_to_grpo_jsonl([episode], path)
    row = json.loads(open(path, encoding="utf-8").readline())
    assert row["prompt"] == "revenue by region" or row["prompt"] == gym.task
    assert json.loads(row["completion"])["tool"] == "sql_query"
    assert row["reward"] == episode.total_reward


# --- OpenEnv bridge (skipped without openenv-core) -----------------------------


def test_openenv_bridge_full_episode():
    pytest.importorskip("openenv")
    from openenv.core import CallToolAction

    from agentsynth.rl import FINAL_ANSWER_TOOL, to_openenv

    env = to_openenv(AgentGym(SQLEnvironment(), task="revenue by region", seed=3))
    obs = env.reset(episode_id="ep-1")
    assert obs.done is False
    assert FINAL_ANSWER_TOOL in obs.result["tools"]

    obs = env.step(CallToolAction(tool_name="sql_query", arguments={"query": GOOD_SQL}))
    assert "EMEA" in obs.result and obs.done is False

    obs = env.step(CallToolAction(tool_name=FINAL_ANSWER_TOOL, arguments={"answer": "EMEA leads."}))
    assert obs.done is True
    assert 0.0 <= obs.reward <= 1.0
    assert obs.metadata["verification"]["verified"] is True
    assert "trajectory" not in obs.metadata  # python objects stay out of the wire format

    state = env.state
    assert state.episode_id == "ep-1"
    assert state.step_count == 2
    env.close()


def test_openenv_bridge_needs_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("openenv"):
            raise ImportError("openenv not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from agentsynth.rl import to_openenv

    with pytest.raises(ImportError, match="agentsynth-ai\\[rl\\]"):
        to_openenv(AgentGym(SQLEnvironment(), task="t"))
