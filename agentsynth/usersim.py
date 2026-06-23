"""Multi-turn, user-simulator scenarios — evaluate a conversation, grade the world.

τ²-bench's lesson: real agent work is a multi-turn exchange with a user, not a single
task, and what matters is whether the world ends up right after the *whole* conversation.
A scenario carries its extra user turns in `metadata["user_turns"]`; the environment
persists across turns, the agent's tool calls accumulate, and the outcome checks run once
at the end on the final state — so an agent that fixes turn one but breaks it on turn three
fails, the way it should.

    scenario = Scenario(
        id="refund-then-cancel",
        task="Refund order 7.",
        metadata={"user_turns": ["Actually, also cancel order 8."]},
        environment={...}, checkers=[...],
    )
    result = run_conversation(policy, scenario)
    result.passed, result.n_turns

The user here is a scripted sequence (deterministic, the testable default); swapping in an
LLM user-simulator is a drop-in replacement for that list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schemas import Trajectory, TrajectoryStep


class ConversationContext:
    """What a policy sees mid-conversation — mirrors the `AgentGym` surface policies use.

    `turn` is the 0-based user turn, `step_count` resets each turn, `task` is the current
    user message, and `history` is the running transcript.
    """

    def __init__(self, scenario: Any, environment: Any) -> None:
        self.scenario = scenario
        self.environment = environment
        self.task = scenario.task
        self.turn = 0
        self.step_count = 0
        self.history: List[Dict[str, str]] = []

    def tool_names(self) -> List[str]:
        return self.environment.tool_names()


class ConversationTurn(BaseModel):
    user: str
    agent: str
    tool_calls: int


class ConversationResult(BaseModel):
    scenario_id: str
    passed: bool
    outcome_score: float
    n_turns: int
    turns: List[ConversationTurn] = Field(default_factory=list)
    checks: List[Dict[str, Any]] = Field(default_factory=list)


def _user_turns(scenario: Any) -> List[str]:
    extra = scenario.metadata.get("user_turns") or []
    return [scenario.task] + [str(t) for t in extra]


def run_conversation(
    policy: Any,
    scenario: Any,
    seed: int = 7,
    max_steps_per_turn: Optional[int] = None,
) -> ConversationResult:
    """Run a policy through a scenario's user turns against one persistent world."""
    environment = scenario.build_environment()
    max_steps = max_steps_per_turn or scenario.max_steps
    ctx = ConversationContext(scenario, environment)

    steps: List[TrajectoryStep] = []
    turns: List[ConversationTurn] = []
    last_reply = ""

    try:
        for turn_index, user_message in enumerate(_user_turns(scenario)):
            ctx.turn = turn_index
            ctx.task = user_message
            ctx.step_count = 0
            ctx.history.append({"role": "user", "content": user_message})
            observation = user_message
            reply = ""
            tool_calls = 0

            while ctx.step_count < max_steps:
                action = policy(observation, ctx)
                answer = action.get("answer") if isinstance(action, dict) else None
                if answer is not None:
                    reply = str(answer)
                    break
                tool_name = str(action.get("tool_name", "")) if isinstance(action, dict) else ""
                args = (action.get("arguments") if isinstance(action, dict) else None) or {}
                if tool_name not in environment.tool_names():
                    observation = f"InvalidAction: unknown tool {tool_name!r}"
                else:
                    try:
                        observation = environment.execute(tool_name, args)
                    except Exception as exc:
                        observation = f"{type(exc).__name__}: {exc}"
                steps.append(
                    TrajectoryStep(step_type="tool_call", tool_name=tool_name, tool_args=args)
                )
                steps.append(TrajectoryStep(step_type="observation", observation=observation))
                tool_calls += 1
                ctx.step_count += 1

            steps.append(TrajectoryStep(step_type="final_answer", content=reply))
            ctx.history.append({"role": "assistant", "content": reply})
            turns.append(ConversationTurn(user=user_message, agent=reply, tool_calls=tool_calls))
            last_reply = reply

        trajectory = Trajectory(
            id=f"convo-{scenario.id}",
            query=scenario.task,
            mode="single_agent",
            tools=environment.tools(),
            steps=steps,
            final_answer=last_reply,
            success=bool(last_reply),
            generator_model="usersim",
            metadata={"source": "usersim", "turns": len(turns)},
        )
        score, outcomes = scenario.run_checks(environment, trajectory)
    finally:
        environment.close()

    return ConversationResult(
        scenario_id=scenario.id,
        passed=score >= 1.0,
        outcome_score=score,
        n_turns=len(turns),
        turns=turns,
        checks=[o.model_dump() for o in outcomes],
    )


def run_conversation_suite(policy: Any, scenarios: List[Any], seed: int = 7) -> Dict[str, Any]:
    """Run a policy through every conversation scenario; a scenario passes on its end state."""
    results = [run_conversation(policy, s, seed=seed) for s in scenarios]
    passed = sum(1 for r in results if r.passed)
    n = len(results) or 1
    return {
        "n": len(results),
        "passed": passed,
        "pass_rate": round(passed / n, 4),
        "results": [r.model_dump() for r in results],
    }


__all__ = [
    "ConversationContext",
    "ConversationTurn",
    "ConversationResult",
    "run_conversation",
    "run_conversation_suite",
]
