"""Drive a gym episode with an LLM.

`llm_policy(client)` returns a policy for `AgentGym.rollout`: each step it shows
the model the task, the tools, and the transcript so far, and expects either a
`{"tool": ..., "args": ...}` call or `{"answer": ...}` back. Anything unparsable
ends the episode as a plain-text answer.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .episode import AgentGym

_INSTRUCTIONS = (
    'Reply with ONE json object and nothing else: {"tool": "<name>", "args": {...}} '
    'to call a tool, or {"answer": "<final answer>"} when the task is done.'
)


def llm_policy(client: Any, max_transcript_chars: int = 4000) -> Callable[[str, AgentGym], Any]:
    def policy(observation: str, gym: AgentGym) -> Any:
        tools = json.dumps(
            [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in gym.environment.tools()
            ]
        )
        transcript = gym.transcript(max_transcript_chars)
        prompt = (
            f"Task: {gym.task}\n\nTools (JSON): {tools}\n\n"
            + (f"Steps so far:\n{transcript}\n\n" if transcript else "")
            + _INSTRUCTIONS
        )
        reply = client.complete([{"role": "user", "content": prompt}])
        return reply if reply else {"answer": ""}

    return policy


__all__ = ["llm_policy"]
