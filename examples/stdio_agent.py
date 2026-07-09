"""A bench agent in twenty lines: one JSON line in, one JSON line out.

Point the bench at it and your process drives every episode:

    agentsynth bench --pack core_v1 --agent "python examples/stdio_agent.py"

Each request carries the task, the tool catalog, the newest observation, and a
transcript of the steps so far. Reply {"tool": ..., "args": {...}} to act, or
{"answer": "..."} to finish the episode. Swap the body of `act` for your real
agent — your framework, your model, any language that can read a line and print
one back.
"""

import json
import sys


def act(request):
    # Placeholder brain: peek at the world once, then wrap up. Yours goes here.
    if request["step"] == 1 and request["tools"]:
        tool = request["tools"][0]
        if tool["name"] == "sql_query":
            return {"tool": "sql_query", "args": {"query": "SELECT name FROM sqlite_master"}}
        return {"tool": tool["name"], "args": {}}
    return {"answer": f"looked around but have no plan for: {request['task'][:80]}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        print(json.dumps(act(json.loads(line))), flush=True)


if __name__ == "__main__":
    main()
