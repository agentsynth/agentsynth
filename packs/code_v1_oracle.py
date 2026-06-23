"""Reference solution for the code_v1 pack: one `python` tool call per task.

    agentsynth bench --pack packs/code_v1.yaml --policy packs.code_v1_oracle:solve
    agentsynth pack validate packs/code_v1.yaml
"""

# Each solution is defined in a single python call; the hidden tests run it afterwards.
_SOLUTION = {
    "is-prime": (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    i = 2\n"
        "    while i * i <= n:\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "        i += 1\n"
        "    return True\n"
    ),
    "fizzbuzz": (
        "def fizzbuzz(n):\n"
        "    out = []\n"
        "    for i in range(1, n + 1):\n"
        "        if i % 15 == 0:\n"
        "            out.append('FizzBuzz')\n"
        "        elif i % 3 == 0:\n"
        "            out.append('Fizz')\n"
        "        elif i % 5 == 0:\n"
        "            out.append('Buzz')\n"
        "        else:\n"
        "            out.append(str(i))\n"
        "    return out\n"
    ),
    "two-sum": (
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        if target - x in seen:\n"
        "            return [seen[target - x], i]\n"
        "        seen[x] = i\n"
        "    return []\n"
    ),
    "dedup-preserve-order": (
        "def dedup(xs):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for x in xs:\n"
        "        if x not in seen:\n"
        "            seen.add(x)\n"
        "            out.append(x)\n"
        "    return out\n"
    ),
}

_ANSWER = {
    "is-prime": "Defined is_prime(n) with trial division up to sqrt(n).",
    "fizzbuzz": "Defined fizzbuzz(n) returning the Fizz/Buzz list.",
    "two-sum": "Defined two_sum with a one-pass hash map.",
    "dedup-preserve-order": "Defined dedup keeping first occurrences in order.",
}


def solve(observation, gym):
    sid = gym.scenario.id if gym.scenario is not None else ""
    code = _SOLUTION.get(sid)
    if code and gym.step_count == 0:
        return {"tool_name": "python", "arguments": {"code": code}}
    return {"answer": _ANSWER.get(sid, "Done.")}
