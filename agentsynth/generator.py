"""Synthetic agent trajectory generation.

`AgentTrajectoryGenerator` builds multi-step trajectories in three modes:
single_agent (tool use), code_execution (grounded Python), and multi_agent
(planner/executor/critic). With no API key it runs fully offline, deriving all
of its "randomness" from `stable_seed` so identical inputs give identical output
(the test suite relies on this). With a real LLM configured it asks the model for
a structured trajectory and falls back to the mock builder on any failure.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from .schemas import Trajectory, TrajectoryStep
from .utils import (
    LLMClient,
    PythonREPL,
    default_tool_catalog,
    extract_json,
    parse_tool_catalog,
    stable_seed,
)

if TYPE_CHECKING:
    from .environments.base import Environment

_TRUTHY = {"1", "true", "yes", "on"}

_VARY_MODE_CYCLE = ("single_agent", "multi_agent", "code_execution")

# Maps a lower-cased substring of an arg name to a value "kind". The kinds
# arith/query/location are computed from the query; anything else is the literal
# value to use.
_ARG_NAME_HINTS = (
    ("expression", "arith"),
    ("query", "query"),
    ("search", "query"),
    ("keyword", "query"),
    ("city", "location"),
    ("town", "location"),
    ("location", "location"),
    ("destination", "location"),
    ("place", "location"),
    ("path", "data/report.csv"),
    ("file", "data/report.csv"),
    ("email", "team@example.com"),
    ("recipient", "team@example.com"),
    ("subject", "Status update"),
    ("body", "Here is the latest status update you requested."),
    ("message", "Here is the latest status update you requested."),
    ("url", "https://example.com"),
    ("name", "report"),
    ("to", "team@example.com"),
)

# These hints must match the whole arg name (or a `_`-suffixed token) rather than
# as a loose substring, so e.g. "token" doesn't match "to".
_EXACT_HINT_NEEDLES = frozenset({"to", "cc", "bcc"})

_KNOWN_CITIES = (
    "new york",
    "san francisco",
    "los angeles",
    "hong kong",
    "mexico city",
    "são paulo",
    "sao paulo",
    "ho chi minh city",
    "rio de janeiro",
    "cape town",
    "paris",
    "london",
    "tokyo",
    "berlin",
    "sydney",
    "toronto",
    "singapore",
    "mumbai",
    "delhi",
    "beijing",
    "shanghai",
    "cairo",
    "dubai",
    "madrid",
    "rome",
    "amsterdam",
    "seattle",
    "austin",
    "boston",
    "chicago",
    "lagos",
    "nairobi",
    "bangkok",
    "seoul",
    "hanoi",
    "jakarta",
    "istanbul",
    "moscow",
    "vienna",
    "zurich",
    "oslo",
    "stockholm",
    "helsinki",
    "dublin",
    "lisbon",
)
# Longest first so multi-word cities match as a unit.
_KNOWN_CITIES_BY_LEN = tuple(sorted(_KNOWN_CITIES, key=len, reverse=True))

_FALLBACK_CITIES = (
    "Paris",
    "Tokyo",
    "London",
    "Berlin",
    "Singapore",
    "Toronto",
    "Sydney",
    "Madrid",
)

# Capitalised proper noun following a locative preposition, e.g. "in Tokyo".
_LOCATION_AFTER_RE = re.compile(
    r"\b(?:in|at|for|near|to|from)\s+([A-Z][\w.\-]+(?:\s+[A-Z][\w.\-]+)?)"
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "i",
        "you",
        "we",
        "it",
        "this",
        "that",
        "what",
        "whats",
        "how",
        "do",
        "does",
        "can",
        "please",
        "me",
        "my",
        "your",
        "our",
        "from",
        "by",
        "at",
        "as",
        "about",
        "into",
    }
)


def _tokens(text: str) -> List[str]:
    """Lower-case word tokens with stop-words removed (order preserved)."""
    return [t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _relevant_tools(query: str, tools: Sequence[Any], k: int, seed: int) -> List[Any]:
    """Up to `k` tools most relevant to the query.

    Ranked by token overlap between the query and each tool's name + description.
    Ties and the no-overlap case break on `seed` so the pick is reproducible.
    """
    if not tools:
        return []
    k = max(1, min(k, len(tools)))
    q_tokens = set(_tokens(query))

    scored: List[tuple] = []
    for idx, tool in enumerate(tools):
        name = getattr(tool, "name", "") or ""
        desc = getattr(tool, "description", "") or ""
        t_tokens = set(_tokens(name)) | set(_tokens(desc))
        overlap = len(q_tokens & t_tokens)
        # Hash of (seed, name) pseudo-shuffles equal-overlap tools while staying stable.
        tiebreak = stable_seed(seed, name, idx)
        scored.append((-overlap, tiebreak, idx, tool))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return [entry[3] for entry in scored[:k]]


def _short_query_phrase(query: str, max_words: int = 8) -> str:
    """A trimmed, single-line version of the query for embedding in args."""
    cleaned = " ".join((query or "").split())
    words = cleaned.split(" ")
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned or "the requested task"


def _numbers_in(query: str) -> List[float]:
    out: List[float] = []
    for m in _NUMBER_RE.findall(query or ""):
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


def _arith_expression(query: str, seed: int) -> str:
    """A small, safe arithmetic expression tied to the query."""
    nums = [int(n) for n in _numbers_in(query) if abs(n) < 1e6]
    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        return f"{a} + {b}"
    if len(nums) == 1:
        return f"{nums[0]} * 2"
    a = (seed % 20) + 2
    b = (seed // 7 % 9) + 1
    return f"{a} * ({b} + 1)"


def _hint_matches(needle: str, lname: str) -> bool:
    if needle in _EXACT_HINT_NEEDLES:
        return lname == needle or lname.endswith("_" + needle)
    return needle in lname


def _extract_location(query: str, seed: int) -> str:
    """Pick a plausible location for a city/location arg, grounded in the query.

    Prefers a known city named in the query, then a capitalised proper noun after
    a locative preposition ("...in Springfield"), and finally a seed-varied
    fallback so it isn't always "Paris".
    """
    q = query or ""
    ql = q.lower()
    for city in _KNOWN_CITIES_BY_LEN:
        if re.search(r"\b" + re.escape(city) + r"\b", ql):
            return city.title()
    match = _LOCATION_AFTER_RE.search(q)
    if match:
        cand = " ".join(match.group(1).split())
        if len(cand) >= 3 and cand.lower() not in _STOPWORDS:
            return cand
    return _FALLBACK_CITIES[seed % len(_FALLBACK_CITIES)]


def _synth_arg(name: str, schema: Dict[str, Any], query: str, seed: int) -> Any:
    """Synthesize a type-appropriate value for one tool argument.

    `schema` is the JSON-schema fragment for the arg. For strings, name hints win
    so the trajectory reads naturally (`city` -> a city in the query, `path` ->
    "data/report.csv").
    """
    schema = schema if isinstance(schema, dict) else {}
    jtype = schema.get("type")
    enum = schema.get("enum")

    # Enums override the declared type.
    if isinstance(enum, list) and enum:
        return enum[0]

    lname = (name or "").lower()

    if jtype in ("integer", "number"):
        val = (seed % 7) + 2
        return int(val) if jtype == "integer" else float(val)
    if jtype == "boolean":
        return True
    if jtype == "array":
        item_schema = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
        return [_synth_arg(name, item_schema, query, seed)]
    if jtype == "object":
        return {}

    # Fall through to string (also covers missing/unknown types).
    for needle, kind in _ARG_NAME_HINTS:
        if _hint_matches(needle, lname):
            if kind == "arith":
                return _arith_expression(query, seed)
            if kind == "query":
                return _short_query_phrase(query, max_words=12) or "latest information"
            if kind == "location":
                return _extract_location(query, seed)
            return kind
    return _short_query_phrase(query, max_words=6)


def _synth_tool_args(tool: Any, query: str, seed: int) -> Dict[str, Any]:
    """Fill all required args, plus the first optional one, for `tool`."""
    args: Dict[str, Any] = {}
    required = tool.required_args()
    for i, arg in enumerate(required):
        args[arg] = _synth_arg(arg, tool.arg_schema(arg), query, stable_seed(seed, arg, i))
    # Toss in one optional arg half the time for variety.
    optional = [a for a in tool.arg_names() if a not in args]
    if optional and (seed % 2 == 0):
        extra = optional[0]
        args[extra] = _synth_arg(extra, tool.arg_schema(extra), query, stable_seed(seed, extra))
    return args


def _format_observation(tool: Any, args: Dict[str, Any], query: str, seed: int) -> str:
    """A plausible observation string that references `args`."""
    name = getattr(tool, "name", "tool")
    arg_repr = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "(no arguments)"

    if name == "get_weather":
        city = (
            args.get("city")
            or args.get("location")
            or args.get("town")
            or args.get("destination")
            or "the city"
        )
        temp = (seed % 18) + 12
        return f"Weather for {city}: {temp}°C, partly cloudy with light winds."
    if name == "calculator":
        expr = args.get("expression", "")
        result = PythonREPL().run(str(expr)) if expr else ""
        if result and "Error" not in result and "error" not in result:
            return f"The expression {expr!r} evaluates to {result}."
        return f"Evaluated the expression {expr!r}."
    if name == "web_search":
        q = args.get("query", _short_query_phrase(query))
        return (
            f"Top results for {q!r}: (1) An authoritative overview article, "
            f"(2) a recent summary with key figures, (3) a related FAQ entry."
        )
    if name == "sql_query":
        n = (seed % 5) + 1
        return f"Query returned {n} row(s); the leading record summarises the requested metric."
    if name == "read_file":
        path = args.get("path", "the file")
        return f"Read {path}: the file contains the tabular data relevant to the task."
    if name == "send_email":
        to = args.get("to", "the recipient")
        return f"Email successfully sent to {to}."
    return f"{name} returned a successful result for arguments {arg_repr}."


def _synth_code(query: str, seed: int) -> str:
    """Short, safe Python for a code_execution step.

    Only whitelisted modules so it runs cleanly inside `PythonREPL`. Summarises any
    numbers in the query; otherwise runs a small deterministic computation.
    """
    nums = _numbers_in(query)
    if nums:
        as_ints = all(float(n).is_integer() for n in nums)
        values = [int(n) for n in nums] if as_ints else nums
        return (
            "import statistics\n"
            f"values = {values}\n"
            "total = sum(values)\n"
            "mean = statistics.mean(values)\n"
            'print(f"count={len(values)} total={total} mean={mean:.2f}")\n'
        )

    n = (seed % 8) + 5
    return (
        "import math\n"
        f"n = {n}\n"
        "squares = [i * i for i in range(1, n + 1)]\n"
        "total = sum(squares)\n"
        'print(f"n={n} sum_of_squares={total} sqrt_total={math.sqrt(total):.3f}")\n'
    )


class AgentTrajectoryGenerator:
    """Generate synthetic agent trajectories, offline-mock by default.

    `model` is the model id for `LLMClient` (None auto-detects a provider from the
    environment); `temperature` and `max_tokens` are forwarded to it. `max_steps`
    caps the number of tool calls / reasoning steps. `use_mock` is "auto" (use the
    LLM if available, else mock), True (always mock), or False (require the LLM,
    but fall back to mock and set `warning` if no client is available). `seed` is
    mixed into every mock decision. Pass `llm_client` to reuse an existing client.
    `tools` sets a default catalog (anything `parse_tool_catalog` accepts, or a
    list of ToolSpec) used when a call doesn't pass its own; None means the
    built-in default.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_steps: int = 6,
        use_mock: Union[str, bool] = "auto",
        seed: int = 0,
        llm_client: Optional[LLMClient] = None,
        max_tokens: int = 1536,
        tools: Optional[Any] = None,
        environment: Optional["Environment"] = None,
    ) -> None:
        self.client = llm_client or LLMClient(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
        self.temperature = temperature
        self.max_steps = max(1, int(max_steps))
        self.seed = seed
        self.environment = environment
        self._env_tools = set(environment.tool_names()) if environment is not None else set()
        self.warning: Optional[str] = None
        self.tools = self._coerce_tools(tools) if tools is not None else None

        env_force = str(os.environ.get("AGENTSYNTH_FORCE_MOCK", "")).strip().lower() in _TRUTHY
        force_mock = (use_mock is True) or env_force
        self.use_llm = (not force_mock) and (use_mock is not True) and self.client.available

        # use_mock=False demands the LLM; if there's none, warn and use mock anyway.
        if use_mock is False and not self.client.available:
            self.use_llm = False
            self.warning = (
                "LLM was requested (use_mock=False) but no LLM client is available "
                f"({self.client.last_error or 'no provider configured'}); "
                "falling back to deterministic mock generation."
            )

    def generate(
        self,
        query: str,
        tools: Optional[Any] = None,
        mode: str = "single_agent",
        domain: Optional[str] = None,
        index: int = 0,
    ) -> Trajectory:
        """Generate a single trajectory for `query` in `mode`."""
        tool_specs = self._coerce_tools(tools)
        mode = mode if mode in _VARY_MODE_CYCLE else "single_agent"
        seed = stable_seed(query, index, mode, self.seed)

        if self.use_llm:
            traj = self._llm_build(query, tool_specs, mode, domain, seed, index)
            if traj is not None:
                return traj
            # LLM gave us nothing usable; fall through to mock.

        return self._mock_build(query, tool_specs, mode, domain, seed, index)

    def generate_batch(
        self,
        queries: Union[str, Sequence[str]],
        tools: Optional[Any] = None,
        mode: str = "single_agent",
        num_trajectories: Optional[int] = None,
        domains: Optional[Sequence[Optional[str]]] = None,
        progress: Optional[Callable[..., Any]] = None,
        vary_modes: bool = False,
    ) -> List[Trajectory]:
        """Generate many trajectories from one or more queries.

        A list of queries gives one trajectory each (cycled or truncated to
        `num_trajectories` when set). A single query string gives
        `num_trajectories` deterministic variations.
        """
        tool_specs = self._coerce_tools(tools)

        plan = self._build_plan(queries, num_trajectories)
        total = len(plan)
        domains = list(domains) if domains is not None else None

        results: List[Trajectory] = []
        for i, (q, idx) in enumerate(plan):
            cur_mode = _VARY_MODE_CYCLE[idx % len(_VARY_MODE_CYCLE)] if vary_modes else mode
            domain = domains[i] if domains is not None and i < len(domains) else None
            self._report(progress, i, total)
            results.append(
                self.generate(q, tools=tool_specs, mode=cur_mode, domain=domain, index=idx)
            )
        self._report(progress, total, total, final=True)
        return results

    @staticmethod
    def _build_plan(
        queries: Union[str, Sequence[str]], num_trajectories: Optional[int]
    ) -> List[tuple]:
        """The list of `(query, index)` pairs describing the batch."""
        if isinstance(queries, str):
            n = num_trajectories if num_trajectories is not None else 1
            n = max(0, int(n))
            return [(queries, i) for i in range(n)]

        q_list = [str(q) for q in queries]
        if not q_list:
            return []
        if num_trajectories is None:
            return [(q, i) for i, q in enumerate(q_list)]

        n = max(0, int(num_trajectories))
        return [(q_list[i % len(q_list)], i) for i in range(n)]

    @staticmethod
    def _report(
        progress: Optional[Callable[..., Any]], i: int, total: int, final: bool = False
    ) -> None:
        """Call a Gradio-style `progress(frac, desc=...)` callback, tolerating bad ones."""
        if not callable(progress):
            return
        denom = total or 1
        frac = 1.0 if final else (i / denom)
        desc = "Generation complete" if final else f"Generating {i + 1}/{total}"
        try:
            progress(frac, desc=desc)
        except Exception:
            # A broken callback shouldn't sink the whole run; retry without desc.
            try:
                progress(frac)
            except Exception:
                pass

    def _coerce_tools(self, tools: Optional[Any]) -> List[Any]:
        """Normalise `tools` into a list of ToolSpec.

        None uses the per-instance default catalog (`self.tools`) if set, else the
        built-in default.
        """
        if tools is None:
            default = getattr(self, "tools", None)
            if default:
                return list(default)
            if self.environment is not None:
                return list(self.environment.tools())
            return default_tool_catalog()
        # Already a list of ToolSpec? Duck-typed on the methods we call.
        if isinstance(tools, list) and all(
            hasattr(t, "required_args") and hasattr(t, "arg_names") for t in tools
        ):
            return list(tools) if tools else default_tool_catalog()
        parsed = parse_tool_catalog(tools)
        return parsed if parsed else default_tool_catalog()

    def _call_tool(self, tool: Any, query: str, seed: int) -> tuple:
        """Produce (args, observation) for a tool call.

        If an environment owns this tool, run it for a real observation;
        otherwise synthesize plausible args and a templated observation.
        """
        env = self.environment
        if env is not None and tool.name in self._env_tools:
            args = env.sample_args(tool.name, query, seed) or _synth_tool_args(tool, query, seed)
            try:
                obs = env.execute(tool.name, args)
            except Exception as exc:
                obs = f"{tool.name} error: {exc}"
            return args, obs
        args = _synth_tool_args(tool, query, seed)
        return args, _format_observation(tool, args, query, seed)

    def _wrap(
        self,
        query: str,
        tools: List[Any],
        mode: str,
        domain: Optional[str],
        steps: List[TrajectoryStep],
        final_answer: str,
        generator_model: str,
        index: int,
        seed: int,
        success: bool = True,
    ) -> Trajectory:
        """Assemble the final Trajectory from its parts.

        The id is derived from the generation seed so the same inputs reproduce the
        same trajectory id (and therefore the same downstream eval scores).
        """
        return Trajectory(
            id=format(seed & 0xFFFFFFFFFFFF, "012x"),
            query=query,
            mode=mode,
            domain=domain,
            tools=tools,
            steps=steps,
            final_answer=final_answer,
            success=success,
            generator_model=generator_model,
            metadata={
                "synthetic": True,
                "mode": mode,
                "n_steps": len(steps),
                "generator": generator_model,
                "index": index,
            },
        )

    def _mock_build(
        self,
        query: str,
        tools: List[Any],
        mode: str,
        domain: Optional[str],
        seed: int,
        index: int,
    ) -> Trajectory:
        if mode == "code_execution":
            return self._mock_code_execution(query, tools, domain, seed, index)
        if mode == "multi_agent":
            return self._mock_multi_agent(query, tools, domain, seed, index)
        return self._mock_single_agent(query, tools, domain, seed, index)

    def _mock_single_agent(
        self,
        query: str,
        tools: List[Any],
        domain: Optional[str],
        seed: int,
        index: int,
    ) -> Trajectory:
        k = min(self.max_steps, len(tools), 4)
        k = max(1, k)
        chosen = _relevant_tools(query, tools, k, seed)

        steps: List[TrajectoryStep] = []
        observations: List[str] = []
        for i, tool in enumerate(chosen):
            tseed = stable_seed(seed, getattr(tool, "name", ""), i)
            args, obs = self._call_tool(tool, query, tseed)
            observations.append(obs)
            steps.append(
                TrajectoryStep(
                    step_type="thought",
                    thought=(
                        f"I'll use the {tool.name} tool to make progress on: "
                        f"{_short_query_phrase(query)}."
                    ),
                )
            )
            steps.append(TrajectoryStep(step_type="tool_call", tool_name=tool.name, tool_args=args))
            steps.append(TrajectoryStep(step_type="observation", observation=obs))

        final = self._synth_final_answer(query, observations)
        steps.append(TrajectoryStep(step_type="final_answer", content=final))
        return self._wrap(query, tools, "single_agent", domain, steps, final, "mock", index, seed)

    def _mock_code_execution(
        self,
        query: str,
        tools: List[Any],
        domain: Optional[str],
        seed: int,
        index: int,
    ) -> Trajectory:
        code = _synth_code(query, seed)
        output = PythonREPL().run(code)

        steps: List[TrajectoryStep] = [
            TrajectoryStep(
                step_type="thought",
                thought=(
                    "I'll write and run a short Python snippet to compute the answer to: "
                    f"{_short_query_phrase(query)}."
                ),
            ),
            TrajectoryStep(step_type="code_execution", code=code, code_output=output),
        ]
        clean_out = (output or "").strip() or "no output"
        final = (
            f"Running the computation produced: {clean_out}. "
            "This grounds the answer in the actual program output."
        )
        steps.append(TrajectoryStep(step_type="final_answer", content=final))
        return self._wrap(query, tools, "code_execution", domain, steps, final, "mock", index, seed)

    def _mock_multi_agent(
        self,
        query: str,
        tools: List[Any],
        domain: Optional[str],
        seed: int,
        index: int,
    ) -> Trajectory:
        n_calls = (seed % 3) + 1  # 1..3 tool_call/observation pairs
        n_calls = max(1, min(n_calls, max(1, len(tools))))
        chosen = _relevant_tools(query, tools, n_calls, seed)

        phrase = _short_query_phrase(query)
        plan_text = (
            "Plan:\n"
            f"- Clarify the goal: {phrase}.\n"
            f"- Gather the needed information using {', '.join(t.name for t in chosen)}.\n"
            "- Synthesize the findings into a concise final answer."
        )

        steps: List[TrajectoryStep] = [
            TrajectoryStep(
                step_type="plan",
                agent="planner",
                content=plan_text,
            )
        ]

        observations: List[str] = []
        for i, tool in enumerate(chosen):
            tseed = stable_seed(seed, getattr(tool, "name", ""), i)
            args, obs = self._call_tool(tool, query, tseed)
            observations.append(obs)
            steps.append(
                TrajectoryStep(
                    step_type="tool_call",
                    agent="executor",
                    tool_name=tool.name,
                    tool_args=args,
                )
            )
            steps.append(TrajectoryStep(step_type="observation", agent="executor", observation=obs))

        steps.append(
            TrajectoryStep(
                step_type="critique",
                agent="critic",
                content=(
                    "Critique: the executor's tool calls are well-targeted and the "
                    "observations directly support the conclusion; no contradictions "
                    "or missing steps were found."
                ),
            )
        )

        final = self._synth_final_answer(query, observations)
        steps.append(TrajectoryStep(step_type="final_answer", agent="planner", content=final))
        return self._wrap(query, tools, "multi_agent", domain, steps, final, "mock", index, seed)

    @staticmethod
    def _synth_final_answer(query: str, observations: List[str]) -> str:
        """A 1-3 sentence answer that references the gathered observations."""
        phrase = _short_query_phrase(query)
        if not observations:
            return f"Based on the analysis, here is the answer regarding {phrase}."
        first = observations[0].rstrip(".")
        if len(observations) == 1:
            return f"Based on the tool results, {first}. This addresses the request about {phrase}."
        return (
            f"Combining the tool results — {first}, among others — I can now "
            f"answer the request about {phrase}. The observations above directly "
            "support this conclusion."
        )

    def _llm_build(
        self,
        query: str,
        tools: List[Any],
        mode: str,
        domain: Optional[str],
        seed: int,
        index: int,
    ) -> Optional[Trajectory]:
        """Ask the LLM for a structured trajectory; None on any failure."""
        from .utils import tool_catalog_to_json  # local import keeps top light

        tools_json = tool_catalog_to_json(tools)
        system = self._llm_system_prompt(mode, tools_json)
        user = (
            f"User query: {query}\n"
            f"Mode: {mode}\n"
            f"Max tool calls / steps: {self.max_steps}\n"
            "Respond with ONLY the JSON object described above."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        reply = self.client.complete(messages)
        if not reply:
            return None

        parsed = extract_json(reply)
        if not isinstance(parsed, dict):
            return None

        raw_steps = parsed.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return None

        steps = self._parse_llm_steps(raw_steps)
        if not steps:
            return None

        final_answer = parsed.get("final_answer")
        if not isinstance(final_answer, str) or not final_answer.strip():
            final_answer = self._final_from_steps(steps, query)

        return self._wrap(
            query,
            tools,
            mode,
            domain,
            steps,
            final_answer.strip(),
            self.client.model or "llm",
            index,
            seed,
        )

    @staticmethod
    def _llm_system_prompt(mode: str, tools_json: str) -> str:
        agent_note = (
            'For multi_agent mode, set an "agent" field on every step '
            '(e.g. "planner", "executor", "critic").'
            if mode == "multi_agent"
            else ""
        )
        return (
            "You are a data generator that produces ONE high-quality synthetic "
            "agent trajectory for fine-tuning agentic LLMs. Use ONLY the tools "
            "provided below — never invent tool names.\n\n"
            f"Available tools (JSON):\n{tools_json}\n\n"
            "Respond with exactly ONE JSON object of the form:\n"
            '{"steps": [{"step_type": "...", "thought": "...", "tool_name": "...", '
            '"tool_args": {...}, "observation": "...", "code": "...", '
            '"code_output": "...", "content": "..."}], "final_answer": "..."}\n\n'
            "Each step_type must be one of: thought, plan, tool_call, observation, "
            "code_execution, critique, final_answer. Only populate the fields "
            "relevant to a step's type. Keep observations consistent with the tool "
            "arguments, and make the final_answer reference what was observed. "
            f"{agent_note}\n"
            "Do not include any prose outside the JSON object."
        )

    def _parse_llm_steps(self, raw_steps: List[Any]) -> List[TrajectoryStep]:
        """Validate raw step dicts into TrajectoryStep, grounding code output."""
        steps: List[TrajectoryStep] = []
        repl = PythonREPL()
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            try:
                step = TrajectoryStep(**raw)
            except Exception:
                continue  # skip malformed steps
            # Re-run code_execution steps in a real REPL so the output is genuine.
            if step.step_type == "code_execution" and step.code:
                step.code_output = repl.run(step.code)
            steps.append(step)
        return steps

    @staticmethod
    def _final_from_steps(steps: List[TrajectoryStep], query: str) -> str:
        """Derive a final answer from a final_answer step, or the last content."""
        for step in reversed(steps):
            if step.step_type == "final_answer" and (step.content or step.thought):
                return step.content or step.thought or ""
        for step in reversed(steps):
            if step.content:
                return step.content
        return f"Completed the requested task: {_short_query_phrase(query)}."


__all__ = ["AgentTrajectoryGenerator"]
