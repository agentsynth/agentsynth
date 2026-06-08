from agentsynth import AgentTrajectoryGenerator
from agentsynth.benchmarks import (
    BUILTIN_CASES,
    BenchmarkCase,
    agentsynth_model,
    compare_models,
    report_table_md,
    run_benchmark,
)


def _fixed_model(tool, args=None):
    def fn(query, tools):
        return tool, dict(args or {})

    return fn


def test_builtin_cases_are_valid():
    assert len(BUILTIN_CASES) >= 10
    for case in BUILTIN_CASES:
        assert case.expected_tool
        assert case.tool_specs()


def test_correct_tool_and_args_scores_one():
    case = BenchmarkCase(
        id="x",
        query="weather in Paris",
        expected_tool="get_weather",
        expected_args={"city": "Paris"},
    )
    report = run_benchmark(_fixed_model("get_weather", {"city": "Paris"}), [case])
    assert report.score == 1.0
    assert report.tool_accuracy == 1.0
    assert report.arg_accuracy == 1.0


def test_right_tool_wrong_args_scores_half():
    case = BenchmarkCase(
        id="x",
        query="weather in Paris",
        expected_tool="get_weather",
        expected_args={"city": "Paris"},
    )
    report = run_benchmark(_fixed_model("get_weather", {"city": "London"}), [case])
    assert report.results[0].tool_ok
    assert not report.results[0].args_ok
    assert report.results[0].score == 0.5


def test_wrong_tool_scores_zero():
    case = BenchmarkCase(id="x", query="weather in Paris", expected_tool="get_weather")
    report = run_benchmark(_fixed_model("calculator", {}), [case])
    assert report.results[0].score == 0.0


def test_key_presence_only_arg():
    case = BenchmarkCase(
        id="x", query="calc", expected_tool="calculator", expected_args={"expression": None}
    )
    assert run_benchmark(_fixed_model("calculator", {"expression": "2+2"}), [case]).score == 1.0
    assert run_benchmark(_fixed_model("calculator", {}), [case]).results[0].score == 0.5


def test_model_exception_scores_zero():
    def boom(query, tools):
        raise RuntimeError("model failure")

    assert run_benchmark(boom, BUILTIN_CASES[:3]).score == 0.0


def test_agentsynth_model_runs_full_benchmark():
    report = run_benchmark(agentsynth_model(AgentTrajectoryGenerator(use_mock=True)))
    assert report.n == len(BUILTIN_CASES)
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.tool_accuracy <= 1.0


def test_compare_models_and_table():
    gen = AgentTrajectoryGenerator(use_mock=True)
    comparison = compare_models(_fixed_model("web_search", {"query": "x"}), agentsynth_model(gen))
    assert {"before", "after", "delta_score", "delta_tool_accuracy"} <= set(comparison)
    table = report_table_md(comparison)
    assert "Before" in table and "After" in table and "Tool accuracy" in table


def test_prompted_model_parses_a_tool_call():
    from agentsynth import default_tool_catalog
    from agentsynth.benchmarks import prompted_model

    model_fn = prompted_model(lambda p: 'Sure! {"tool": "get_weather", "args": {"city": "Paris"}}')
    tool, args = model_fn("weather in Paris", default_tool_catalog())
    assert tool == "get_weather"
    assert args == {"city": "Paris"}


def test_prompted_model_handles_non_json():
    from agentsynth.benchmarks import prompted_model

    tool, args = prompted_model(lambda p: "no json at all")("x", [])
    assert tool is None and args == {}
