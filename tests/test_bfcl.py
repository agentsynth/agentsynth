import json

import pytest

from agentsynth import AgentTrajectoryGenerator
from agentsynth.benchmarks import (
    SAMPLE_BFCL,
    agentsynth_model,
    bfcl_case,
    load_bfcl,
    load_sample_bfcl,
    run_benchmark,
    run_tau_bench,
    sample_cases,
    tau_bench_available,
)


def test_bfcl_case_parses_question_and_ground_truth():
    case = bfcl_case(*SAMPLE_BFCL[0])
    assert case.expected_tool == "get_weather"
    assert case.expected_args == {"city": ["Paris", "paris"]}
    assert case.query == "What is the weather in Paris?"
    assert [t.name for t in case.tool_specs()] == ["get_weather"]


def test_bfcl_normalizes_dict_type_to_object():
    case = bfcl_case(*SAMPLE_BFCL[0])
    assert case.tools[0]["parameters"]["type"] == "object"


def test_sample_cases_score_with_acceptable_value_lists():
    cases = sample_cases()
    assert len(cases) == 2

    def perfect(query, tools):
        if "weather" in query.lower():
            return "get_weather", {"city": "paris"}  # lowercase still accepted
        return "math_factorial", {"n": 15}

    report = run_benchmark(perfect, cases)
    assert report.score == 1.0
    assert report.arg_accuracy == 1.0


def test_load_bfcl_from_files(tmp_path):
    questions = tmp_path / "q.jsonl"
    answers = tmp_path / "a.jsonl"
    questions.write_text("\n".join(json.dumps(q) for q, _ in SAMPLE_BFCL))
    answers.write_text("\n".join(json.dumps(a) for _, a in SAMPLE_BFCL))
    cases = load_bfcl(str(questions), str(answers))
    assert len(cases) == 2
    assert cases[0].expected_tool == "get_weather"


def test_load_sample_bfcl_returns_the_bundled_real_slice():
    cases = load_sample_bfcl()
    assert len(cases) == 25
    first = cases[0]
    assert first.id == "simple_python_0"
    assert first.expected_tool == "calculate_triangle_area"
    assert "base" in first.expected_args
    assert all(c.query and c.expected_tool for c in cases)


def test_run_benchmark_on_the_bfcl_slice():
    cases = load_sample_bfcl()
    model = agentsynth_model(AgentTrajectoryGenerator(use_mock=True))
    report = run_benchmark(model, cases=cases)
    assert report.n == len(cases)
    assert 0.0 <= report.tool_accuracy <= 1.0
    assert 0.0 <= report.score <= 1.0


def test_load_sample_bfcl_multiple_split_has_candidate_tools():
    cases = load_sample_bfcl(split="multiple")
    assert len(cases) == 25
    # the whole point of this split: every case offers a real choice of tools
    assert all(len(c.tools) >= 2 for c in cases)
    for case in cases:
        assert case.expected_tool in [t["name"] for t in case.tools]


def test_load_sample_bfcl_rejects_unknown_split():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        load_sample_bfcl(split="nope")


def test_tau_bench_reports_missing_package():
    assert tau_bench_available() is False
    with pytest.raises(ImportError):
        run_tau_bench(model="gpt-4o")
