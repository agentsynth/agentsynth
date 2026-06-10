"""Regression tests for bugs found in the pre-release review."""

from agentsynth.environments import SQLEnvironment
from agentsynth.pipelines import Recipe, run_recipe
from agentsynth.utils import DEFAULT_TOOL_CATALOG, default_tool_catalog, parse_tool_catalog


def test_tool_catalog_is_not_aliased():
    # Editing one parsed tool's schema must not leak into other copies or the global.
    a = default_tool_catalog()
    b = default_tool_catalog()
    a[0].parameters["properties"]["__leak__"] = 1
    assert "__leak__" not in b[0].parameters.get("properties", {})
    assert "__leak__" not in DEFAULT_TOOL_CATALOG[0]["parameters"]["properties"]


def test_parse_tool_catalog_does_not_mutate_input():
    raw = [{"name": "t", "parameters": {"properties": {"a": {"type": "string"}}}}]
    parse_tool_catalog(raw)
    assert "type" not in raw[0]["parameters"]  # input left untouched


def test_recipe_uses_all_explicit_queries():
    queries = [f"analyze data set number {i} by region" for i in range(15)]
    result = run_recipe(Recipe(queries=queries, evaluate=False))
    assert len(result.trajectories) == 15  # not truncated to the default num_trajectories


def test_recipe_explicit_count_still_caps_queries():
    queries = ["one query here", "another distinct query"]
    result = run_recipe(Recipe(queries=queries, num_trajectories=5, evaluate=False))
    assert len(result.trajectories) == 5  # explicit count wins


def test_sql_multistatement_does_not_raise():
    out = SQLEnvironment().execute("sql_query", {"query": "SELECT 1; DROP TABLE sales"})
    assert out.startswith("SQLError")


def test_verification_round_trips_through_jsonl(tmp_path):
    # Verified runs carry the per-trajectory verdict into the JSONL and back (#21).
    from agentsynth import load_jsonl, to_jsonl

    result = run_recipe(Recipe(num_trajectories=4, vary_modes=True, verify=True))
    assert all(t.verification is not None for t in result.trajectories)

    path = str(tmp_path / "verified.jsonl")
    to_jsonl(result.trajectories, path)
    loaded = load_jsonl(path)
    assert [t.verification for t in loaded] == [t.verification for t in result.trajectories]


def test_jsonl_without_verification_still_loads(tmp_path):
    import json

    from agentsynth import load_jsonl, to_jsonl

    result = run_recipe(Recipe(num_trajectories=2, evaluate=False))
    path = str(tmp_path / "plain.jsonl")
    to_jsonl(result.trajectories, path)
    # Strip the field to simulate a pre-#21 file.
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            rec.pop("verification", None)
            fh.write(json.dumps(rec) + "\n")
    loaded = load_jsonl(path)
    assert all(t.verification is None for t in loaded)
