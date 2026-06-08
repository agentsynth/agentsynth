from agentsynth import load_jsonl
from agentsynth.pipelines import Recipe, load_recipe, run_recipe


def test_run_recipe_single_query():
    recipe = Recipe(
        query="weather in Tokyo and a 15% tip on $80",
        num_trajectories=6,
        vary_modes=True,
    )
    result = run_recipe(recipe)
    assert len(result.trajectories) == 6
    assert result.metrics["num_trajectories"] == 6
    assert 0.0 <= result.metrics["pass_rate"] <= 1.0
    assert len(result.eval_results) == 6


def test_run_recipe_samples_from_taxonomy():
    recipe = Recipe(num_trajectories=8, domains=["coding", "data_analysis"], evaluate=False)
    result = run_recipe(recipe)
    assert len(result.trajectories) == 8


def test_run_recipe_with_sql_environment_is_grounded():
    recipe = Recipe(
        query="revenue by region",
        num_trajectories=3,
        environment="sql",
        evaluate=False,
    )
    result = run_recipe(recipe)
    observations = [
        s.observation for t in result.trajectories for s in t.steps if s.step_type == "observation"
    ]
    assert any("row" in (o or "") for o in observations)


def test_concurrent_matches_sequential():
    kwargs = dict(
        query="mean of 4 8 15 16 23 42",
        num_trajectories=8,
        vary_modes=True,
        evaluate=False,
    )
    seq = run_recipe(Recipe(max_workers=1, **kwargs)).trajectories
    conc = run_recipe(Recipe(max_workers=4, **kwargs)).trajectories
    assert [t.tool_signature() for t in seq] == [t.tool_signature() for t in conc]
    assert [t.final_answer for t in seq] == [t.final_answer for t in conc]


def test_load_recipe_from_yaml(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("name: t\nquery: hello world\nnum_trajectories: 3\nevaluate: false\n")
    recipe = load_recipe(str(path))
    assert recipe.num_trajectories == 3 and recipe.query == "hello world"
    assert len(run_recipe(recipe).trajectories) == 3


def test_recipe_exports_dataset(tmp_path):
    out = tmp_path / "ds.jsonl"
    recipe = Recipe(
        query="x",
        num_trajectories=4,
        export_format="jsonl",
        export_path=str(out),
    )
    result = run_recipe(recipe)
    assert result.output_path == str(out)
    assert len(load_jsonl(str(out))) == 4
