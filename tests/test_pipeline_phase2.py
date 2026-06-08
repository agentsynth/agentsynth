from agentsynth.pipelines import Recipe, run_recipe


def test_recipe_verify_populates_results_and_metric():
    result = run_recipe(
        Recipe(
            query="compute the mean of 4 8 15 16 23 42",
            num_trajectories=6,
            vary_modes=True,
            verify=True,
        )
    )
    assert len(result.verifications) == 6
    assert "verified_rate" in result.metrics
    assert 0.0 <= result.metrics["verified_rate"] <= 1.0


def test_recipe_dedup_preserves_distinct_trajectories():
    # Distinct seed-task queries are not duplicates; dedup must keep them all.
    result = run_recipe(
        Recipe(
            domains=["coding", "data_analysis", "weather_math"],
            num_trajectories=6,
            dedup=True,
            evaluate=False,
        )
    )
    assert len(result.trajectories) == 6
    assert result.duplicates_removed == 0


def test_recipe_rubric_preset_changes_pass_rate():
    common = dict(query="weather in Tokyo", num_trajectories=8, vary_modes=True)
    strict = run_recipe(Recipe(rubric="strict", **common))
    lenient = run_recipe(Recipe(rubric="lenient", **common))
    assert lenient.metrics["pass_rate"] >= strict.metrics["pass_rate"]


def test_recipe_verify_and_dedup_together():
    result = run_recipe(
        Recipe(
            domains=["coding", "data_analysis"],
            num_trajectories=8,
            verify=True,
            dedup=True,
            rubric="balanced",
        )
    )
    assert len(result.verifications) == len(result.trajectories)
    assert "verified_rate" in result.metrics
