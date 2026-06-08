import json

from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator, build_preference_pairs
from agentsynth.training import build_sft_dataset, to_dpo_records, to_sft_records

GEN = AgentTrajectoryGenerator(use_mock=True)


def test_sft_records_are_conversational():
    trajs = GEN.generate_batch("weather in Tokyo", num_trajectories=4, vary_modes=True)
    records = to_sft_records(trajs)
    assert len(records) == 4
    for record in records:
        assert record["messages"][0]["role"] == "user"
        assert "id" in record


def test_sft_only_passed_filters_everything_at_impossible_threshold():
    trajs = GEN.generate_batch("weather in Tokyo", num_trajectories=5)
    results = TrajectoryEvaluator(use_mock=True, pass_threshold=2.0).evaluate_batch(trajs)
    assert to_sft_records(trajs, only_passed=True, eval_results=results) == []


def test_dpo_records_shape():
    pairs = build_preference_pairs(GEN, TrajectoryEvaluator(use_mock=True), "analyze sales", k=6)
    records = to_dpo_records(pairs)
    assert len(records) == len(pairs)
    assert all(set(r) == {"prompt", "chosen", "rejected"} for r in records)


def test_build_sft_dataset_writes_jsonl(tmp_path):
    trajs = GEN.generate_batch("x", num_trajectories=3)
    path = tmp_path / "sft.jsonl"
    build_sft_dataset(trajs, str(path))
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
