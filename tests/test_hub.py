import os

from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator
from agentsynth.hub import dataset_card, prepare_dataset_dir


def test_dataset_card_has_frontmatter_and_fields():
    card = dataset_card(count=12000, pretty_name="Test DS", generator="mock", pass_rate=0.95)
    assert card.startswith("---")
    assert "license: mit" in card
    assert "10K<n<100K" in card  # size bucket for 12k rows
    assert "95.0%" in card
    assert "Test DS" in card


def test_prepare_dataset_dir_writes_all_files(tmp_path):
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch("weather in Tokyo", num_trajectories=5, vary_modes=True)
    results = TrajectoryEvaluator(use_mock=True).evaluate_batch(trajs)
    out = prepare_dataset_dir(
        trajs, str(tmp_path / "ds"), eval_results=results, repo_id="me/test-ds"
    )
    assert os.path.exists(os.path.join(out, "train.jsonl"))
    assert os.path.exists(os.path.join(out, "README.md"))
    assert os.path.exists(os.path.join(out, "sharegpt.json"))
    card = open(os.path.join(out, "README.md")).read()
    assert "me/test-ds" in card
