"""Publish a dataset to the Hugging Face Hub.

`dataset_card` builds the README/card, `prepare_dataset_dir` writes a ready-to-push
folder (train.jsonl + card + ShareGPT copy) with no network, and `push_dataset`
uploads it (lazy `huggingface_hub` import).
"""

from __future__ import annotations

import os
from typing import List, Optional

from .schemas import EvalResult, Trajectory

_CARD = """---
license: mit
language:
- en
task_categories:
- text-generation
tags:
- synthetic
- agent
- agentic
- tool-use
- trajectories
- llm-as-judge
size_categories:
- {size}
pretty_name: {pretty_name}
---

# {pretty_name}

{summary}

- **Trajectories:** {count}
- **Generator:** {generator}
- **Pass@1:** {pass_rate}

Generated and verified with [AgentSynth](https://github.com/agentsynth/agentsynth).

## Fields (JSONL)

`id`, `query`, `mode` (single_agent / multi_agent / code_execution), `domain`,
`messages` (OpenAI-style), `tools`, `steps`, `final_answer`, `success`,
`generator_model`, `metadata`. A ShareGPT copy is included as `sharegpt.json`.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")
```

## Limitations

This is synthetic data and reflects the generator and judge used to make it.
Validate on your own benchmark before relying on it.
"""


def _size_bucket(n: int) -> str:
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    return "100K<n<1M"


def dataset_card(
    count: int,
    pretty_name: str = "AgentSynth Trajectories",
    generator: str = "mock",
    pass_rate: Optional[float] = None,
    summary: Optional[str] = None,
    repo_id: str = "agentsynth/agentsynth-trajectories",
) -> str:
    return _CARD.format(
        size=_size_bucket(count),
        pretty_name=pretty_name,
        count=count,
        generator=generator,
        pass_rate=f"{pass_rate * 100:.1f}%" if isinstance(pass_rate, (int, float)) else "n/a",
        summary=summary
        or "Synthetic, verified multi-step agent trajectories for fine-tuning agentic LLMs.",
        repo_id=repo_id,
    )


def prepare_dataset_dir(
    trajectories: List[Trajectory],
    out_dir: str,
    eval_results: Optional[List[EvalResult]] = None,
    pretty_name: str = "AgentSynth Trajectories",
    repo_id: str = "agentsynth/agentsynth-trajectories",
) -> str:
    """Write a push-ready dataset folder (no network). Returns the folder path."""
    from .exporters import to_jsonl, to_sharegpt

    os.makedirs(out_dir, exist_ok=True)
    to_jsonl(trajectories, os.path.join(out_dir, "train.jsonl"))
    to_sharegpt(trajectories, os.path.join(out_dir, "sharegpt.json"))

    pass_rate = None
    if eval_results:
        pass_rate = sum(1 for e in eval_results if e.passed) / len(eval_results)
    generator = trajectories[0].generator_model if trajectories else "mock"

    card = dataset_card(
        count=len(trajectories),
        pretty_name=pretty_name,
        generator=generator,
        pass_rate=pass_rate,
        repo_id=repo_id,
    )
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    return out_dir


def push_dataset(
    trajectories: List[Trajectory],
    repo_id: str,
    token: Optional[str] = None,
    eval_results: Optional[List[EvalResult]] = None,
    private: bool = False,
    pretty_name: str = "AgentSynth Trajectories",
    out_dir: Optional[str] = None,
) -> str:
    """Build the dataset folder and upload it to the Hub. Returns the repo URL.

    Requires `pip install "agentsynth-ai[hub]"` and a write token (arg or HF login).
    """
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise ImportError(
            'push_dataset needs huggingface_hub: pip install "agentsynth-ai[hub]"'
        ) from exc

    import tempfile

    folder = out_dir or tempfile.mkdtemp(prefix="agentsynth_ds_")
    prepare_dataset_dir(
        trajectories, folder, eval_results=eval_results, pretty_name=pretty_name, repo_id=repo_id
    )

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"


__all__ = ["dataset_card", "prepare_dataset_dir", "push_dataset"]
