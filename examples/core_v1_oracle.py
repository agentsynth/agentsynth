"""Reference solution for the core_v1 pack: the best score the tasks allow.

The expert lives in `agentsynth.demo` so the playground can run it too; this
module stays the documented oracle path for validate, teach, and CI:

    agentsynth bench --pack packs/core_v1.yaml --policy examples.core_v1_oracle:solve
    agentsynth pack validate packs/core_v1.yaml --oracle examples/core_v1_oracle.py:solve

It works the way a careful operator would — look at the rows first, make the
change, read it back, then answer from what it saw.
"""

from agentsynth.demo import expert as solve

__all__ = ["solve"]
