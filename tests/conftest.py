import os

# Force offline mock for the whole suite before any test imports agentsynth,
# so a provider key sitting in the environment can't make tests hit the network.
os.environ.setdefault("AGENTSYNTH_FORCE_MOCK", "1")
