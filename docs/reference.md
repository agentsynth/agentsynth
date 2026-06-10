# API reference

The public surface, re-exported from the top-level `agentsynth` package.

## Generation

::: agentsynth.AgentTrajectoryGenerator

## Evaluation & verification

::: agentsynth.TrajectoryEvaluator

::: agentsynth.verify_trajectory

::: agentsynth.EnsembleEvaluator

## Pipelines

::: agentsynth.Recipe

::: agentsynth.run_recipe

## Environments

::: agentsynth.SQLEnvironment

::: agentsynth.PythonSandbox

::: agentsynth.MCPEnvironment

::: agentsynth.BrowserEnvironment

::: agentsynth.RestEnvironment

## Learned verifier

::: agentsynth.train_learned_verifier

::: agentsynth.LearnedVerifier

## Flywheel

::: agentsynth.mine_failures

::: agentsynth.mine_judge_failures

::: agentsynth.recipe_from_failures

## RL

::: agentsynth.AgentGym

::: agentsynth.make_reward_fn

::: agentsynth.rl.to_openenv

## Benchmark

::: agentsynth.run_benchmark

::: agentsynth.compare_models

## Preference data & dedup

::: agentsynth.build_preference_pairs

::: agentsynth.dedup_trajectories

## Schemas

::: agentsynth.Trajectory

::: agentsynth.ToolSpec

::: agentsynth.EvalResult
