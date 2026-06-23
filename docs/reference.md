# API reference

The public surface, re-exported from the top-level `agentsynth` package.

## Generation

::: agentsynth.AgentTrajectoryGenerator

## Evaluation & verification

::: agentsynth.TrajectoryEvaluator

::: agentsynth.verify_trajectory

::: agentsynth.batch_verify

::: agentsynth.EnsembleEvaluator

::: agentsynth.Verifier

::: agentsynth.VerificationResult

::: agentsynth.ExecutionVerifier

::: agentsynth.ToolArgVerifier

::: agentsynth.SafetyVerifier

::: agentsynth.ExpectedAnswerVerifier

::: agentsynth.get_rubric

::: agentsynth.rubric_names

::: agentsynth.RUBRIC_PRESETS

## Learned verifier

::: agentsynth.train_learned_verifier

::: agentsynth.LearnedVerifier

## Scenarios

::: agentsynth.Scenario

::: agentsynth.SqlCheck

::: agentsynth.HttpCheck

::: agentsynth.CalledTool

::: agentsynth.AnswerContains

::: agentsynth.CodeCheck

::: agentsynth.run_scenario_suite

::: agentsynth.load_scenarios

::: agentsynth.save_scenarios

## Robustness (reward-hacking audit)

::: agentsynth.audit_pack

::: agentsynth.RobustnessReport

::: agentsynth.perturb_scenario

::: agentsynth.ipt_report

## Synthesize verifiers from a demonstration

::: agentsynth.scenario_from_demonstration

::: agentsynth.pack_from_demonstrations

## Export a pack (OpenEnv / verifiers)

::: agentsynth.scenario_reward

::: agentsynth.reward_from_messages

::: agentsynth.export_pack

## Reliability (beyond pass@1)

::: agentsynth.reliability_report

::: agentsynth.ReliabilityReport

::: agentsynth.wilson_interval

## Contamination audit

::: agentsynth.contamination_report

::: agentsynth.ContaminationReport

::: agentsynth.canary_for

::: agentsynth.held_out_pack

## Reproducible submissions

::: agentsynth.run_manifest

::: agentsynth.verify_run

::: agentsynth.pack_fingerprint

## Multi-turn (user simulator)

::: agentsynth.run_conversation

::: agentsynth.run_conversation_suite

::: agentsynth.ConversationResult

## RL

::: agentsynth.AgentGym

::: agentsynth.make_reward_fn

::: agentsynth.rl.to_openenv

## Bring your own loop (adapters)

::: agentsynth.to_openai_tools

::: agentsynth.action_from_openai_tool_call

## Environments

::: agentsynth.Environment

::: agentsynth.SQLEnvironment

::: agentsynth.PythonSandbox

::: agentsynth.MCPEnvironment

::: agentsynth.BrowserEnvironment

::: agentsynth.RestEnvironment

::: agentsynth.CompositeEnvironment

## Pipelines

::: agentsynth.Recipe

::: agentsynth.run_recipe

::: agentsynth.load_recipe

::: agentsynth.make_environment

## Benchmark

::: agentsynth.run_benchmark

::: agentsynth.compare_models

::: agentsynth.BenchmarkCase

::: agentsynth.BenchmarkReport

::: agentsynth.BUILTIN_CASES

::: agentsynth.agentsynth_model

::: agentsynth.prompted_model

::: agentsynth.report_table_md

## Trace import & redaction

::: agentsynth.trajectory_from_messages

::: agentsynth.import_traces

::: agentsynth.load_traces_jsonl

::: agentsynth.importers.trajectory_from_otel_spans

::: agentsynth.redact_text

::: agentsynth.redact_trajectory

## Flywheel

::: agentsynth.mine_failures

::: agentsynth.mine_judge_failures

::: agentsynth.recipe_from_failures

::: agentsynth.evolve_queries

## Scale

::: agentsynth.CachingLLMClient

::: agentsynth.CostMeter

::: agentsynth.BudgetExceeded

::: agentsynth.run_resumable

## Preference data & dedup

::: agentsynth.build_preference_pairs

::: agentsynth.PreferencePair

::: agentsynth.to_dpo_jsonl

::: agentsynth.load_dpo_jsonl

::: agentsynth.dedup_trajectories

::: agentsynth.decontaminate

## Training data prep

::: agentsynth.build_sft_dataset

::: agentsynth.build_dpo_dataset

::: agentsynth.to_sft_records

::: agentsynth.to_dpo_records

## Tasks

::: agentsynth.SeedTask

::: agentsynth.sample_tasks

## Metrics

::: agentsynth.compute_dataset_metrics

::: agentsynth.diversity_score

::: agentsynth.run_report_md

## Exporters

::: agentsynth.to_jsonl

::: agentsynth.to_sharegpt

::: agentsynth.to_adp

::: agentsynth.save_dataset

::: agentsynth.load_jsonl

## Hugging Face Hub

::: agentsynth.push_dataset

::: agentsynth.dataset_card

::: agentsynth.prepare_dataset_dir

## Utilities

::: agentsynth.parse_tool_catalog

::: agentsynth.default_tool_catalog

::: agentsynth.PythonREPL

::: agentsynth.LLMClient

## Schemas

::: agentsynth.Trajectory

::: agentsynth.TrajectoryStep

::: agentsynth.ToolSpec

::: agentsynth.RubricScores

::: agentsynth.EvalResult
