"""Score a model on τ-bench (multi-turn, agentic) — when it's installed.

    pip install git+https://github.com/sierra-research/tau-bench
    export OPENAI_API_KEY=...          # for the agent + user-simulator models
    python examples/tau_bench_demo.py

Unlike BFCL, τ-bench actually drives an LLM through a multi-turn retail/airline task
against a user simulator, so it needs the package and API keys. Without them this
prints the setup steps and exits cleanly.
"""

from agentsynth.benchmarks import run_tau_bench, tau_bench_available


def main() -> None:
    if not tau_bench_available():
        print("tau-bench is not installed. To run it:")
        print("  pip install git+https://github.com/sierra-research/tau-bench")
        print("  export OPENAI_API_KEY=...")
        print("  python examples/tau_bench_demo.py")
        return

    # Point `model` at your fine-tuned model behind an OpenAI-compatible endpoint.
    result = run_tau_bench(model="gpt-4o-mini", env_name="retail", task_ids=[0, 1, 2])
    print(f"τ-bench {result['env']}: pass_rate={result['pass_rate']} over {result['n']} tasks")


if __name__ == "__main__":
    main()
