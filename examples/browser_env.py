"""Generate a trajectory whose tool calls drive a real headless browser.

    pip install "agentsynth-ai[browser]"
    playwright install chromium
    python examples/browser_env.py

The generator stays in offline-mock mode for its reasoning, but every browser tool
call runs against a live Chromium page, so the observations are real page content.
"""

from agentsynth import AgentTrajectoryGenerator
from agentsynth.environments import BrowserEnvironment


def main() -> None:
    env = BrowserEnvironment(start_url="https://example.com")
    try:
        gen = AgentTrajectoryGenerator(environment=env)
        traj = gen.generate("open the page and read what it says")

        print("tools used:", traj.tool_names_used())
        for step in traj.steps:
            if step.step_type == "tool_call":
                print(f"\n-> {step.tool_name}({step.tool_args})")
            elif step.step_type == "observation":
                print(step.observation)
        print("\nfinal answer:", traj.final_answer)
    finally:
        env.close()


if __name__ == "__main__":
    main()
