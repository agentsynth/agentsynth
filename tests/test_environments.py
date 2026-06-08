from agentsynth import AgentTrajectoryGenerator
from agentsynth.environments import (
    CompositeEnvironment,
    PythonSandbox,
    SQLEnvironment,
)


def test_sql_runs_real_query():
    env = SQLEnvironment()
    out = env.execute("sql_query", {"query": "SELECT SUM(revenue) AS total FROM sales"})
    assert "total" in out
    assert any(ch.isdigit() for ch in out)


def test_sql_rejects_writes():
    env = SQLEnvironment()
    for bad in ("DROP TABLE sales", "DELETE FROM sales", "UPDATE sales SET revenue=0"):
        assert env.execute("sql_query", {"query": bad}).startswith("SQLError")


def test_sql_sample_args_is_runnable():
    env = SQLEnvironment()
    args = env.sample_args("sql_query", "revenue by region", 1)
    out = env.execute("sql_query", args)
    assert "SQLError" not in out and "row" in out


def test_python_returns_real_stdout():
    env = PythonSandbox()
    assert env.execute("python", {"code": "print(2 + 2)"}) == "4"


def test_python_captures_errors():
    out = PythonSandbox().execute("python", {"code": "print(1 / 0)"})
    assert "ZeroDivisionError" in out


def test_composite_routes_by_tool():
    env = CompositeEnvironment([SQLEnvironment(), PythonSandbox()])
    assert set(env.tool_names()) == {"sql_query", "python"}
    assert env.execute("python", {"code": "print('hi')"}) == "hi"


def test_generator_grounds_observations_in_environment():
    gen = AgentTrajectoryGenerator(use_mock=True, environment=SQLEnvironment())
    traj = gen.generate("Total revenue by region?", mode="single_agent")
    # The catalog comes from the environment, and the tool actually ran.
    assert traj.tool_names_used()
    assert all(name == "sql_query" for name in traj.tool_names_used())
    observations = [s.observation for s in traj.steps if s.step_type == "observation"]
    assert any("row" in (o or "") for o in observations)
