"""The pack scaffold and the gates a pack must pass to ship."""

import pytest

from agentsynth.cli import main as cli_main


def _scaffold(tmp_path, pack_id="demo_v1"):
    assert cli_main(["pack", "new", pack_id, "--dir", str(tmp_path)]) == 0
    return tmp_path / f"{pack_id}.yaml", tmp_path / f"{pack_id}_oracle.py"


def test_scaffold_validates_clean(tmp_path, capsys):
    pack, oracle = _scaffold(tmp_path)
    assert pack.exists() and oracle.exists()

    code = cli_main(["pack", "validate", str(pack)])
    out = capsys.readouterr().out
    assert code == 0
    assert "PACK OK" in out
    assert "oracle passes 3/3" in out
    assert "do-nothing policy passes 0/3" in out


def test_scaffold_refuses_overwrite(tmp_path):
    _scaffold(tmp_path)
    with pytest.raises(SystemExit):
        cli_main(["pack", "new", "demo_v1", "--dir", str(tmp_path)])


def test_core_pack_validates_with_module_oracle(capsys):
    code = cli_main(
        ["pack", "validate", "packs/core_v1.yaml", "--oracle", "examples.core_v1_oracle:solve"]
    )
    assert code == 0
    assert "oracle passes 10/10" in capsys.readouterr().out


def test_broken_oracle_fails_the_gate(tmp_path, capsys):
    pack, oracle = _scaffold(tmp_path)
    oracle.write_text(
        "def solve(observation, gym):\n    return {'answer': 'no idea'}\n", encoding="utf-8"
    )
    code = cli_main(["pack", "validate", str(pack)])
    out = capsys.readouterr().out
    assert code == 1
    assert "every scenario must be solvable" in out


def test_trivial_pack_trips_the_lazy_guard(tmp_path, capsys):
    pack = tmp_path / "trivial.yaml"
    pack.write_text(
        "\n".join(
            f"""- id: t{i}
  task: Say done.
  environment:
    type: sql
    schema: CREATE TABLE x (id INTEGER PRIMARY KEY)
    table: x
    rows: [[1]]
  checkers:
    - kind: answer
      any_of: ["done"]"""
            for i in range(3)
        ),
        encoding="utf-8",
    )
    (tmp_path / "trivial_oracle.py").write_text(
        "def solve(observation, gym):\n    return {'answer': 'done'}\n", encoding="utf-8"
    )
    code = cli_main(["pack", "validate", str(pack)])
    out = capsys.readouterr().out
    assert code == 1
    assert "lazy guard" in out


def test_schema_gates(tmp_path, capsys):
    small = tmp_path / "small.yaml"
    small.write_text(
        """- id: only-one
  task: Do a thing.
  environment:
    type: sql
    schema: CREATE TABLE x (id INTEGER PRIMARY KEY)
    table: x
    rows: [[1]]
  checkers:
    - kind: called_tool
      name: sql_query
""",
        encoding="utf-8",
    )
    assert cli_main(["pack", "validate", str(small), "--oracle", "x.py:solve"]) == 1
    assert "at least 3 scenarios" in capsys.readouterr().out


def test_missing_oracle_is_a_clear_error(tmp_path):
    pack, oracle = _scaffold(tmp_path)
    oracle.unlink()
    with pytest.raises(SystemExit) as err:
        cli_main(["pack", "validate", str(pack)])
    assert "no oracle" in str(err.value)
