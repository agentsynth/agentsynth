"""pack new --from-schema: a starter pack generated from a DB schema validates."""

import pytest

from agentsynth.cli import main as cli_main


def _gen(tmp_path, sql, pack_id="gen_v1"):
    schema = tmp_path / "schema.sql"
    schema.write_text(sql, encoding="utf-8")
    code = cli_main(["pack", "new", pack_id, "--from-schema", str(schema), "--dir", str(tmp_path)])
    return code, tmp_path / f"{pack_id}.yaml", tmp_path / f"{pack_id}_oracle.py"


def test_generated_pack_validates(tmp_path, capsys):
    code, pack, oracle = _gen(
        tmp_path, "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, status TEXT);"
    )
    assert code == 0 and pack.exists() and oracle.exists()
    assert "self-check: oracle 3/3, do-nothing 0/3 — PACK OK" in capsys.readouterr().out

    # the real gate, run independently
    assert cli_main(["pack", "validate", str(pack)]) == 0
    assert "PACK OK" in capsys.readouterr().out


def test_prefers_a_state_like_column(tmp_path):
    _, pack, _ = _gen(tmp_path, "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, status TEXT);")
    body = pack.read_text()
    assert "Set status of t row 1" in body  # status wins over name via the hint


def test_skips_pk_and_unique_text_columns(tmp_path):
    # the only safe target is `status`: sku is the PK, code is UNIQUE
    code, pack, _ = _gen(
        tmp_path,
        "CREATE TABLE `inventory` (`sku` TEXT PRIMARY KEY, code TEXT UNIQUE, "
        "status TEXT, qty INTEGER);",
    )
    assert code == 0
    assert "Set status of inventory" in pack.read_text()
    assert cli_main(["pack", "validate", str(pack)]) == 0


def test_errors_without_an_integer_key(tmp_path):
    with pytest.raises(SystemExit) as err:
        _gen(tmp_path, "CREATE TABLE t (sku TEXT PRIMARY KEY, status TEXT);")
    assert "integer key" in str(err.value)


def test_errors_without_a_text_column(tmp_path):
    with pytest.raises(SystemExit) as err:
        _gen(tmp_path, "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b REAL);")
    assert "text column" in str(err.value)


def test_errors_on_no_create_table(tmp_path):
    with pytest.raises(SystemExit) as err:
        _gen(tmp_path, "-- just a comment, no table here")
    assert "CREATE TABLE" in str(err.value)
