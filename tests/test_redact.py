"""Redaction for donated traces: secrets out, plain numbers untouched."""

import json

from agentsynth import redact_text, redact_trajectory, trajectory_from_messages
from agentsynth.cli import main as cli_main


def test_secrets_get_redacted():
    cases = {
        "mail an.tran@example.com today": "[redacted-email]",
        "key sk-abc123def456ghi789jkl0": "[redacted-api_key]",
        "use ghp_abcdefghij0123456789klmnop": "[redacted-api_key]",
        "aws AKIAIOSFODNN7EXAMPLE": "[redacted-api_key]",
        "Authorization: Bearer abc.def-ghi_jkl+mno=": "[redacted-token]",
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.dQw4w9WgXcQ_signature": "[redacted-token]",
        "hash 0123456789abcdef0123456789abcdef": "[redacted-hex_id]",
        "call +84 912 345 678 now": "[redacted-phone]",
        "or 0123-456-789": "[redacted-phone]",
    }
    for text, marker in cases.items():
        out = redact_text(text)
        assert marker in out, f"{text!r} -> {out!r}"


def test_plain_numbers_and_ids_survive():
    for text in (
        "refund order 7 in the orders database",
        "the mean of 12, 19, 7, 22, 31",
        "ticket 4 is closed",
        "set stock to 35",
        "trajectory a502b2a53e75",  # 12-hex id stays
    ):
        assert redact_text(text) == text


def test_redact_trajectory_covers_every_surface():
    traj = trajectory_from_messages(
        [
            {"role": "user", "content": "email bob@corp.io about order 7"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "send_email",
                            "arguments": json.dumps(
                                {"to": "bob@corp.io", "key": "sk-secretsecretsecret123"}
                            ),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "sent to bob@corp.io"},
            {"role": "assistant", "content": "Done, emailed bob@corp.io."},
        ]
    )
    out = redact_trajectory(traj)
    blob = out.model_dump_json()
    assert "bob@corp.io" not in blob
    assert "sk-secretsecret" not in blob
    assert "order 7" in out.query  # the task itself survives


def test_cli_import_redact_flag(tmp_path):
    trace = {
        "messages": [
            {"role": "user", "content": "ping carol@x.dev re order 7"},
            {"role": "assistant", "content": "Emailed carol@x.dev."},
        ]
    }
    src = tmp_path / "traces.jsonl"
    src.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    out = tmp_path / "clean.jsonl"

    assert cli_main(["import", "--in", str(src), "--out", str(out), "--redact"]) == 0
    text = out.read_text()
    assert "carol@x.dev" not in text
    assert "[redacted-email]" in text
    assert "order 7" in text
