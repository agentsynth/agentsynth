"""REST/OpenAPI environment tests.

Everything runs against a tiny loopback HTTP API (stdlib only), so the suite stays
hermetic — no network, no extra dependencies, works on the 3.9 interpreter too.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from agentsynth import AgentTrajectoryGenerator
from agentsynth.environments import RestEnvironment

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Tiny API", "version": "1"},
    "servers": [{"url": "http://spec-default.example"}],
    "components": {
        "schemas": {
            "EchoIn": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "What to echo"}},
                "required": ["text"],
            }
        }
    },
    "paths": {
        "/users/{user_id}": {
            "get": {
                "operationId": "get_user",
                "summary": "Fetch one user by id.",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    },
                    {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
                ],
            }
        },
        "/echo": {
            "post": {
                "operationId": "echo",
                "summary": "Echo a message back.",
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/EchoIn"}}
                    }
                },
            }
        },
        "/boom": {"get": {"operationId": "boom", "summary": "Always fails."}},
        "/paged": {"get": {"summary": "A long listing."}},
    },
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith("/users/"):
            user_id = url.path.rsplit("/", 1)[-1]
            verbose = parse_qs(url.query).get("verbose", ["false"])[0]
            self._send(200, {"id": int(user_id), "name": f"user{user_id}", "verbose": verbose})
        elif url.path == "/boom":
            self._send(500, {"error": "kaboom"})
        elif url.path == "/paged":
            self._send(200, {"items": list(range(200))})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path == "/echo":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, {"echo": payload.get("text")})
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture(scope="module")
def api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    yield f"http://{host}:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def env(api):
    return RestEnvironment(SPEC, base_url=api)


def test_operations_become_tools(env):
    names = set(env.tool_names())
    assert {"get_user", "echo", "boom", "get_paged"} <= names  # fallback name for /paged
    by_name = {t.name: t for t in env.tools()}
    assert by_name["get_user"].description == "Fetch one user by id."
    assert set(by_name["get_user"].required_args()) == {"user_id"}
    assert by_name["echo"].required_args() == ["text"]  # flattened from the $ref body
    assert "What to echo" in str(by_name["echo"].parameters)


def test_path_and_query_params_route_to_the_wire(env):
    out = env.execute("get_user", {"user_id": 7, "verbose": True})
    assert '"id": 7' in out
    assert '"user7"' in out
    assert '"True"' in out or '"true"' in out  # query string made it through


def test_json_body_routes_to_the_wire(env):
    assert '"echo": "hello"' in env.execute("echo", {"text": "hello"})


def test_http_errors_become_clean_observations(env):
    out = env.execute("boom", {})
    assert out.startswith("RestError: HTTP 500")
    assert "kaboom" in out


def test_unknown_tool_raises(env):
    with pytest.raises(KeyError):
        env.execute("nope", {})


def test_missing_path_param_is_reported(env):
    assert "user_id" in env.execute("get_user", {})


def test_responses_are_truncated(api):
    small = RestEnvironment(SPEC, base_url=api, max_chars=80)
    out = small.execute("get_paged", {})
    assert len(out) <= 80 and out.endswith("…")


def test_sample_args_then_execute(env):
    args = env.sample_args("get_user", "look up account 12", seed=3)
    assert "user_id" in args
    assert "RestError" not in env.execute("get_user", args)


def test_methods_filter_drops_writes(api):
    readonly = RestEnvironment(SPEC, base_url=api, methods=("get",))
    assert "echo" not in readonly.tool_names()
    assert "get_user" in readonly.tool_names()


def test_base_url_falls_back_to_the_spec_servers():
    env = RestEnvironment(SPEC)  # no base_url argument
    assert env.base_url == "http://spec-default.example"


def test_spec_from_json_string_and_file(api, tmp_path):
    from_string = RestEnvironment(json.dumps(SPEC), base_url=api)
    assert "get_user" in from_string.tool_names()
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(SPEC), encoding="utf-8")
    from_file = RestEnvironment(str(path), base_url=api)
    assert "echo" in from_file.tool_names()


def test_generator_drives_rest_tools(env):
    gen = AgentTrajectoryGenerator(use_mock=True, environment=env)
    traj = gen.generate("fetch user 5 and echo a greeting", mode="single_agent")
    used = traj.tool_names_used()
    assert used and all(name in set(env.tool_names()) for name in used)
