"""Generate a trajectory whose tool calls hit a real REST API, from its OpenAPI spec.

    python examples/rest_env.py

Spins up a tiny JSON API on loopback, points RestEnvironment at its OpenAPI spec,
and lets the generator drive it — every observation is a real HTTP response. Swap
the spec/base_url for a real service (a staging server, or any public OpenAPI URL)
and the same code applies. Runs fully offline.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentsynth import AgentTrajectoryGenerator
from agentsynth.environments import RestEnvironment

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Order API", "version": "1"},
    "paths": {
        "/orders/{order_id}": {
            "get": {
                "operationId": "get_order",
                "summary": "Fetch one order with its status and total.",
                "parameters": [
                    {
                        "name": "order_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
            }
        },
        "/orders": {
            "get": {
                "operationId": "list_orders",
                "summary": "List recent orders, newest first.",
                "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
            }
        },
    },
}

_ORDERS = {n: {"id": n, "status": "shipped", "total": round(19.99 * n, 2)} for n in range(1, 6)}


class _Api(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/orders":
            payload = list(_ORDERS.values())
        elif path.startswith("/orders/"):
            payload = _ORDERS.get(int(path.rsplit("/", 1)[-1]), {"error": "no such order"})
        else:
            payload = {"error": "not found"}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address

    env = RestEnvironment(SPEC, base_url=f"http://{host}:{port}")
    print("tools from the spec:", env.tool_names())

    gen = AgentTrajectoryGenerator(environment=env)
    traj = gen.generate("check the status of order 3")
    for step in traj.steps:
        if step.step_type == "tool_call":
            print(f"\n-> {step.tool_name}({step.tool_args})")
        elif step.step_type == "observation":
            print(step.observation)
    print("\nfinal answer:", traj.final_answer)
    server.shutdown()


if __name__ == "__main__":
    main()
