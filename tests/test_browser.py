"""Browser environment tests.

The text/keyword helpers and the tool surface are tested with no browser, so they
run everywhere (including the 3.9 dev interpreter where Playwright isn't installed).
The live tests serve a tiny two-page site over loopback HTTP and drive a real headless
Chromium; they're skipped unless `playwright` (and its browser) are available.
"""

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentsynth.environments.browser import (
    BrowserEnvironment,
    _clean_text,
    _find_snippet,
    _pick_keyword,
)

BROWSER_TOOLS = {
    "browser_navigate",
    "browser_read",
    "browser_links",
    "browser_find",
    "browser_click",
}

INDEX_HTML = """<!doctype html><html><head><title>AgentSynth Test Home</title></head>
<body>
  <h1>Welcome to the AgentSynth test site</h1>
  <p>This page mentions trajectories and verification for the search test.</p>
  <a href="page2.html">Go to the pricing page</a>
</body></html>"""

PAGE2_HTML = """<!doctype html><html><head><title>Pricing</title></head>
<body>
  <h1>Pricing details</h1>
  <p>The enterprise plan costs 4242 dollars per month.</p>
  <a href="index.html">Back home</a>
</body></html>"""


# --- pure helpers (no browser needed) --------------------------------------


def test_clean_text_strips_blank_lines_and_caps():
    assert _clean_text("  a  \n\n  b \n") == "a\nb"
    assert _clean_text("x" * 50, limit=10).endswith("…")
    assert len(_clean_text("x" * 50, limit=10)) == 10


def test_pick_keyword_is_deterministic_and_skips_stopwords():
    kw = _pick_keyword("what is the weather in Paris", seed=7)
    assert kw.lower() not in {"the", "what"}
    assert kw == _pick_keyword("what is the weather in Paris", seed=7)
    assert _pick_keyword("???", seed=1) == "example"


def test_find_snippet_hit_and_miss():
    text = "alpha beta gamma delta epsilon"
    assert "gamma" in (_find_snippet(text, "gamma") or "")
    assert _find_snippet(text, "omega") is None


def test_tool_surface_without_launching_a_browser():
    env = BrowserEnvironment(start_url="https://example.com")
    assert set(env.tool_names()) == BROWSER_TOOLS
    assert env.sample_args("browser_navigate", "q", 1) == {"url": "https://example.com"}
    assert env.sample_args("browser_read", "q", 1) == {}
    text_arg = env.sample_args("browser_find", "compare the pricing tiers", 3)
    assert isinstance(text_arg["text"], str) and text_arg["text"]


# --- live browser (skipped unless playwright + chromium are installed) ------


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("site")
    (root / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (root / "page2.html").write_text(PAGE2_HTML, encoding="utf-8")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    yield f"http://{host}:{port}/"
    server.shutdown()


@pytest.fixture(scope="module")
def browser_env(site):
    pytest.importorskip("playwright")
    env = BrowserEnvironment(start_url=site, headless=True)
    try:
        env._page_obj()  # probe: is a browser actually installed?
    except Exception as exc:  # pragma: no cover - depends on the host
        env.close()
        pytest.skip(f"playwright chromium not available: {exc}")
    yield env
    env.close()


def test_navigate_returns_title_and_text(browser_env, site):
    out = browser_env.execute("browser_navigate", {"url": site})
    assert "AgentSynth Test Home" in out
    assert "Welcome" in out


def test_read_and_find_and_links(browser_env, site):
    browser_env.execute("browser_navigate", {"url": site})
    assert "trajectories" in browser_env.execute("browser_read", {})
    assert browser_env.execute("browser_find", {"text": "verification"}).startswith("found")
    assert browser_env.execute("browser_find", {"text": "zzz-nope"}).startswith("not found")
    assert "pricing page" in browser_env.execute("browser_links", {}).lower()


def test_click_follows_the_link(browser_env, site):
    browser_env.execute("browser_navigate", {"url": site})
    out = browser_env.execute("browser_click", {"text": "pricing page"})
    assert "Pricing" in out
    assert "4242" in out


def test_bad_click_is_a_clean_observation(browser_env, site):
    browser_env.execute("browser_navigate", {"url": site})
    assert browser_env.execute("browser_click", {"text": "no such link"}).startswith("BrowserError")


def test_generator_drives_browser_tools(browser_env):
    from agentsynth import AgentTrajectoryGenerator

    gen = AgentTrajectoryGenerator(use_mock=True, environment=browser_env)
    traj = gen.generate("open the site and read the pricing page", mode="single_agent")
    used = traj.tool_names_used()
    assert used and all(name in BROWSER_TOOLS for name in used)
