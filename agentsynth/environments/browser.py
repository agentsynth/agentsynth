"""A browser environment backed by Playwright.

Point the generator at a real (headless) Chromium browser so trajectories carry
grounded web tool-use — opening pages, reading their text, following links — instead
of templated observations. Playwright is heavy and optional, so we import it lazily
and only launch a browser the first time a tool actually runs.

    env = BrowserEnvironment(start_url="https://example.com")
    gen = AgentTrajectoryGenerator(environment=env)

Requires `pip install "agentsynth-ai[browser]"` plus a one-time
`python -m playwright install chromium` (Python 3.10+).

Not safe to share across threads: Playwright's sync objects are bound to the thread
that launched them, so for concurrent generation use one environment per worker (or
the inline runner).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..schemas import ToolSpec
from ..utils import stable_seed
from .base import Environment

_DEFAULT_START = "https://example.com"
_SNIPPET = 600  # chars of page text returned after navigate / click
_MAX_TEXT = 2000  # cap on browser_read output
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_STOP = {"the", "and", "for", "with", "that", "this", "from", "what", "where", "which", "your"}


def _clean_text(text: str, limit: int = _MAX_TEXT) -> str:
    """Strip each line, drop the blank ones, and cap the total length."""
    cleaned = "\n".join(ln.strip() for ln in (text or "").splitlines() if ln.strip())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _pick_keyword(query: str, seed: int) -> str:
    """A deterministic content word from the query, for sample tool args."""
    words = [w for w in _WORD_RE.findall(query or "") if w.lower() not in _STOP]
    if not words:
        return "example"
    return words[stable_seed(seed, "browser") % len(words)]


def _find_snippet(text: str, needle: str, width: int = 160) -> Optional[str]:
    """The slice of `text` around the first case-insensitive hit of `needle`."""
    i = text.lower().find((needle or "").lower())
    if i < 0:
        return None
    start = max(0, i - width // 2)
    end = min(len(text), i + len(needle) + width // 2)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _one_string(name: str, desc: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": desc}},
        "required": [name],
    }


def _no_args() -> Dict[str, Any]:
    return {"type": "object", "properties": {}, "required": []}


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else exc.__class__.__name__


class BrowserEnvironment(Environment):
    name = "browser"

    def __init__(
        self,
        start_url: str = _DEFAULT_START,
        headless: bool = True,
        timeout: float = 20.0,
    ) -> None:
        self._start_url = start_url
        self._headless = headless
        self.timeout = timeout
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None

    # -- browser lifecycle ---------------------------------------------------

    def _page_obj(self) -> Any:
        """Launch Chromium on first use and hand back the live page."""
        if self._page is None:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self._headless)
            self._page = self._browser.new_page()
            self._page.set_default_timeout(self.timeout * 1000)
        return self._page

    def reset(self) -> None:
        if self._page is not None:
            try:
                self._page.goto("about:blank")
            except Exception:
                pass

    def close(self) -> None:
        for obj, meth in ((self._browser, "close"), (self._pw, "stop")):
            try:
                if obj is not None:
                    getattr(obj, meth)()
            except Exception:
                pass
        self._pw = self._browser = self._page = None

    # -- Environment API -----------------------------------------------------

    def tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                name="browser_navigate",
                description="Open a URL and return the page title plus a text excerpt.",
                parameters=_one_string("url", "The URL to open"),
            ),
            ToolSpec(
                name="browser_read",
                description="Return the visible text of the current page.",
                parameters=_no_args(),
            ),
            ToolSpec(
                name="browser_links",
                description="List the visible link texts on the current page.",
                parameters=_no_args(),
            ),
            ToolSpec(
                name="browser_find",
                description="Search the current page for text and return the surrounding snippet.",
                parameters=_one_string("text", "Text to look for on the page"),
            ),
            ToolSpec(
                name="browser_click",
                description="Click the link or button matching the text and return the new page.",
                parameters=_one_string("text", "Visible text of the element to click"),
            ),
        ]

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        args = args or {}
        try:
            if tool_name == "browser_navigate":
                return self._navigate(str(args.get("url", "")).strip())
            if tool_name == "browser_read":
                return self._read()
            if tool_name == "browser_links":
                return self._links()
            if tool_name == "browser_find":
                return self._find(str(args.get("text", "")).strip())
            if tool_name == "browser_click":
                return self._click(str(args.get("text", "")).strip())
        except Exception as exc:  # Playwright raises a wide range of errors
            return f"BrowserError: {_first_line(exc)}"
        raise KeyError(tool_name)

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        if tool_name == "browser_navigate":
            return {"url": self._start_url}
        if tool_name in ("browser_find", "browser_click"):
            return {"text": _pick_keyword(query, seed)}
        return {}

    # -- tool implementations ------------------------------------------------

    def _navigate(self, url: str) -> str:
        if not url:
            return "BrowserError: no url given"
        if "://" not in url:
            url = "https://" + url
        self._page_obj().goto(url, wait_until="domcontentloaded")
        return self._summary()

    def _read(self) -> str:
        if self._page is None:
            return "BrowserError: no page open; navigate first"
        return _clean_text(self._page.inner_text("body")) or "(empty page)"

    def _links(self) -> str:
        if self._page is None:
            return "BrowserError: no page open; navigate first"
        texts = self._page.locator("a").evaluate_all(
            "els => els.map(e => (e.innerText || '').trim()).filter(Boolean)"
        )
        seen: List[str] = []
        for raw in texts:
            text = " ".join(raw.split())
            if text and text not in seen:
                seen.append(text)
            if len(seen) >= 30:
                break
        return "\n".join(f"- {t}" for t in seen) if seen else "(no links)"

    def _find(self, text: str) -> str:
        if self._page is None:
            return "BrowserError: no page open; navigate first"
        if not text:
            return "BrowserError: no search text given"
        snippet = _find_snippet(_clean_text(self._page.inner_text("body"), 100_000), text)
        return f"found: {snippet}" if snippet else f"not found: {text!r}"

    def _click(self, text: str) -> str:
        if self._page is None:
            return "BrowserError: no page open; navigate first"
        if not text:
            return "BrowserError: no click text given"
        page = self._page
        target = page.locator("a, button").filter(has_text=text)
        if target.count() == 0:
            target = page.get_by_text(text)
        if target.count() == 0:
            return f"BrowserError: nothing matching {text!r} to click"
        target.first.click()
        page.wait_for_load_state("domcontentloaded")
        return self._summary()

    def _summary(self) -> str:
        page = self._page
        title = (page.title() or "").strip()
        head = f"[{title}] {page.url}" if title else page.url
        body = _clean_text(page.inner_text("body"), _SNIPPET)
        return f"{head}\n{body}" if body else head


__all__ = ["BrowserEnvironment"]
