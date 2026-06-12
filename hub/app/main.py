"""The Scenario Hub: packs, submissions, and a leaderboard.

Serves the scenario packs from `packs/`, accepts benchmark submissions from
`agentsynth bench --submit`, and renders a leaderboard per pack. Storage is
whatever DATABASE_URL points at (Neon Postgres in production, SQLite by default).

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine, func, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

PACKS_DIR = Path(os.environ.get("PACKS_DIR", Path(__file__).resolve().parents[2] / "packs"))
MAX_SUBMISSIONS_PER_HOUR = int(os.environ.get("MAX_SUBMISSIONS_PER_HOUR", "20"))


class Base(DeclarativeBase):
    pass


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    pack_id = Column(String(120), index=True, nullable=False)
    model = Column(String(200), nullable=False)
    pass_rate = Column(Float, nullable=False)
    passed = Column(Integer, nullable=False)
    n = Column(Integer, nullable=False)
    results = Column(JSON, nullable=False)
    client_version = Column(String(40), default="")
    ip_hash = Column(String(64), default="")


def _engine():
    url = os.environ.get("DATABASE_URL", "sqlite:///./hub.db")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    kwargs: Dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


engine = _engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base.metadata.create_all(engine)

app = FastAPI(title="AgentSynth Scenario Hub", version="0.1.0")


def _load_packs() -> Dict[str, List[Dict[str, Any]]]:
    import yaml

    packs: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(PACKS_DIR.glob("*.yaml")):
        try:
            packs[path.stem] = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return packs


PACKS = _load_packs()


class SubmissionIn(BaseModel):
    pack_id: str
    model: str = Field(min_length=1, max_length=200)
    report: Dict[str, Any]
    client_version: str = ""


_recent: Dict[str, List[float]] = {}


def _rate_limited(ip: str) -> bool:
    now = time.time()
    window = [t for t in _recent.get(ip, []) if now - t < 3600]
    window.append(now)
    _recent[ip] = window
    return len(window) > MAX_SUBMISSIONS_PER_HOUR


_INDEX_HTML = ""
_index_file = Path(__file__).resolve().parent / "index.html"
if _index_file.exists():
    _INDEX_HTML = _index_file.read_text(encoding="utf-8")

_OG_FILE = Path(__file__).resolve().parent / "og.png"

# Inline SVG so both pages get a tab icon without another asset to serve.
_FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
    "%3Crect width='64' height='64' rx='14' fill='%234f46e5'/%3E"
    "%3Cpath d='M18 33l10 10 18-22' stroke='%23fff' stroke-width='7' fill='none' "
    "stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _INDEX_HTML or "<h1>AgentSynth</h1><p><a href='/leaderboard'>Leaderboard</a></p>"


@app.get("/og.png", include_in_schema=False)
def og_image() -> FileResponse:
    if not _OG_FILE.exists():
        raise HTTPException(status_code=404, detail="no og image")
    return FileResponse(_OG_FILE, media_type="image/png")


@app.get("/healthz")
def healthz() -> dict:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok", "packs": sorted(PACKS)}


@app.get("/v1/packs")
def list_packs() -> dict:
    return {
        pack_id: {"scenarios": len(scenarios), "ids": [s.get("id") for s in scenarios]}
        for pack_id, scenarios in PACKS.items()
    }


@app.get("/v1/packs/{pack_id}")
def get_pack(pack_id: str) -> list:
    if pack_id not in PACKS:
        raise HTTPException(status_code=404, detail="no such pack")
    return PACKS[pack_id]


@app.post("/v1/submissions", status_code=201)
def submit(payload: SubmissionIn, request: Request) -> dict:
    if payload.pack_id not in PACKS:
        raise HTTPException(status_code=404, detail=f"unknown pack '{payload.pack_id}'")
    report = payload.report
    n = int(report.get("n", 0))
    passed = int(report.get("passed", -1))
    pass_rate = report.get("pass_rate")
    results = report.get("results")
    if n != len(PACKS[payload.pack_id]):
        raise HTTPException(status_code=422, detail="report size doesn't match the pack")
    if not isinstance(results, list) or len(results) != n:
        raise HTTPException(status_code=422, detail="report.results must cover every scenario")
    if not (isinstance(pass_rate, (int, float)) and 0.0 <= pass_rate <= 1.0 and 0 <= passed <= n):
        raise HTTPException(status_code=422, detail="implausible pass numbers")

    ip = (request.client.host if request.client else "") or ""
    if _rate_limited(ip):
        raise HTTPException(status_code=429, detail="too many submissions; try later")

    with SessionLocal() as db:
        row = Submission(
            pack_id=payload.pack_id,
            model=payload.model.strip()[:200],
            pass_rate=float(pass_rate),
            passed=passed,
            n=n,
            results=[{"id": r.get("id"), "passed": bool(r.get("passed"))} for r in results],
            client_version=payload.client_version[:40],
            ip_hash=hashlib.sha256(ip.encode()).hexdigest()[:16],
        )
        db.add(row)
        db.commit()
        return {"id": row.id, "pack_id": row.pack_id, "pass_rate": row.pass_rate}


def _best_per_model(pack: str, limit: int = 200) -> List[Submission]:
    """Each model's best run for a pack, ranked."""
    with SessionLocal() as db:
        rows = (
            db.query(Submission)
            .filter(Submission.pack_id == pack)
            .order_by(Submission.pass_rate.desc(), Submission.created_at.asc())
            .limit(max(1, min(limit, 500)))
            .all()
        )
    best: Dict[str, Submission] = {}
    for row in rows:
        if row.model not in best:
            best[row.model] = row
    return list(best.values())


@app.get("/v1/leaderboard")
def leaderboard(pack: str = "core_v1", limit: int = 50) -> dict:
    if pack not in PACKS:
        raise HTTPException(status_code=404, detail="no such pack")
    best = {row.model: row for row in _best_per_model(pack, limit)}
    entries = [
        {
            "rank": i + 1,
            "model": row.model,
            "pass_rate": row.pass_rate,
            "passed": row.passed,
            "n": row.n,
            "submitted": row.created_at.isoformat() if row.created_at else None,
        }
        for i, row in enumerate(best.values())
    ]
    return {"pack": pack, "entries": entries}


@app.get("/v1/packs/{pack_id}/breakdown")
def pack_breakdown(pack_id: str) -> dict:
    """Per-scenario pass rates across each model's best run — what breaks models."""
    if pack_id not in PACKS:
        raise HTTPException(status_code=404, detail="no such pack")
    best = _best_per_model(pack_id)
    counts: Dict[str, Dict[str, int]] = {}
    for row in best:
        for r in row.results or []:
            sid = str(r.get("id"))
            slot = counts.setdefault(sid, {"passes": 0, "attempts": 0})
            slot["attempts"] += 1
            if r.get("passed"):
                slot["passes"] += 1
    scenarios = [
        {
            "id": sid,
            "attempts": c["attempts"],
            "passes": c["passes"],
            "pass_rate": round(c["passes"] / c["attempts"], 4) if c["attempts"] else None,
        }
        for sid, c in counts.items()
    ]
    scenarios.sort(key=lambda s: (s["pass_rate"] if s["pass_rate"] is not None else 2.0, s["id"]))
    return {"pack": pack_id, "models": len(best), "scenarios": scenarios}


@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page(pack: str = "core_v1") -> str:
    data = leaderboard(pack=pack)
    rows = (
        "".join(
            f"<tr><td>{e['rank']}</td><td class='m'>{_esc(e['model'])}</td>"
            f"<td><b>{e['pass_rate']:.0%}</b></td><td>{e['passed']}/{e['n']}</td>"
            f"<td class='d'>{(e['submitted'] or '')[:10]}</td></tr>"
            for e in data["entries"]
        )
        or '<tr><td colspan="5" class="empty">No submissions yet — be the first.</td></tr>'
    )
    n_scenarios = len(PACKS.get(pack, []))
    cmd = (
        "pip install agentsynth-ai && agentsynth bench "
        f"--pack {pack} --model &lt;model&gt; --submit https://api.agentsynth.tech"
    )

    hardest_html = ""
    breakdown = pack_breakdown(pack)
    toughest = [s for s in breakdown["scenarios"] if s["pass_rate"] is not None][:3]
    if toughest and breakdown["models"] >= 2:
        rows_h = "".join(
            f"<tr><td class='m'>{_esc(s['id'])}</td>"
            f"<td><b>{s['pass_rate']:.0%}</b></td>"
            f"<td class='d'>{s['passes']}/{s['attempts']} models</td></tr>"
            for s in toughest
        )
        hardest_html = (
            '<h2 class="sub-h">Hardest scenarios</h2>'
            '<table><tr><th>scenario</th><th>pass rate</th><th class="d">across</th></tr>'
            f"{rows_h}</table>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>AgentSynth — {pack} leaderboard</title>
<link rel="icon" href="{_FAVICON}">
<style>
:root{{--ink:#11141a;--muted:#5b6471;--line:#e7e9ee;--accent:#4f46e5;--accent-soft:#eef0fe}}
*{{box-sizing:border-box}}
body{{margin:0;background:#fff;color:var(--ink);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
a{{color:var(--accent);text-decoration:none}} a:hover{{text-decoration:underline}}
.wrap{{max-width:760px;margin:0 auto;padding:0 20px}}
header{{border-bottom:1px solid var(--line)}}
nav{{display:flex;align-items:center;justify-content:space-between;height:64px}}
.brand{{font-weight:700;font-size:18px;color:var(--ink)}} .brand b{{color:var(--accent)}}
nav .links a{{color:var(--ink);margin-left:22px;font-size:15px}}
nav .links a:hover{{color:var(--accent)}}
h1{{font-size:30px;letter-spacing:-.02em;margin:40px 0 6px}}
h2.sub-h{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
margin:34px 0 6px;font-weight:700}}
p.sub{{color:var(--muted);margin:0 0 28px}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:left;font-size:15px}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
td.m{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px}}
td.d{{color:var(--muted);font-size:13.5px}}
td.empty{{color:var(--muted)}}
.how{{margin:30px 0 60px;background:var(--accent-soft);border-radius:12px;padding:18px 20px}}
.how p{{margin:0 0 10px;font-size:14.5px}}
code{{display:block;background:#0e1117;color:#e6edf3;border-radius:8px;padding:12px 14px;
font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-x:auto}}
@media(max-width:560px){{td.d,th.d{{display:none}}}}
</style></head>
<body>
<header><div class="wrap"><nav>
<a class="brand" href="/">Agent<b>Synth</b></a>
<span class="links"><a href="https://github.com/agentsynth/agentsynth">GitHub</a>
<a href="https://agentsynth.github.io/agentsynth/">Docs</a></span>
</nav></div></header>
<div class="wrap">
<h1>{pack} leaderboard</h1>
<p class="sub">{n_scenarios} outcome-checked scenarios — a run passes only when the world
ends up in the goal state. Packs are deterministic; reproduce any entry.</p>
<table><tr><th>#</th><th>model</th><th>pass rate</th><th>scenarios</th>
<th class="d">submitted</th></tr>
{rows}</table>
{hardest_html}
<div class="how"><p><b>Get on the board</b> — any LiteLLM model, or your own agent loop via
<a href="https://github.com/agentsynth/agentsynth#bench-a-model-get-on-the-leaderboard">--policy</a>:</p>
<code>{cmd}</code></div>
</div>
</body></html>"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
