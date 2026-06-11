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
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _INDEX_HTML or "<h1>AgentSynth</h1><p><a href='/leaderboard'>Leaderboard</a></p>"


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


@app.get("/v1/leaderboard")
def leaderboard(pack: str = "core_v1", limit: int = 50) -> dict:
    if pack not in PACKS:
        raise HTTPException(status_code=404, detail="no such pack")
    with SessionLocal() as db:
        rows = (
            db.query(Submission)
            .filter(Submission.pack_id == pack)
            .order_by(Submission.pass_rate.desc(), Submission.created_at.asc())
            .limit(max(1, min(limit, 200)))
            .all()
        )
    best: Dict[str, Submission] = {}
    for row in rows:
        if row.model not in best:
            best[row.model] = row
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


@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page(pack: str = "core_v1") -> str:
    data = leaderboard(pack=pack)
    rows = (
        "".join(
            f"<tr><td>{e['rank']}</td><td>{_esc(e['model'])}</td>"
            f"<td>{e['pass_rate']:.0%}</td><td>{e['passed']}/{e['n']}</td></tr>"
            for e in data["entries"]
        )
        or '<tr><td colspan="4">No submissions yet — be the first.</td></tr>'
    )
    cmd = (
        "pip install agentsynth-ai && agentsynth bench "
        f"--pack {pack} --model &lt;model&gt; --submit https://api.agentsynth.tech"
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>AgentSynth — {pack} leaderboard</title>
<style>body{{font:16px/1.5 system-ui;max-width:720px;margin:3rem auto;padding:0 1rem;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}
code{{background:#f4f4f4;padding:.2rem .4rem;border-radius:4px;font-size:14px}}</style></head>
<body><h1>{pack} leaderboard</h1>
<p>Outcome-checked scenarios: a run passes only when the world ends up in the goal state.
Reproduce any entry — packs are deterministic.</p>
<table><tr><th>#</th><th>model</th><th>pass rate</th><th>scenarios</th></tr>{rows}</table>
<p>Submit yours: <code>{cmd}</code></p>
<p><a href="https://github.com/agentsynth/agentsynth">github.com/agentsynth/agentsynth</a></p>
</body></html>"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
