# Scenario Hub

The API behind api.agentsynth.tech: serves the scenario packs, takes
`agentsynth bench --submit` results, and renders the leaderboard. A submission
that carries a run manifest gets a public per-scenario log at `/runs/{id}`
(`/v1/submissions/{id}` for the JSON) and, when the client metered an LLM
policy, a cost column on the board.

Run locally:

    cd hub
    pip install -r requirements.txt
    uvicorn app.main:app --reload     # SQLite unless DATABASE_URL is set

Deploy: build `hub/Dockerfile` with the repo root as context, set DATABASE_URL
(Neon Postgres) and optionally MAX_SUBMISSIONS_PER_HOUR.
