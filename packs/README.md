# Scenario packs

A pack is a set of outcome-checked tasks over a seeded world: a scenario passes
only when the world ends up in the goal state, so a policy that talks its way
through scores nothing. Every pack ships with an oracle — a reference solution
that must pass 100% — which keeps the pack solvable and the leaderboard honest.

| Pack | Scenarios | World | Author |
| --- | --- | --- | --- |
| [`core_v1`](core_v1.yaml) | 10 | business ops over writable SQL | agentsynth |

Live leaderboards: [agentsynth.tech/leaderboard](https://agentsynth.tech/leaderboard).

## Contribute one

```bash
pip install agentsynth-ai
agentsynth pack new my_domain_v1 --dir packs
# or start from a real schema:  agentsynth pack new my_domain_v1 --from-schema db.sql
# edit packs/my_domain_v1.yaml + packs/my_domain_v1_oracle.py
agentsynth pack validate packs/my_domain_v1.yaml
```

`--from-schema` reads the first `CREATE TABLE` (needs an integer key and a
non-unique text column) and writes a three-scenario starter that already passes
the gate — a fast way in from a database you already have.

`validate` is the merge gate, and CI runs it on every PR:

- the schema parses, ids are unique, at least 3 scenarios;
- the oracle passes **every** scenario (proves the pack is solvable);
- two runs with the same seed agree (deterministic worlds);
- a do-nothing policy stays under 50% (checkers assert on the world, not the words).

Open a PR adding the two files plus a row in the table above. Once merged, the
pack is served by the hub, `agentsynth bench --pack my_domain_v1` works for
everyone, and it gets its own leaderboard at
`agentsynth.tech/leaderboard?pack=my_domain_v1`.

Good packs read like real work in your domain: a handful of tables, tasks with
a state change to verify, at least one "policy says no" scenario, and answer
checks that catch silent failures.
