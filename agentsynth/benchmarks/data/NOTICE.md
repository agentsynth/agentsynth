# Bundled BFCL sample

`bfcl_sample.questions.jsonl` and `bfcl_sample.answers.jsonl` are a small slice
(25 cases) of the **Berkeley Function-Calling Leaderboard (BFCL)** `simple_python`
split, redistributed here so the benchmark runs out of the box.

- Source: https://github.com/ShishirPatil/gorilla (`berkeley-function-call-leaderboard`)
- License: Apache License 2.0 — © the Gorilla / BFCL authors.

It's a convenience sample for a quick before/after signal, not the full suite. Point
`load_bfcl(questions_path, answers_path)` at the official BFCL files to score the whole
benchmark, and use BFCL's own AST checker for leaderboard-grade numbers.
