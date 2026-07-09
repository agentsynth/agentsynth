# Submitting `main.tex` to arXiv

## Category

- Primary: `cs.SE` (Software Engineering) — the paper is centrally about
  benchmark/tooling correctness, not a new model or a new learning method.
- Cross-list: `cs.CL`, `cs.LG` — agent evaluation and RLVR readers live there.

## Before uploading

- [ ] Confirm the final author line in `main.tex` (name(s), affiliation string).
- [ ] Run `python paper/experiments.py` and confirm it produces no diff
      against the committed `numbers.json` (CI does this on every push that
      touches `paper/**`; check the `Paper` workflow is green on the commit
      being submitted).
- [ ] `tectonic paper/main.tex` and proofread the built PDF once end to end —
      table numbers, section refs, and citation numbers all resolve.
- [ ] Add a `\thanks{}` or Acknowledgments line for any external contributor
      credit that isn't full co-authorship (e.g. a merged PR too small for
      authorship but worth a mention).
- [ ] Update `CITATION.cff`'s `preferred-citation` block to point at the
      arXiv id once assigned.

## arXiv's LaTeX package pins

arXiv's TeX Live is usually a year or so behind. All packages used here
(`amsmath`, `booktabs`, `microtype`, `hyperref`, `xurl`, `listings`, `xcolor`)
are old and stable — no known compatibility risk, but rebuild once with
arXiv's own compiler after upload before finalizing, since their engine can
differ subtly from `tectonic`.
