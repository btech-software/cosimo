# Cosimo dataset v2 — iteration 2

Work in `dataset/`. This prompt is self-contained; the original brief is preserved at
`docs/prompts/003_dataset_rev1_brief.md` — read it for the full rationale, but do not
wait on it to start.

## Objective

A corpus that teaches an assistant to a **Head of Quantitative Asset Management**: one
that reasons about valuation, risk, market microstructure and papers, and is honest
about what it does not know. Exam accuracy is a milestone, not the target. The first
corpus raised exam accuracy while collapsing the model's mean response from ~750 tokens
to 120 and teaching it to answer every question as a four-step exam trace. Everything
below exists to stop that recurring.

## State

**Done — do not redo, do not re-audit:**
- The generation pipeline. Deterministic, resumable, idempotent, all 277 generators run.
- The verification harness: 9 gates in `verification/`, the blind A/B in `eval/ab_eval.py`,
  the gold-bar validator in `goldbar/validate.py`.
- 70 analysis gold transcripts (mean 579 tokens, genuinely varied — median pairwise
  Jaccard 0.08). **Keep these. Do not regenerate them.**

**Not done:** everything in the ordered list below. The corpus is still 6,704 rows and
99.96% exam records; no bulk generation has run.

## Definition of done

These three commands all exit 0:

```
python3 scripts/smoke_generate.py      # 8 checks, incl. parameter variety
python3 goldbar/validate.py            # gold bar schema + composition
python3 verification/verify_all.py     # 9 gates across all 6 record types
```

**The gates are the spec.** Do not weaken a threshold, loosen a required-field list, or
special-case a record to make one pass. If you believe a threshold is wrong, say so and
stop. A previous iteration reported success against a validator that was too permissive;
that is the specific failure this rule prevents.

---

## Do the work in this order

### 0. REPAIR THE GOLD BAR — 148 structural failures

The bar is 200 transcripts with the right composition, but **118 of them are shells**.
`goldbar/validate.py` now checks per-type structure and fails:

| Missing field | Count |
|---|---|
| `agentic.conversation` | 30 of 30 |
| `agentic.tool_schemas` | 30 of 30 |
| `implementation.code` | 25 of 25 |
| `exam.reasoning_trace` | 33 of 45 |
| `abstention.metadata.defect` | 30 of 30 |

Fix these in place — keep the questions and answers already written, add the missing
structure:

- **agentic** needs a real `conversation` (list of turns) and `tool_schemas`. Roles are
  `user` / `assistant` / `tool` — **never `tool_result`**, which the chat template does
  not recognise. Assistant tool calls use
  `{"type":"function","function":{"name":..., "arguments":{...}}}`. Cover single-call,
  multi-call, parallel-call, failed-call recovery, and no-call-appropriate.
- **implementation** needs `code`: idiomatic Python that parses and executes, plus an
  honest note on what breaks with real data.
- **exam** needs a `reasoning_trace`. Deliberately vary the shape across the 45 — some
  prose, some enumerated, some tabular, some working backwards from the answer.
- **abstention** needs `metadata.defect` ∈ `underspecified` / `unanswerable` /
  `false_premise`.

This is first because the bar is the reference the blind A/B scores against. Hollow
agentic transcripts would make generated agentic records win by default.

Verify: `python3 goldbar/validate.py` → PASS.

### 1. PARAMETER VARIETY

`abstention`, `agentic` and `implementation` each produce **one unique question per
generator regardless of seed** (20%). Parameterise the question *text* — a different
instrument, a different defect, a different tool set — not just the numbers.

Bulk generation is pointless until this is fixed: the idempotency skip would collapse
28,600 intended abstention rows to 62.

Verify: smoke check 8 reads >90% for every type.

### 2. ANALYSIS LENGTH

The length gate fails. The 43 generators in `pipelines/templates/v2_analysis.py` produce
answers averaging **117 approx tokens** against FORMAT.md's 800–2000 spec; mixed-set p95
is 178 against an 800 floor.

Rewrite them to produce genuinely long-form answers that show reasoning — mechanism,
trade-offs, what would change the conclusion — not just more numbers. The 70 analysis
gold transcripts are the quality reference for what this should look like.

Verify: `python3 verification/length_gate.py` → PASS.

### 3. TAXONOMY COVERAGE

**55 of 207 subtopics have a generator (26.6%).** The brief calls closing this the single
highest-value thing the corpus does.

| Program | Gap |
|---|---:|
| CFA Level I | 43 |
| CFA Level II | 46 |
| CFA Level III | 28 |
| FRM Part 1 | 15 |
| FRM Part 2 | 20 |

Write **new generators** for uncovered subtopics. Do not add variants to existing stems —
that is the memorisation failure, not coverage.

### 4. TRACE VARIETY

`core.render_trace` exists and **zero** templates use it. All 103 exam generators still
hand-build `ASSUMPTIONS:` / `Step N.` strings, which is exactly what taught the model that
being Cosimo means answering in four steps.

Migrate them to return structured `{assumptions, steps, conclusion}` and render through
`render_trace`. The style is drawn from the variant's seeded RNG, so recomputation stays
byte-identical and the numeric gate keeps working. Ids are unaffected — the trace is not
in the id payload.

### 5. PREFERENCE PAIRS

`pipelines/templates/v2_preference.py` has 8 generators proving all four failure modes
(`false_confidence`, `wrong_assumption`, `answers_different_question`, `invented_term`).
Expand for coverage across record types, weighted toward `false_confidence` — abstention
pairs are the highest-value type and the 62 abstention generators already supply the
`chosen` side.

Pairs live in the `cosimopref_` id namespace and shard to `shards/preference/`. Keep it
that way: the disjoint namespace is what makes SFT/DPO overlap structurally impossible.

### 6. ONLY THEN, BULK GENERATE

```
PER_TEMPLATE=250 python3 -m pipelines.generate
```

Row count is an **outcome of generator count**, not a dial. Hold variants per generator at
≤250. The v1 corpus was 71 stems × 1000 variants and produced a 45-point generalisation
gap (62.9% in-domain vs 18.0% held-out). Reaching 200K means ~800 generators, not more
variants. Generation is resumable and idempotent; re-running skips what already exists.

---

## Hard constraints

- **Never edit `jobs/fine-tune/suites/*.jsonl`.** They are held-out measurement
  instruments. The `suite_overlap` gate already caught one real contamination.
- **`FINAL ANSWER:` only on exam records.** It is a grading contract, not a house style.
- **Every number computed by code, never written.** Verification re-executes from the
  stored seed.
- **Bump `round` in `config/seed.json`** before a new generation round.
- **Do not touch `jobs/fine-tune/`.** The training-harness work is separate and is not
  part of this goal.

## Housekeeping

Delete the scratch files a previous iteration left in `dataset/goldbar/`: `a1.py`–`al.py`,
`write_*.py`, `gold_bar.jsonl.bak3`, `.bak4`, `gold_bar_clean.jsonl`, `gold_bar_new.jsonl`.
The `.bak3`/`.bak4` files are identical copies of the current 200-line bar; `_clean` (82)
and `_new` (62) are superseded partial drafts. Work in the real file, not in copies.

## Report honestly

State what passed, what failed, and what you did not get to. Do not report a step complete
because a gate went green — check that the gate actually measures the thing. If you run out
of scope, stop at a clean boundary and say exactly where you stopped.
