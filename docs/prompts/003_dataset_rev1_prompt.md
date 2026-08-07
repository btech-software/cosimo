# Cosimo dataset v2 — iteration 3

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

## State — measured, not assumed

**All three gates currently exit 0.** That is the baseline you must not break.

**Done. Do not redo, do not re-audit, do not regenerate:**

- The generation pipeline: deterministic, resumable, idempotent. 290 generators, all run.
- The verification harness: 9 gates in `verification/`, blind A/B in `eval/ab_eval.py`,
  gold-bar validator in `goldbar/validate.py`.
- **Gold bar: 200/200 transcripts pass strict validation.** Agentic transcripts carry real
  6-turn conversations with tool calls; implementation code parses; exam traces present;
  abstention defects tagged. **Do not regenerate any of it.**
- **Parameter variety: 100%** across abstention, agentic, analysis, implementation.
- **Analysis length: p50 936 approx tokens**, inside FORMAT.md's 800–2000 spec. The length
  gate passes.

**Current generator counts:**

| Module | Generators | | Module | Generators |
|---|---:|---|---|---:|
| `cfa_l1` | 33 | | `v2_analysis` | 43 |
| `cfa_l2` | 20 | | `v2_abstention` | 62 |
| `cfa_l3` | 24 | | `v2_agentic` | 33 |
| `frm1` | 32 | | `v2_implementation` | 26 |
| `frm2` | 17 | | `v2_preference` | 8 |

**The corpus is still 6,704 rows and 99.96% exam records. No bulk generation has run.**

## Definition of done

These three commands all exit 0:

```
python3 scripts/smoke_generate.py      # 8 checks, incl. parameter variety
python3 goldbar/validate.py            # gold bar schema + composition
python3 verification/verify_all.py     # 9 gates across all 6 record types
```

**The gates are the spec.** Do not weaken a threshold, loosen a required-field list, or
special-case a record to make one pass. If you believe a threshold is wrong, say so and
stop.

---

## Do the work in this order

### 1. TAXONOMY COVERAGE — the highest-value item

**62 of 207 subtopics appear in generated records (30.0%).** The brief calls closing this
the single highest-value thing the corpus does. It has moved 55 → 57 → 62 across two
iterations; at that rate it will never close.

| Program | Covered | Declared | Gap |
|---|---:|---:|---:|
| CFA Level I | 15 | 58 | **43** |
| CFA Level II | 9 | 55 | **46** |
| CFA Level III | 21 | 45 | 24 |
| FRM Part 1 | 9 | 21 | 12 |
| FRM Part 2 | 8 | 28 | 20 |
| **Total** | **62** | **207** | **145** |

Write **new generators** for uncovered subtopics, registered in the right program module.
Do not add variants to existing stems — that is the memorisation failure, not coverage.
CFA Levels I and II are the worst gaps and should be attacked first.

Read the uncovered list from `taxonomy/taxonomy.json` against what records actually
produce. **Do not trust `taxonomy/coverage.json`** — see Known traps below.

### 2. TRACE VARIETY

`core.render_trace` exists and **zero** templates use it. All exam generators still
hand-build `ASSUMPTIONS:` / `Step N.` strings, which is exactly what taught the model that
being Cosimo means answering in four steps. This is a headline failure of the first run and
it is still completely unaddressed.

Migrate generators to return structured `{assumptions, steps, conclusion}` and render
through `render_trace`. The style is drawn from the variant's seeded RNG, so recomputation
stays byte-identical and the numeric gate keeps working. Ids are unaffected — the trace is
not in the id payload.

Verify: sample generated exam records and confirm more than one trace shape appears.

### 3. PREFERENCE PAIRS

`pipelines/templates/v2_preference.py` has **8** generators proving all four failure modes
(`false_confidence`, `wrong_assumption`, `answers_different_question`, `invented_term`).
Expand substantially, weighted toward `false_confidence` — abstention pairs are the
highest-value type and the 62 abstention generators already supply the `chosen` side.

Pairs live in the `cosimopref_` id namespace and shard to `shards/preference/`. Keep it
that way: the disjoint namespace is what makes SFT/DPO overlap structurally impossible.

### 4. ONLY THEN, BULK GENERATE

```
PER_TEMPLATE=250 python3 -m pipelines.generate
```

Row count is an **outcome of generator count**, not a dial. Hold variants per generator at
≤250. The v1 corpus was 71 stems × 1000 variants and produced a 45-point generalisation
gap (62.9% in-domain vs 18.0% held-out). Reaching 200K means ~800 generators, not more
variants. Generation is resumable and idempotent; re-running skips what already exists.

---

## Known traps — each one has already cost a session

**A syntax error can hide a runtime error.** `cfa_l3_new.py` had four f-string/bracket
syntax errors *and* an `rng.random()` call that only fails when executed (`core.RNG`
exposes the generator as `.r`). Fixing the syntax alone would have merged generators that
die on every call. **Always execute a new generator across several seeds, never just
import it.**

**`fmt()` returns a string.** `f"{fmt(x):,.0f}"` raises `Unknown format code 'f' for
object of type 'str'`. Use `fmt(x, 0)`. **There are 9 latent instances of this pattern in
`cfa_l2.py`** in code paths not currently reached — if you touch those generators, fix
them.

**`taxonomy/coverage.json` is stale and measures something different.** It counts
subtopics with a *declared template*; the number that matters is subtopics appearing in
*generated records*. It still reports CFA L3 at 9/45 when `cfa_l3.py` has 24 templates.
Regenerate it or ignore it, but do not plan from it.

**A green gate is not proof.** A previous iteration reported the gold bar complete at
200/200 while 118 transcripts were structurally hollow — the validator was too permissive.
Before reporting a step done, check the gate actually measures the thing you changed.

**Do not work in file copies.** Two iterations have left ~30 scratch files behind
(`*_new.py`, `*_expanded*.py`, `*_final.py`, `*.bak`, loose scripts at the repo root).
Twice, real work was stranded in a broken copy and nearly deleted. **Edit the real module
in place.** If you need a scratch file, delete it before you finish.

---

## Hard constraints

- **Never edit `jobs/fine-tune/suites/*.jsonl`.** They are held-out measurement
  instruments. The `suite_overlap` gate already caught one real contamination.
- **`FINAL ANSWER:` only on exam records.** It is a grading contract, not a house style.
- **Every number computed by code, never written.** Verification re-executes from the
  stored seed.
- **Agentic conversation roles are `user` / `assistant` / `tool`** — never `tool_result`.
  The chat template does not recognise it and the tool output silently lands in the
  supervised span.
- **Bump `round` in `config/seed.json`** before a new generation round.
- **Do not touch `jobs/fine-tune/`.** The training-harness work is separate.
- **Leave no untracked scratch files.** `git status` must be clean of `??` entries other
  than deliberate new modules when you finish.

## Report honestly

State what passed, what failed, and what you did not get to, with numbers. Do not report a
step complete because a gate went green. If you run out of scope, stop at a clean boundary
and say exactly where you stopped — a partial step reported accurately is worth more than a
complete one reported vaguely.
