# Cosimo Pipeline Architecture

This document maps the Cosimo synthetic dataset pipeline — components, data flow,
responsibilities — and the operational rules for running it. For agent behavioural
guidelines (including parallel agent loops) see [`AGENTS.md`](AGENTS.md); for the
record schema see [`dataset/FORMAT.md`](dataset/FORMAT.md); for the dataset
overview see [`dataset/README.md`](dataset/README.md).

**Path convention:** every path below is relative to the **repository root**. The
pipeline itself lives under `dataset/`.

**Counts are not documented here.** Record, shard, template and coverage numbers
are generated artifacts — read them from `dataset/progress/progress.md`, refreshed
by `dataset/pipelines/progress.py`. Numbers written into prose go stale silently.

---

## 1. Design principles

The pipeline is built around four invariants:

1. **Verifiability** — every numerical answer is computed by code, never
   hallucinated. Any record can be independently reproduced by re-executing its
   generating template from its stored seed.
2. **Determinism / resumability** — each `(program, template, variant)` tuple maps
   to a fixed seed, so generation is idempotent and can extend the corpus safely.
3. **Atomic, append-only shards** — records are written to a temp file and renamed
   into place on finalize; shards are crash-safe and resumable.
4. **Traces derived from computed numbers** — reasoning traces reference
   already-computed intermediates, so they are numerically consistent by
   construction.

---

## 2. Data flow at a glance

```
dataset/taxonomy/taxonomy.json ──► (topic scaffolding) ──► dataset/pipelines/templates/*.py
                                                        │  fn(rng, seq) -> rich dict
                                                        │  {meta, question, answer,
                                                        │   distractors, reasoning_trace,
                                                        │   flawed{...}}
                                                        ▼
dataset/config/seed.json ──► pipelines/generate.py ──► pipelines/core.py (IDs, RNG, fmt)
                          │  deterministic seed per (program, template, variant)
                          ▼
              dataset/shards/<program>/<program>_shard_XXXX.jsonl   (atomic, append-only)
                          │
                          ▼
              dataset/verification/verify_all.py   (multi-gate regression gate)
              dataset/verification/run_verify.py   (independent live harness)
                          │
                          ▼
              dataset/pipelines/progress.py ──► dataset/progress/progress.md  (live report)
              dataset/eval/ab_eval.py, diversity.py                (gold-bar A/B, novelty)
              dataset/goldbar/validate.py                          (gold-bar structural check)
```

---

## 3. Component map

### 3.1 Taxonomy (`dataset/taxonomy/taxonomy.json`)

Topic/subtopic scaffolding that informs template organization. It is the
curriculum source of truth; templates map topics to question stems.

### 3.2 Templates (`dataset/pipelines/templates/*.py`)

The generative heart of the pipeline. Exam modules correspond to a program
(`cfa_l1.py`, `cfa_l2.py`, `cfa_l3.py`, `frm1.py`, `frm2.py`); the `v2_*` modules
(`v2_analysis.py`, `v2_abstention.py`, `v2_agentic.py`, `v2_implementation.py`,
`v2_preference.py`) generate the non-exam record types. Each module exposes a
`TEMPLATES` dict mapping a stem name (e.g. `eq_gordon`, `tvm_annuity_fv`) to a
function `fn(rng, seq) -> dict` with this contract:

```python
{
  "meta":      {"topic", "subtopic", "difficulty", "question_type", "pitfalls_addressed", ...},
  "question":  "novel original question text",
  "answer":    "correct answer option letter + text",
  "distractors": ["a", "b", "c"],
  "reasoning_trace": "step-by-step CoT referencing computed intermediates",
  "flawed":    {"answer", "reasoning_trace", "pitfall"}   # numerically-grounded wrong variant
               # (or None -> record carries no preference_pair)
}
```

Records carry a `record_type` discriminator (`exam`, `analysis`, `abstention`,
`agentic`, `implementation`); required fields differ per type — `FORMAT.md` is
authoritative. Shared wrappers live in `wrappers.py` (`wrap_vignette`, `wrap_cr`,
`wrap_mcq`) and deterministically decorate a base stem into a `Vignette`,
`Constructed Response` (no distractors), or `MCQ` question.

### 3.3 Core helpers (`dataset/pipelines/core.py`)

Shared utilities consumed by the generator: content-hashed record IDs, the
deterministic `RNG` wrapper over `random.Random(seed)`, number/percent formatting,
and shard-path helpers. Defines `BASE_DIR`, `SHARDS_DIR` and `PROGRESS_DIR` —
see §5.1 for how these resolve.

### 3.4 Generator driver (`dataset/pipelines/generate.py`)

Orchestrates generation. For each program → template → variant:

- **Seed derivation** (deterministic): `seed = 1000 * hash((program, template)) % 10**9 + variant * 7919`, then `seed = seed % 2**31`. Variant index and template name are encoded into the record `seq` (`100000 + variant*1000 + abs(hash % 1000)`), making IDs reproducible.
- **Shard allocation**: `shard = produced // SHARD_SIZE`.
- **Write**: `append_record(program, shard, rec, finalize=False)` appends to a temp file; shards are **finalized** (renamed) at program end. Supports `PER_TEMPLATE`, `PROGRAM`, and `TEMPLATE` env filters.
- **Preference gating**: a pair is emitted only when the template returns `flawed`
  **and** `rng.r.random() < PAIR_RATIO` (read from `dataset/config/seed.json`
  `preference_pair_ratio`).
- **Deterministic finalize**: `_dedup_distractors` / `_dedup_wrong` rewrite any
  stored distractor or flawed wrong answer numerically equal to the correct
  answer (nudged by `+7.0`, preserving `$`/`%` formatting). It never touches
  question/answer/trace, so the reproducibility axes stay green.

### 3.5 Preference pairs

Built inline in `generate.py` (`build_preference`) whenever a template returns
`flawed` **and** `rng.r.random() < PAIR_RATIO`, plus the dedicated generators in
`templates/v2_preference.py`. The legacy `pipelines/preference.py` helper was
**removed**; inline + `v2_preference` only.

### 3.6 Verification (`dataset/verification/`)

- `verify_all.py` — the **regression gate** (must stay green). Loads every record
  on disk and runs these gates: structure, numeric reproducibility, format,
  implementation, agentic, preference, terminology (`terms.py`), response length
  (`length_gate.py`), and held-out suite overlap (`suite_overlap.py`).
  `--quick` skips the implementation and suite-overlap gates. Exits 0 on PASS,
  1 on FAIL.
- `gates.py` — shared plumbing: `Result` accumulator, record loading, per-type
  grouping, `supervised_text`, `approx_tokens`, percentiles.
- `terms.py` — terminology gate; blocks invented technical collocations.
- `length_gate.py` / `length_analysis.py` — response-length distribution gate and
  reporting.
- `suite_overlap.py` (+ `suite_overlap.json`) — guards against contaminating the
  held-out assistant-eval suites in `jobs/fine-tune/suites/`.
- `run_verify.py` — an independent live harness; also refreshes the progress page.
- `fix_distractors.py` — deterministic dedup fix: finds distractors numerically
  equal to the answer and perturbs them (preserves answer/trace integrity).
- `sanitize_distractors.py` — validates distractors stay within a plausible
  magnitude band of the answer.
- `nums.py` — robust numeric tokenizer (handles thousands separators) shared by
  the sanitizer and verification.

### 3.7 Progress (`dataset/pipelines/progress.py`)

Scans shards and emits the live report (`dataset/progress/progress.md` and
`progress.html`) of counts, coverage, and known gaps. This is the single source
of truth for corpus numbers.

### 3.8 Evaluation and gold bar (`dataset/eval/`, `dataset/goldbar/`)

- `goldbar/gold_bar.jsonl` — the curated assistant-transcript gold bar that
  defines the quality target.
- `goldbar/validate.py` — structural validation of the gold bar, per record type.
  Exits non-zero on failure.
- `eval/ab_eval.py` — blind A/B of generated records vs. the gold bar. Where the
  gold bar overlaps the shards, this is calibration-vs-gold, not an independent
  oracle.
- `eval/diversity.py` — structural-novelty report (distinct stems, per-topic
  coverage).

### 3.9 Config (`dataset/config/seed.json`)

Central seed configuration consumed by generation, including
`preference_pair_ratio`.

### 3.10 Scripts and publishing (`dataset/scripts/`, `dataset/publish/`)

- `scripts/smoke_generate.py` — Phase A verification: one variant per generator
  into a scratch directory (`dataset/.smoke/shards`). Proves the pipeline is whole
  without a bulk run — checks every generator produces a record, all record types
  are present, per-type required fields exist, numeric recomputation passes,
  agentic records render through the chat template, seeds are process-stable, and
  a second run is idempotent. Exits non-zero on any failure.
- `scripts/publish_dataset.py`, `publish/push_to_hub.py`, `publish/dataset_card.md`
  — Hub publishing and the dataset card.

---

## 4. Output artifacts

- **Shards** (`dataset/shards/<program>/<program>_shard_XXXX.jsonl`) — append-only,
  atomic, resumable. Gitignored.
- **Progress report** (`dataset/progress/progress.md`, `.html`) — live counts,
  coverage heat-map, response-length distribution, honest gaps.
- **Gold bar** (`dataset/goldbar/gold_bar.jsonl`) — curated exemplars defining the
  quality target.

---

## 5. Hard rules / conventions

1. **Paths are anchored to `dataset/`, not to your shell — with two exceptions.**
   Almost every script resolves `BASE_DIR` from `__file__`
   (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`) and builds
   absolute paths from it, so those commands work from **any** working directory.
   The exceptions glob a hardcoded relative `shards/` and **must be run with
   `dataset/` as the working directory**:
   - `eval/diversity.py` (line 7)
   - `scripts/publish_dataset.py` (line 142)

   Run from anywhere else they silently see **zero records** — `diversity.py` then
   crashes in `min()` on the empty counter. Either `cd dataset` first (see §6) or
   fix the glob to use `core.SHARDS_DIR`.

   Set `COSIMO_SHARDS_DIR` to redirect shard **and** progress output to a scratch
   tree — this is the supported way to keep a trial run, or a parallel agent, from
   touching the real corpus (see `AGENTS.md` §5.3). Note the two scripts above
   ignore it.
2. **Never hand-edit shard records.** Records are derived artifacts — regenerate
   via the pipeline. If you must repair data, use the deterministic fix scripts
   (`fix_distractors.py`, `sanitize_distractors.py`), which preserve answer/trace
   integrity.
3. **Generation is deterministic.** The seed is a pure function of
   `(program, template, variant)`; `seq` is reproducible. Never introduce
   wall-clock, PID, or dispatch-order randomness — it breaks resumability,
   idempotency, and every verification gate that recomputes from a stored seed.
4. **Shards are append-only and atomic.** `append_record(..., finalize=False)`
   writes a temp file; finalize renames it into place at program end. Don't write
   shards inline, and never let two writers target the same shard.
5. **Preference pairs are ratio-gated.** A pair is emitted only when a template
   returns `flawed` (`{"answer", "reasoning_trace", "pitfall"}`) **and**
   `rng.r.random() < PAIR_RATIO`, from `config/seed.json` `preference_pair_ratio`.
6. **Deterministic finalize is part of the contract.** `generate.py` rewrites any
   stored distractor or flawed wrong answer numerically equal to the correct
   answer (preserving `$`/`%` formatting). It never touches question/answer/trace.
   `run_verify.py` reproduces this finalize when checking stored pairs.
7. **Keep `verify_all.py` green.** It is the regression gate; any generator change
   requires re-running it (exit 0 == PASS, exit 1 == FAIL). A scoped or `--quick`
   run is not a substitute for the full gate before declaring work done.

---

## 6. Commands

Except where marked, these are safe to run from the repository root.

```bash
# Generate / extend (deterministic, resumable)
python3 dataset/pipelines/generate.py

# Filters
PER_TEMPLATE=50 python3 dataset/pipelines/generate.py
PROGRAM=CFA_Level_II python3 dataset/pipelines/generate.py
TEMPLATE=eq_gordon python3 dataset/pipelines/generate.py

# Generate into a scratch tree instead of the real corpus
COSIMO_SHARDS_DIR=/tmp/cosimo-scratch python3 dataset/pipelines/generate.py

# Full regression gate (must pass after any change)
python3 dataset/verification/verify_all.py
python3 dataset/verification/verify_all.py --quick   # skips implementation + suite-overlap

# Pipeline smoke test: one variant per generator, into a scratch dir
python3 dataset/scripts/smoke_generate.py

# Independent live harness + progress refresh
python3 dataset/verification/run_verify.py

# Progress report only
python3 dataset/pipelines/progress.py

# Gold bar structural validation
python3 dataset/goldbar/validate.py

# Quality fixes (deterministic)
python3 dataset/verification/fix_distractors.py
python3 dataset/verification/sanitize_distractors.py

# Evaluation
python3 dataset/eval/ab_eval.py

# Must run with dataset/ as CWD — globs a relative 'shards/' (see §5.1).
# diversity.py additionally assumes the v1 record shape and currently crashes (§8).
(cd dataset && python3 eval/diversity.py)
(cd dataset && python3 scripts/publish_dataset.py)
```

---

## 7. Adding a new question stem

1. Add a template function to the relevant module under
   `dataset/pipelines/templates/` — an exam program module or the `v2_*` module
   for the record type — matching the contract in §3.2.
2. Register it in that module's `TEMPLATES` dict (stem name → fn).
3. Generate it in isolation first:
   `TEMPLATE=<stem> python3 dataset/pipelines/generate.py`.
4. Run `python3 dataset/scripts/smoke_generate.py` — it must exit 0.
5. Run `python3 dataset/verification/verify_all.py` — it must exit 0.
6. Refresh `python3 dataset/pipelines/progress.py`.

---

## 8. Gotchas

Check these before editing.

- **Two scripts are CWD-dependent** — `eval/diversity.py` and
  `scripts/publish_dataset.py` glob a relative `shards/`, so from the repo root
  they report an empty corpus instead of erroring usefully. Everything else is
  `__file__`-anchored. See §5.1.
- **`eval/diversity.py` is stale against the v2 schema.** It reads
  `r['verification']['template']` and `r['metadata']['question_type']`
  unconditionally, which the non-`exam` record types do not carry — it raises
  `KeyError: 'template'` on the current corpus even with the right CWD. Fix it to
  `.get()` its way through, or restrict it to `record_type == "exam"`, before
  relying on its novelty numbers.
- **Counts in prose are unreliable.** `dataset/README.md` and
  `dataset/progress/progress.md` have disagreed on template and shard counts.
  Treat `progress.md` (regenerated from disk) as authoritative and reconcile the
  README rather than copying either into new documents.
- **Novelty is bounded by distinct stems, not rows.** Within-stem records differ
  only by sampled numbers. Diversity gains require new generators — re-randomising
  existing stems buys row count and nothing else.
- **Gold-bar overlap** — where the gold bar overlaps generated shards, `ab_eval.py`
  measures calibration, not an independent oracle.
- **Held-out suites are measurement instruments.** `jobs/fine-tune/suites/` must
  not be contaminated by near-duplicate generated prompts; `suite_overlap.py`
  checks this and the check must stay green.
- **`FINAL ANSWER:` is a grading contract, not house style.** It belongs on `exam`
  records only; the same applies to universal `ASSUMPTIONS:` / `Step N.` scaffolding.
  Uniform response shape is a known training failure mode — see `FORMAT.md`.
- **Preference pairs must not collide with SFT rows.** A `chosen` side that is also
  a supervised target makes the pair unusable for preference learning.
- **Dead code already removed** — `pipelines/preference.py`,
  `templates/cfa_l1_a.py`, and `templates/cfa_l1_b.py` were deleted. Don't
  reintroduce imports of them.
