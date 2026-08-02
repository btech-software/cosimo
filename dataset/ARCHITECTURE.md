# Cosimo Pipeline Architecture

This document maps the Cosimo synthetic dataset pipeline: components, data flow,
and responsibilities. It is code-reference oriented; for operational guidance see
[`AGENTS.md`](AGENTS.md), for the record schema see [`FORMAT.md`](FORMAT.md).

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
taxonomy/taxonomy.json ──► (topic scaffolding) ──► pipelines/templates/*.py
                                                        │  fn(rng, seq) -> rich dict
                                                        │  {meta, question, answer,
                                                        │   distractors, reasoning_trace,
                                                        │   flawed{...}}
                                                        ▼
config/seed.json ──► pipelines/generate.py ──► pipelines/core.py (IDs, RNG, fmt)
                          │  deterministic seed per (program, template, variant)
                          ▼
                 shards/<program>/<program>_shard_XXXX.jsonl   (500 rec/shard, atomic)
                          │
                          ▼
                 verification/verify_all.py  (4-axis regression gate)
                 verification/run_verify.py  (independent live harness)
                          │
                          ▼
                 pipelines/progress.py ──► progress/progress.md   (live report)
                 eval/ab_eval.py, eval/diversity.py               (gold-bar A/B)
                 verification/fix_distractors.py, sanitize_distractors.py (quality fixes)
```

---

## 3. Component map

### 3.1 Taxonomy (`taxonomy/taxonomy.json`)

Topic/subtopic scaffolding that informs template organization. It is the
curriculum source of truth; templates map topics to question stems.

### 3.2 Templates (`pipelines/templates/*.py`)

The generative heart of the pipeline. Each module corresponds to a program
(`cfa_l1.py`, `cfa_l2.py`, `cfa_l3.py`, `frm1.py`, `frm2.py`) and exposes a
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

Template counts by module: **55 total** — CFA L1 (28), CFA L2 (9), CFA L3 (7),
FRM1 (6), FRM2 (5). Shared wrappers live in `pipelines/templates/wrappers.py`
(`wrap_vignette`, `wrap_cr`) and deterministically decorate a base stem into a
`Vignette` or `Constructed Response` question (the latter carries no distractors).

### 3.3 Core helpers (`pipelines/core.py`)

Shared utilities consumed by the generator: content-hashed record IDs, the
deterministic `RNG` wrapper over `random.Random(seed)`, number/percent formatting,
and shard-path helpers.

### 3.4 Generator driver (`pipelines/generate.py`)

Orchestrates generation. For each program → template → variant:

- **Seed derivation** (deterministic): `seed = 1000 * hash((program, template)) % 10**9 + variant * 7919`, then `seed = seed % 2**31`. Variant index and template name are encoded into the record `seq` (`100000 + variant*1000 + abs(hash % 1000)`), making IDs reproducible.
- **Shard allocation**: `shard = produced // SHARD_SIZE` (500 records/shard).
- **Write**: `append_record(program, shard, rec, finalize=False)` appends to a temp file; shards are **finalized** (renamed) at program end. Supports `PER_TEMPLATE`, `PROGRAM`, and `TEMPLATE` env filters.
- **Preference gating**: a pair is emitted only when the template returns `flawed`
  **and** `rng.r.random() < PAIR_RATIO` (read from `config/seed.json`
  `preference_pair_ratio`, default 0.35) — yielding ~35% of records with pairs.
- **Deterministic finalize**: `_dedup_distractors` / `_dedup_wrong` rewrite any
  stored distractor or flawed wrong answer numerically equal to the correct
  answer (nudged by `+7.0`, preserving `$`/`%` formatting). It never touches
  question/answer/trace, so the reproducibility axes stay green.

### 3.5 Preference pairs

Preference pairs are built inline in `generate.py` (`build_preference`) whenever
a template returns `flawed` **and** `rng.r.random() < PAIR_RATIO`. The legacy
helper `pipelines/preference.py` was **removed**; pairing is inline-only.

### 3.6 Verification (`verification/`)

- `verify_all.py` — the **regression gate** (must stay green). Scans all records
  and checks four axes:
  1. final answer reproducible from template + seed;
  2. reasoning trace byte-identical to deterministic recomputation;
  3. preference-pair wrong answer concrete and ≠ correct;
  4. no distractor numerically equals the correct answer.
  Exits 0 on PASS, 1 on FAIL.
- `run_verify.py` — an independent live harness; also refreshes the progress page.
- `fix_distractors.py` — deterministic dedup fix: finds distractors numerically
  equal to the answer and perturbs them (preserves answer/trace integrity).
- `sanitize_distractors.py` — validates distractors stay within a plausible
  magnitude band of the answer.
- `nums.py` — robust numeric tokenizer (handles thousands separators) shared by
  the sanitizer and verification.

### 3.7 Progress (`pipelines/progress.py`)

Scans shards and emits a live report (`progress/progress.md`) of counts,
coverage, and known gaps.

### 3.8 Evaluation (`eval/`)

- `ab_eval.py` — blind A/B of generated records vs. the curated gold bar
  (`goldbar/gold_bar.jsonl`). Because the gold bar overlaps the shards, this is
  calibration-vs-gold, not an independent oracle.
- `diversity.py` — structural-novelty report (distinct stems, per-topic coverage).

### 3.9 Config (`config/seed.json`)

Central seed configuration consumed by generation.

---

## 4. Output artifacts

- **Shards** (`shards/<program>/<program>_shard_XXXX.jsonl`): 142 shards × 500
  records = **71,000 records** (CFA L1 33k / L2 12k / L3 9k / FRM1 10k / FRM2 7k).
  Append-only, atomic, resumable.
- **Progress report** (`progress/progress.md`): live counts + honest gaps.
- **Gold bar** (`goldbar/gold_bar.jsonl`): curated exemplars for A/B.

---

## 5. Known issues

These are intentional-to-document findings; keep them in mind when editing.

- **Preference-ratio resolved** — pairs are gated at ~35% via
  `config/seed.json` `preference_pair_ratio` (see generator §3.4); the
  deterministic finalize guarantees every emitted pair is concrete (wrong ≠
  correct). The legacy `pipelines/preference.py` helper was removed.
- **Question-type wrappers added** — `pipelines/templates/wrappers.py`
  (`wrap_vignette`, `wrap_cr`, `wrap_mcq`) implement `Vignette`, `Constructed
  Response`, and `MCQ` qtypes.
- **Dead code removed** — `pipelines/templates/cfa_l1_a.py`, `cfa_l1_b.py`, and
  `pipelines/preference.py` were deleted.
- **Absolute-path coupling** — verification/generate scripts `sys.path.insert`
  absolute repo paths and glob relative `shards/`; they **must be run from the
  repository root** (see `AGENTS.md`).
- **Structural novelty bound** — 71 distinct stems caps diversity; within-stem
  records differ only by sampled numbers.
- **Gold-bar overlap** — the gold bar overlaps generated shards, so A/B is
  calibration, not independent oracle.
