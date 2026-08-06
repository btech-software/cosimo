# Cosimo Synthetic CFA/FRM Dataset

Cosimo is a **synthetic, verified** financial-exam question dataset for training
reasoning models and preference-tuned (DPO/ORPO) models. It contains **71,000**
original, numerically-grounded questions across
the CFA Level I–III and FRM Part 1/2 curricula.

Every numerical answer is **computed by code, never hallucinated**. Each record
carries a step-by-step reasoning trace *derived from* the computed intermediates,
so traces are numerically consistent by construction. A large fraction of records
also carries a `preference_pair` (chosen vs. flawed rejected trace) for
preference-learning.

---

## Contents

- [What this dataset is](#what-this-dataset-is)
- [Dataset composition](#dataset-composition)
- [Record schema (JSONL)](#record-schema-jsonl)
- [Integrity guarantees](#integrity-guarantees)
- [Directory layout](#directory-layout)
- [Quick start](#quick-start)
- [Extending / regenerating](#extending--regenerating)
- [Further reading](#further-reading)

---

## What this dataset is

Cosimo generates original exam-style questions from per-topic **templates**
(stems). Each template `fn(rng, seq)` returns a rich dict with the question,
answer, distractors, a reasoning trace, and a numerically-grounded *flawed*
variant. A deterministic RNG per `(program, template, variant)` makes generation
**resumable and idempotent**: re-running never duplicates and can extend the
corpus over days.

The design goal is *verifiability*: every claim in the dataset can be
independently reproduced and checked by re-executing the generating template from
its stored seed.

## Dataset composition

| Program | Templates | Records | Shards |
|---|---|---|---|---|
| CFA Level I | 33 | 33,000 | 66 |
| CFA Level II | 12 | 12,000 | 24 |
| CFA Level III | 9 | 9,000 | 18 |
| FRM Part 1 | 10 | 10,000 | 20 |
| FRM Part 2 | 7 | 7,000 | 14 |
| **Total** | **71** | **71,000** | **142** |

- **Shard layout**: `shards/<program>/<program>_shard_XXXX.jsonl`, 500 records
  per shard.
- **Shards are append-only** and written atomically (temp file, then `os.replace`
  on finalize), so they are crash-safe and resumable.
- **Question types**: `Calculation`, `Vignette`, `Constructed Response`, and
  `MCQ` (wrapped via `pipelines/templates/wrappers.py`). See [Known gaps](#known-gaps).

## Record schema (JSONL)

One JSON record per line. The authoritative spec is [`FORMAT.md`](FORMAT.md); a
representative shape:

```json
{
  "id": "cosimo_<program>_<seq>_<sha>",
  "program": "CFA_Level_I",
  "topic": "Quantitative Methods",
  "subtopic": "Time Value of Money",
  "difficulty": "L1_Medium",
  "question_type": "Calculation",
  "question": "novel original question text",
  "answer": "correct answer option letter + text",
  "distractors": ["a", "b", "c"],
  "reasoning_trace": "full step-by-step CoT with formulas and explicit assumptions",
  "verified": true,
  "verification": {
    "status": "PASS",
    "method": "reference_code_exec",
    "final_answer_match": true,
    "checked": ["formula1", "formula2", "final_number"],
    "reruns": 3
  },
  "metadata": {
    "topic": "...", "subtopic": "...", "difficulty": "...",
    "question_type": "...", "pitfalls_addressed": ["..."],
    "source": "synthetic_generator", "seed": 12345,
    "generator_version": "gen_0.1.0"
  },
  "preference_pair": {
    "chosen": "full strong reasoning trace",
    "rejected": "flawed reasoning trace with a specific pitfall",
    "pitfall": "the flaw being trained against"
  }
}
```

`preference_pair` is present on **~35%** of generated records, gated
deterministically by `config/seed.json` → `preference_pair_ratio` (default 0.35).
The `chosen` is the verified strong trace; `rejected` is a generated flawed trace
that commits exactly one targeted pitfall error (wrong formula, sign flip, unit
error, …) while keeping the question identical — yielding DPO/ORPO-ready pairs.
During generation a deterministic finalize step guarantees the stored
`wrong_answer` never numerically equals the correct answer, so every pair is
concrete.

## Integrity guarantees

1. **Answers are computed, not guessed.** Templates compute then stringify; the
   verification gate re-runs the template from the stored seed and compares the
   recomputed answer to the persisted one.
2. **Traces are derived from the computed numbers.** The trace text references
   already-computed intermediates, so traces are numerically consistent by
   construction.
3. **Verification gate.** `verification/verify_all.py` scans all records and
   checks four axes: final answer reproducible, reasoning trace byte-identical
   to deterministic recomputation, preference-pair wrong answer concrete and ≠
   correct, and no distractor numerically equals the correct answer.
4. **Deterministic finalize.** A write-time guard in `pipelines/generate.py`
   (`_dedup_distractors` / `_dedup_wrong`) rewrites any distractor or flawed
   wrong answer that numerically equals the correct answer, preserving `$`/`%`
   formatting. It never touches question/answer/trace, so reproducibility axes
   stay green.
5. **Content-hashed IDs.** Every record id is a content hash of the question +
   verified answer.

## Directory layout

```
dataset/
├── FORMAT.md                  # record schema specification
├── config/seed.json           # generation seed configuration
├── eval/                      # gold-bar A/B + diversity reports
├── goldbar/gold_bar.jsonl     # curated gold-bar exemplars (A/B baseline)
├── pipelines/                 # generation pipeline
│   ├── core.py                # shared helpers: IDs, RNG, formatting
│   ├── generate.py            # main generator driver
│   ├── progress.py            # live progress reporting
│   └── templates/             # per-topic question templates (71 stems)
│       ├── wrappers.py        # vignette / constructed-response wrappers
│       ├── cfa_l1.py … frm2.py
├── progress/                  # live progress report (gitignored, generated)
├── publish/                   # Hugging Face publishing script + dataset card
├── shards/                    # output dataset (gitignored, generated)
├── taxonomy/taxonomy.json     # curriculum taxonomy
└── verification/              # verification & quality fixes
    ├── verify_all.py          # 4-axis full-scan regression gate
    ├── run_verify.py          # independent live verification harness
    ├── fix_distractors.py     # deterministic distractor dedup fix
    ├── sanitize_distractors.py# distractor magnitude sanitizer
    └── nums.py                # robust numeric tokenizer
```

Generated output (`shards/`, `progress/`) is **gitignored**: the released
corpus lives on Hugging Face at
[btech-software/cosimo-cfa-frm-71k](https://huggingface.co/datasets/btech-software/cosimo-cfa-frm-71k),
and any shard is deterministically regenerable from the pipeline.

## Quick start

Run from **this `dataset/` directory** (the scripts resolve imports relative to
their own location and glob relative `shards/`).

```bash
# Generate / extend the dataset (deterministic, resumable)
python3 pipelines/generate.py

# Per-program or per-template generation
PER_TEMPLATE=50 python3 pipelines/generate.py
PROGRAM=CFA_Level_II python3 pipelines/generate.py
TEMPLATE=eq_gordon python3 pipelines/generate.py

# Full 4-axis verification gate (exit code 0 == PASS)
python3 verification/verify_all.py

# Independent live verification harness + progress page refresh
python3 verification/run_verify.py

# Refresh the live progress report only
python3 pipelines/progress.py

# Distractor quality fixes (already applied; deterministic, preserves integrity)
python3 verification/fix_distractors.py
python3 verification/sanitize_distractors.py

# Gold-bar A/B evaluation
python3 eval/ab_eval.py
python3 eval/diversity.py
```

## Extending / regenerating

- Add a new question stem by adding a template function to the relevant
  `pipelines/templates/<program>.py` module and registering it in that module's
  `TEMPLATES` dict. See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the exact
  contract and the step-by-step checklist.
- Generation is deterministic per `(program, template, variant)`, so extending
  the corpus is safe and idempotent.
- **Always** re-run `verification/verify_all.py` after any generator change — it
  is the regression gate that must stay green.

## Known gaps

Honest current limitations (mirrored in `progress/progress.md`):

- Structural novelty is bounded by **71 distinct stems**; within-stem records
  differ only in sampled numbers.
- Question types are `Calculation`, `Vignette`, `Constructed Response`, and
  `MCQ` (implemented via `wrap_mcq` in `pipelines/templates/wrappers.py`).
- The curated gold bar (`goldbar/gold_bar.jsonl`) overlaps the generated shards,
  so `eval/ab_eval.py` is calibration-vs-gold, not an independent oracle.

## Further reading

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — component map, data flow, conventions,
  run commands, and gotchas.
- [`AGENTS.md`](../AGENTS.md) — behavioural guidelines for agents/humans working in
  this repo, including parallel agent loops.
- [`FORMAT.md`](FORMAT.md) — authoritative record schema.
