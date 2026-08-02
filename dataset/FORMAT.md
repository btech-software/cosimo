# Cosimo Dataset Storage Format

## Record schema (JSONL, one record per line)

```json
{
  "id": "cosimo_<program>_<seq>_<sha>",
  "program": "CFA_Level_I",
  "topic": "Quantitative Methods",
  "subtopic": "Time Value of Money",
  "difficulty": "L1_Medium",
  "question_type": "MCQ",
  "question": "novel original question text",
  "answer": "correct answer option letter + text",
  "distractors": ["a", "b", "c", "d"],
  "reasoning_trace": "full step-by-step Chain-of-Thought with formulas and explicit assumptions",
  "verified": true,
  "verification": {
    "status": "PASS",
    "method": "reference_code_exec",
    "final_answer_match": true,
    "checked": ["formula1", "formula2", "final_number"],
    "reruns": 3
  },
  "metadata": {
    "topic": "...",
    "subtopic": "...",
    "difficulty": "...",
    "question_type": "...",
    "pitfalls_addressed": ["..."],
    "source": "synthetic_generator",
    "seed": 12345,
    "generator_version": "gen_0.1.0"
  },
  "preference_pair": {
    "chosen": "full strong reasoning trace",
    "rejected": "flawed reasoning trace with a specific pitfall",
    "pitfall": "the flaw being trained against"
  }
}
```

## Verification guarantee
Every record with `verified: true` has its final numerical answer checked by
executing independent reference code. The reasoning trace is *derived from* the
computed numbers, so traces are numerically consistent by construction. Records
that fail verification are dropped and never written to a shard.

## Sharding
- 500 records per shard: `shards/<program>/<program>_<topic>_<shard>.jsonl`
- Shards are append-only, atomic (write to temp then rename), and resumable.
- A shard is only finalized (renamed) after all 500 records verify.

## Preference pairs
~35% of records carry `preference_pair`, gated deterministically by
`config/seed.json` `preference_pair_ratio`. The `chosen` is the verified strong
trace; `rejected` is a generated flawed trace that makes exactly one targeted
pitfall error (wrong formula, sign flip, unit error, etc.) while keeping the
question identical. A write-time finalize (`_dedup_wrong` in
`pipelines/generate.py`) guarantees the stored `wrong_answer` never numerically
equals `correct_answer`, so every emitted pair is concrete.

## Question types
`question_type` is one of `Calculation`, `Vignette`, or `Constructed Response`
(wrappers in `pipelines/templates/wrappers.py`). `Constructed Response` records
carry **no** `distractors`; `Vignette` records keep the underlying calculation's
answer and distractors but re-prompt as an item-set. This yields DPO/ORPO-ready pairs.

## Integrity
- Every record id is a content hash of the question + verified answer.
- No proprietary CFA/FRM exam items are used; all content is original, inspired
  only by public Learning Outcome Statements.
