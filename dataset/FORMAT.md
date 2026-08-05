# Cosimo Dataset Storage Format

## Record schema (JSONL, one record per line)

Every record carries a top-level `record_type` discriminator so the fine-tuning
harness can mix and weight them:

| `record_type` | Purpose |
|---|---|
| `exam` | Multiple-choice/calculation item with verified numeric answer, distractors, and reasoning trace. |
| `analysis` | Open-ended, prose question. Long-form answer. **No** `FINAL ANSWER:`, no mandatory step enumeration. |
| `abstention` | An underspecified, unanswerable, or false-premise prompt plus the response that identifies what is missing. |
| `agentic` | A full multi-turn conversation with tool schemas, calls, results, and a final answer. |
| `implementation` | A described model or paper, plus an idiomatic Python implementation and an honest note on what fails with real data. |

### `exam` record (extends the schema defined in version 1)

```json
{
  "id": "cosimo_<program>_<seq>_<sha>",
  "record_type": "exam",
  "program": "CFA_Level_I",
  "topic": "Quantitative Methods",
  "subtopic": "Time Value of Money",
  "difficulty": "L1_Medium",
  "question_type": "MCQ",
  "question": "novel original question text",
  "answer": "correct answer option letter + text",
  "distractors": ["a", "b", "c", "d"],
  "reasoning_trace": "step-by-step chain-of-thought with formulas and explicit assumptions",
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
    "generator_version": "gen_1.0.0"
  },
  "preference_pair": {
    "chosen": "full strong reasoning trace",
    "rejected": "plausible but wrong trace with one specific flaw",
    "pitfall": "the flaw being trained against"
  }
}
```

Constraints for `exam` records:
- `FINAL ANSWER:` **only** on `exam` records. It is a grading contract, not a house style.
- `ASSUMPTIONS:` and `Step N.` must not be universal. Vary the shape of exam traces:
  some prose, some enumerated, some tabular, some working backwards. A model cannot
  learn that structure is a *choice* if it only ever sees one structure.
- `reasoning_trace` must not collapse to 120 tokens — target a distribution with a
  substantial tail above 800 tokens.
- Every numeric answer is computed by code, never written. Verification re-executes
  from the stored seed and compares.
- `question_type` is one of `Calculation`, `Vignette`, `Constructed Response`, or `MCQ`.
  Wrappers in `pipelines/templates/wrappers.py`.

### `analysis` record — open-ended quantitative analysis

```json
{
  "id": "cosimo_analysis_<seq>_<sha>",
  "record_type": "analysis",
  "program": "CFA_Level_II",
  "topic": "Equity Valuation",
  "subtopic": "FCFF DCF",
  "difficulty": "L2_Hard",
  "question_type": "Analysis",
  "question": "The analyst wants to value a mature consumer-products company using an FCFF DCF. Walk through the key modelling decisions — working-capital normalization, capex vs depreciation treatment, terminal value assumptions — and explain how each choice affects the valuation. Include a numerical example using realistic parameters and show how the value changes under a ±20% perturbation of the terminal growth rate.",
  "answer": "Long-form prose answer (typically 800-2000 tokens). No FINAL ANSWER line. No mandatory step enumeration. The model's chain-of-thought IS the answer — it shows reasoning, shows work, shows trade-offs, and may include embedded numerical work.",
  "verified": true,
  "verification": {
    "status": "PASS",
    "method": "reference_code_exec",
    "checked": ["numerical_examples_in_answer"],
    "reruns": 0
  },
  "metadata": {
    "topic": "Equity Valuation",
    "subtopic": "FCFF DCF",
    "difficulty": "L2_Hard",
    "question_type": "Analysis",
    "source": "synthetic_generator",
    "source_template": "analysis_eq_fcff_dcf_001",
    "seed": 45678,
    "generator_version": "gen_1.0.0",
    "answer_token_length": 1250
  }
}
```

Constraints for `analysis` records:
- **NO** `FINAL ANSWER:`, **NO** `ASSUMPTIONS:`, **NO** required `Step N.` enumeration.
- Answers must be long-form, varied in structure, and show genuine reasoning depth.
- Where numbers appear they are computed, not hallucinated.
- `reasoning_trace` field is **not used** (answers live in `answer`).
- `distractors` field is **not used**.
- `preference_pair` may carry chosen/rejected pairs of analyses for DPO training (see Preference pairs section below).

### `abstention` record — calibration training

```json
{
  "id": "cosimo_abstention_<seq>_<sha>",
  "record_type": "abstention",
  "program": "CFA_Level_III",
  "topic": "Portfolio Management",
  "subtopic": "Rebalancing",
  "difficulty": "L3_Hard",
  "question_type": "Calibration",
  "defect": "underspecified",
  "question": "Rebalance my portfolio to its target allocations.",
  "answer": "short-form calibration response (50-300 tokens). The model identifies what information is missing (current holdings, target allocations, assets/brokerage) and asks for them, or states what it cannot do without more input.",
  "verified": true,
  "verification": {
    "status": "PASS",
    "method": "structural",
    "checks": ["no_final_answer", "asks_for_info", "identifies_gap"]
  },
  "metadata": {
    "topic": "Portfolio Management",
    "subtopic": "Rebalancing",
    "difficulty": "L3_Hard",
    "question_type": "Calibration",
    "source": "synthetic_generator",
    "source_template": "abstention_underspecified_001",
    "seed": 78901,
    "generator_version": "gen_1.0.0",
    "answer_token_length": 120
  }
}
```

Constraints for `abstention` records:
- `defect` field is one of: `underspecified`, `unanswerable`, `false_premise`.
- `answer` must be a *correct* response: ask for missing info, decline to answer, or
  point out the false premise. A confident wrong answer fails verification.
- **NO** `FINAL ANSWER:`, **NO** `distractors`, **NO** `reasoning_trace`.
- `preference_pair` is valuable here: *asked for missing input* vs *answered anyway*.

### `agentic` record — multi-turn tool conversation

```json
{
  "id": "cosimo_agentic_<seq>_<sha>",
  "record_type": "agentic",
  "program": "CFA_Level_II",
  "topic": "Equity Valuation",
  "subtopic": "Multiplier Models",
  "difficulty": "L2_Medium",
  "question_type": "Agentic",
  "question": "I need to compare valuation multiples between Apple and Microsoft. Can you pull their PE ratios and ROE figures?",
  "tool_schemas": [
    {"type": "function", "function": {"name": "get_fundamentals", "description": "...", "parameters": {...}}}
  ],
  "conversation": [
    {"role": "user", "content": "I need to compare valuation multiples..."},
    {"role": "assistant", "tool_calls": [{"name": "get_fundamentals", "arguments": {"symbol": "AAPL", "metrics": ["pe", "roe"]}}]},
    {"role": "tool_result", "content": "{\"AAPL\": {\"pe\": 28.4, \"roe\": 1.47}}"},
    {"role": "assistant", "tool_calls": [{"name": "get_fundamentals", "arguments": {"symbol": "MSFT", "metrics": ["pe", "roe"]}}]},
    {"role": "tool_result", "content": "{\"MSFT\": {\"pe\": 32.1, \"roe\": 0.39}}"},
    {"role": "assistant", "content": "Apple trades at 28.4x P/E with 147% ROE while Microsoft..."}
  ],
  "answer": "Final assistant response combining tool results.",
  "verified": true,
  "verification": {
    "status": "PASS",
    "method": "structural",
    "checks": ["tool_calls_valid", "tool_results_present", "answer_uses_results"]
  },
  "metadata": {
    "topic": "Equity Valuation",
    "subtopic": "Multiplier Models",
    "difficulty": "L2_Medium",
    "question_type": "Agentic",
    "source": "synthetic_generator",
    "source_template": "agentic_fundamentals_001",
    "seed": 23456,
    "generator_version": "gen_1.0.0",
    "call_depth": 2,
    "answer_token_length": 350
  }
}
```

Constraints for `agentic` records:
- `conversation` is a full multi-turn history (2-5 tool calls for the target corpus).
- Tools may succeed or fail; include recovery from failed calls.
- **NO** `FINAL ANSWER:`, **NO** `reasoning_trace`, **NO** `distractors`.
- Tool schemas must match the real tool definitions from `jobs/fine-tune/cosimo_ft/tools.py`
  or a clearly documented synthetic tool set.
- Each tool call result is structurally valid JSON matching the schema.

### `implementation` record — paper → code

```json
{
  "id": "cosimo_impl_<seq>_<sha>",
  "record_type": "implementation",
  "program": "CFA_Level_II",
  "topic": "Quantitative Methods",
  "subtopic": "Machine Learning",
  "difficulty": "L2_Hard",
  "question_type": "Implementation",
  "question": "Implement a single-stage FCFF DCF valuation model in Python. The model should: (a) accept a sequence of projected free cash flows for years 1 through N, (b) compute the present value using a supplied discount rate, (c) compute terminal value using both perpetuity-growth and exit-multiple approaches, (d) return enterprise value and equity value per share. Include docstrings, type hints, and a numerical example.",
  "answer": "Long-form answer containing an idiomatic Python implementation with comments explaining key decisions.",
  "code": "```python\ndef fcff_dcf(fcfs, discount_rate, terminal_g):\n    \"\"\"...\"\"\"\n    pv = sum(cf / (1 + discount_rate)**t for t, cf in enumerate(fcfs, 1))\n    # ... implementation ...\n```\n\n### What breaks with real data\n\n- The model assumes FCFF is available for every year; in practice, LBO models back into it from revenue assumptions.\n- Terminal growth rate > risk-free rate breaks the perpetuity formula.\n- Exit multiples can diverge significantly from growth-based TV if the company's trajectory changes.\n\nImplementation notes about production concerns, edge cases, and what would fail with real data.",
  "verified": true,
  "verification": {
    "status": "PASS",
    "method": "reference_code_exec",
    "checks": ["code_executes", "numerical_output_matches", "no_import_errors"]
  },
  "metadata": {
    "topic": "Quantitative Methods",
    "subtopic": "Machine Learning",
    "difficulty": "L2_Hard",
    "question_type": "Implementation",
    "source": "synthetic_generator",
    "source_template": "impl_fcff_dcf_001",
    "seed": 34567,
    "generator_version": "gen_1.0.0",
    "answer_token_length": 1800,
    "code_token_length": 950
  }
}
```

Constraints for `implementation` records:
- `code` is a fenced Python code block. Must execute without errors.
- `answer` includes both the code and an honest assessment of production limitations.
- **NO** `FINAL ANSWER:`, **NO** `reasoning_trace`, **NO** `distractors`.
- The model description in `question` may be loosely inspired by a real paper or framework.
- The `answer` must state what would break with real data (data quality, computational limits, assumptions).

---

## Record_type-specific formatting rules

| Rule | exam | analysis | abstention | agentic | implementation |
|---|---|---|---|---|---|
| `FINAL ANSWER:` | required | not present | not present | not present | not present |
| `reasoning_trace` | present | not present | not present | not present | not present |
| `answer` | computed value | long-form prose | calibration response | final answer | code + analysis |
| `distractors` | present | not present | not present | not present | not present |
| `conversation` | not present | not present | not present | present | not present |
| `code` | not present | not present | not present | not present | present |
| `defect` | not present | not present | present | not present | not present |
| `tool_schemas` | not present | not present | not present | present | not present |

## Verification guarantee

`exam` records with `verified: true` have their final numerical answer checked by executing
independent reference code. The reasoning trace is *derived from* the computed numbers, so
traces are numerically consistent by construction. Records that fail verification are dropped
and never written to a shard.

All other record types carry a `verification` block but may use `method: "structural"`
when no numerical ground truth exists (e.g., checking format, tool-call validity, etc.).

## Sharding

- 500 records per shard: `shards/<program>/<program>_<record_type>_shard_XXXX.jsonl`
- Shards are append-only, atomic (write to temp then rename), and resumable.
- A shard is only finalized (renamed) after all 500 records verify.

## Preference pairs

Preference pairs now live in a **disjoint id space** from the SFT rows, or carry an
explicit flag (`preference_pair.mode`) the preparation step can split on: `"sft_train"`
(pairs whose `chosen` also appears as SFT target) vs `"dpo_only"` (pairs whose `chosen`
appears **only** in preference data). The harness validates zero overlap between `chosen`
and SFT training rows.

The `rejected` must be plausible and wrong for an interesting reason. Target patterns:
- right method with a wrong assumption
- a confident answer to an underspecified question (abstention context)
- a correct calculation answering a different question than the one asked
- a fluent answer with an invented term
- For `exam`: wrong formula branch, sign error, unit error — while keeping structure varied

Cover every `record_type`, not just `exam`. The most valuable pairs are `abstention` ones:
*asked for the missing input* versus *answered anyway*.

**Never** let the `chosen` side of a preference pair be identical to an SFT target row.
This was the cause of the DPO no-op in version 1: SFT trained on reasoning traces that
were also the `chosen` side, causing 100% overlap, saturated rewards, and `loss = 0.0` from step 10.

## Question types

`question_type` values depend on `record_type`:

- `exam`: `Calculation`, `Vignette`, `Constructed Response`, `MCQ` (wrappers in `pipelines/templates/wrappers.py`)
- `analysis`: `Analysis`
- `abstention`: `Calibration`
- `agentic`: `Agentic`
- `implementation`: `Implementation`

## Integrity

- Every record id is a content hash of the question + verified answer. For non-exam
  records the hash includes `record_type`, `defect` (for abstention), or other
  identifying fields.
- No proprietary CFA/FRM exam items are used; all content is original, inspired
  only by public Learning Outcome Statements.
- Technical terms in every record must pass a terminology gate (valid eponym paired
  with correct concept, not fabricated collocations like "Durbin-Watson duration").
