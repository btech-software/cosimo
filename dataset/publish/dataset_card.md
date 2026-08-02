---
license: mit
pretty_name: Cosimo CFA/FRM Synthetic Reasoning Dataset
language:
- en
task_categories:
- question-answering
- text-generation
tags:
- finance
- cfa
- frm
- synthetic
- reasoning
- chain-of-thought
- dpo
- orpo
- preference-optimization
- exam
size_categories:
- 10K<n<100K
dataset_info:
- config_name: default
  features:
  - name: id
    dtype: string
  - name: program
    dtype: string
  - name: topic
    dtype: string
  - name: subtopic
    dtype: string
  - name: difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: distractors
    list: string
  - name: reasoning_trace
    dtype: string
  - name: verified
    dtype: bool
  - name: verification
    struct:
    - name: answer_matches_recomputation
      dtype: bool
    - name: flawed_answer_concrete
      dtype: string
    - name: method
      dtype: string
    - name: recomputed
      dtype: bool
    - name: seed
      dtype: int64
    - name: template
      dtype: string
  - name: metadata
    struct:
    - name: difficulty
      dtype: string
    - name: generator
      dtype: string
    - name: generator_version
      dtype: string
    - name: pitfalls_addressed
      list: string
    - name: question_type
      dtype: string
    - name: seed
      dtype: int64
    - name: source
      dtype: string
    - name: subtopic
      dtype: string
    - name: topic
      dtype: string
  - name: preference_pair
    struct:
    - name: chosen
      struct:
      - name: answer
        dtype: string
      - name: reasoning_trace
        dtype: string
    - name: pitfall
      dtype: string
    - name: rejected
      struct:
      - name: answer
        dtype: string
      - name: reasoning_trace
        dtype: string
  splits:
  - name: cfa_level_i
    num_bytes: 33862988
    num_examples: 33000
  - name: cfa_level_ii
    num_bytes: 11679506
    num_examples: 12000
  - name: cfa_level_iii
    num_bytes: 8906918
    num_examples: 9000
  - name: frm_part_1
    num_bytes: 8894989
    num_examples: 10000
  - name: frm_part_2
    num_bytes: 6619198
    num_examples: 7000
  download_size: 9642224
  dataset_size: 69963599
- config_name: preference_pairs
  features:
  - name: id
    dtype: string
  - name: program
    dtype: string
  - name: topic
    dtype: string
  - name: subtopic
    dtype: string
  - name: difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: prompt
    dtype: string
  - name: chosen
    struct:
    - name: answer
      dtype: string
    - name: reasoning_trace
      dtype: string
  - name: rejected
    struct:
    - name: answer
      dtype: string
    - name: reasoning_trace
      dtype: string
  - name: pitfall
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 18935143
    num_examples: 24711
  download_size: 2898321
  dataset_size: 18935143
configs:
- config_name: default
  data_files:
  - split: cfa_level_i
    path: data/cfa_level_i-*
  - split: cfa_level_ii
    path: data/cfa_level_ii-*
  - split: cfa_level_iii
    path: data/cfa_level_iii-*
  - split: frm_part_1
    path: data/frm_part_1-*
  - split: frm_part_2
    path: data/frm_part_2-*
- config_name: preference_pairs
  data_files:
  - split: train
    path: preference_pairs/train-*
---

# Cosimo: Synthetic CFA/FRM Financial Reasoning Dataset

Cosimo is a **synthetic, code-verified** financial-exam question dataset for
training reasoning models and preference-tuned (DPO/ORPO) models. It contains
**71,000** original, numerically-grounded questions spanning the CFA Level I–III
and FRM Part 1/2 curricula, each with a step-by-step chain-of-thought reasoning
trace.

Every numerical answer is **computed by reference code, never sampled from a
language model**. Reasoning traces are *derived from* the computed
intermediates, so they are numerically consistent by construction. About 35% of
records additionally carry a **preference pair** — a verified strong trace
(`chosen`) versus a flawed trace committing exactly one targeted pitfall error
(`rejected`) — ready for DPO/ORPO training.

This dataset was built for
[**Cosimo**](https://github.com/btech-software/cosimo), a project fine-tuning a
compact model (Phi-4-mini-flash, 3.8B) into a financial-reasoning specialist
using [Unsloth](https://github.com/unslothai/unsloth).

## Composition

| Program | Records | Split name |
|---|---|---|
| CFA Level I | 33,000 | `cfa_level_i` |
| CFA Level II | 12,000 | `cfa_level_ii` |
| CFA Level III | 9,000 | `cfa_level_iii` |
| FRM Part 1 | 10,000 | `frm_part_1` |
| FRM Part 2 | 7,000 | `frm_part_2` |
| **Total** | **71,000** | |

Coverage spans 59 topic × subtopic cells across quantitative methods, fixed
income, derivatives, equity valuation, portfolio management, market/credit/
operational/liquidity risk, economics, FSA, ethics-adjacent performance topics,
and more. Question types: `Calculation`, `Vignette`, `Constructed Response`,
and `MCQ`. Difficulty tiers follow the program level (e.g. `L1_Easy` …
`L3_Hard`, `FRM1_*`, `FRM2_*`).

## Configs

### `default` — full records, one split per program

```python
from datasets import load_dataset

ds = load_dataset("btech-software/cosimo-cfa-frm-71k", "default")
ds["cfa_level_i"][0]
```

Each record:

| Field | Description |
|---|---|
| `id` | `cosimo_<program>_<seq>_<sha>` — content hash of question + verified answer |
| `program` | `CFA_Level_I` … `FRM_Part_2` |
| `topic` / `subtopic` | curriculum taxonomy cell |
| `difficulty` | tiered difficulty label |
| `question_type` | `Calculation`, `Vignette`, `Constructed Response`, `MCQ` |
| `question` | original question text |
| `answer` | correct answer (computed) |
| `distractors` | plausible wrong options (empty for constructed-response) |
| `reasoning_trace` | step-by-step CoT with formulas and explicit assumptions |
| `verified` | `true` — only verified records are shipped |
| `verification` | method, template, seed, recomputation flags |
| `metadata` | pitfalls addressed, generator name/version, seed |
| `preference_pair` | `chosen`/`rejected` traces + `pitfall` (null on ~65% of rows) |

### `preference_pairs` — flattened DPO/ORPO rows

24,711 rows with `prompt`, `chosen` (`{answer, reasoning_trace}`), `rejected`
(`{answer, reasoning_trace}`), and the named `pitfall` the rejected trace
commits (e.g. *"geometric vs arithmetic"*, *"annuity due vs ordinary"*, *"sign
flip"*). The rejected answer is guaranteed numerically different from the
correct answer.

```python
prefs = load_dataset("btech-software/cosimo-cfa-frm-71k", "preference_pairs")

def to_dpo(row):
    return {
        "prompt": row["prompt"],
        "chosen": row["chosen"]["reasoning_trace"],
        "rejected": row["rejected"]["reasoning_trace"],
    }

dpo = prefs["train"].map(to_dpo, remove_columns=prefs["train"].column_names)
```

## Integrity guarantees

The full corpus passes a 4-axis verification gate (100% on all axes at release):

1. **Answers are computed, not guessed.** Every template computes its result
   numerically; the verification gate re-runs the template from the stored seed
   and compares the recomputed answer to the persisted one.
2. **Traces are derived from computed numbers.** Trace text references the
   already-computed intermediates and is byte-identical under deterministic
   recomputation.
3. **Concrete preference pairs.** Every `rejected` answer is verified to differ
   numerically from the correct answer.
4. **Clean distractors.** No distractor numerically equals the correct answer.

Generation is deterministic per `(program, template, variant)` with
content-hashed IDs, so every record is independently reproducible from its
stored seed.

## Limitations

- **Structural novelty is bounded by 71 distinct question stems** (templates);
  within a stem, records differ in sampled numbers, entities, and phrasing.
  Deduplicate by `metadata.generator` if you need stem-level splits.
- Content is synthetic exam-*style* material aligned to public learning
  objectives; it is not a substitute for official curriculum or mock exams.
- English only.

## Provenance and trademarks

All questions are **original synthetic content** generated from independently
written templates inspired only by publicly available learning outcome
statements. **No proprietary CFA Institute or GARP exam items were used.**
CFA® is a registered trademark of CFA Institute; FRM® is a registered
trademark of the Global Association of Risk Professionals (GARP). This dataset
is not affiliated with, endorsed by, or sponsored by CFA Institute or GARP.

## License

MIT. Attribution appreciated:

```bibtex
@misc{cosimo2026,
  title  = {Cosimo: A Synthetic, Code-Verified CFA/FRM Financial Reasoning Dataset},
  author = {Sant'Anna, Bruno},
  year   = {2026},
  url    = {https://huggingface.co/datasets/btech-software/cosimo-cfa-frm-71k}
}
```
