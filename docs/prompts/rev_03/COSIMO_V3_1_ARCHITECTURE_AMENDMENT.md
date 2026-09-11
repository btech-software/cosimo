# Cosimo v3.1 — architecture amendment against current `main`

This amends Part III. It does not add a v4 backend. The target is still
`btech-software/cosimo-quant-assistant-v3` distilled into `Qwen/Qwen3.8-27B`.

What `main` already got right and must keep:

- five fact computers, inventory, CLI verbs, Airflow shells the same verbs
- invented-number + must-mention + forbidden-claim repair loop
- student `Qwen/Qwen3.8-27B` QLoRA, `mix: []`, family holdout in prepare
- publish refuses without `gold_bar_v3.jsonl`
- default teacher routing in `config.py` (DeepSeek reasoning / Qwen prose)

What `main` got wrong, and this amendment exists to fix:

- live unbounded render is the operator default (`dataset_build.sh`)
- analysis/memo/grounded have `think=True`
- SFT rows are teacher transcripts (`messages` includes the teacher system)
- `fact_pack` / `verified` are not first-class on the shard row
- register is a label, not a gate
- packs allow two roundings of the same quantity
- `01_prepare_data` can train the labeling protocol

---

## A. Two surfaces (the load-bearing change)

Every renderer writes **two objects**. Only one is trainable.

```
Teacher log  (debug, gitignored, optional)
  shards/v3/teacher_logs/{kind}/{id}.json
  full messages, think text if any, usage, repair history

Student row  (the only thing verify/publish/prepare see)
  shards/v3/sft/{kind}/*.jsonl
```

Student row schema, required:

```
id                  cosimov3_{kind}_{pack_seed:016x}
record_type
work_type
scenario_id         {work_type}.{family}
family              explicit, not parsed after the fact
holdout             bool copied from inventory
variant
register
question            pack.question only — never the teacher brief
answer              visible assistant text, stripped of think + leading blanks
fact_pack           FactPack.to_dict()  (JSON object, not only a string)
verified            true
verification        {computed_by, pack_seed, teacher.model, attempts,
                     invented_numbers: [], missing_mentions: [],
                     register_ok: true}
```

Forbidden on the student row:

- teacher system prompt
- `allowed_numbers` pasted into `question`
- think / reasoning blocks
- `role: system` anything from the factory

`COSIMO_V3_KEEP_TEACHER_MESSAGES=1` writes the log file. Default off.
`01_prepare_data.py` reads **only** student rows. If `messages` is present and
contains `"Cosimo v3 teacher"`, prepare drops the row and increments
`dropped_teacher_leak`.

`write.py` grows `path_for("teacher_logs", ...)` and `path_for("sft", ...)`.
`prose.py` / `agentic.py` / `prefer.py` stop stuffing `messages=exchange`.

---

## B. Think policy (routing.py)

Current `_LANES` on `main`:

```
exam, implementation, agentic, critique  → reasoning, think=True
abstention                               → reasoning, think=False
analysis, memo, grounded                 → prose,     think=True   # WRONG
```

Amended:

```
exam, critique, implementation           → reasoning, think=True
agentic                                  → reasoning, think=True
                                           only on the first planner turn;
                                           tool-call turns think=False
abstention, analysis, grounded           → prose,     think=False
memo                                     → prose,     think=False
                                           think=True only if
                                           COSIMO_V3_MEMO_THINK=1
preference rejected                      → same lane, think=False
```

`dataset_build.sh` currently forces both lanes to `qwen3.8-flash-next`.
Leave `config.py` defaults alone (DeepSeek reasoning / Qwen prose). The
script must stop overriding `TEACHER_REASONING` unless a bake-off is on.

Cap `max_tokens` for think-off lanes at 800. Cap think-on at 2048 until a
row type proves it needs more. Delete the 900s timeout. `config.py`
default 120s is the ceiling; live-slice that exceeds it is a brief bug.

Amendment test: 20 analysis rows think-off vs think-on. Promote think-on
only if invented-number rate drops by ≥ 2 points. Otherwise it stays off.

---

## C. Register is a gate, not a field

Add `dataset/pipelines/v3/verification/register.py`.

| register        | required shape                         | forbidden shape              |
|-----------------|----------------------------------------|------------------------------|
| desk_chat       | ≤ 12 sentences, no section headings    | Finding:/Evidence:/Call:     |
| ic_memo         | Finding + Evidence + Call allowed      | `FINAL ANSWER:`              |
| risk_committee  | limit / horizon / assumption language  | trade recommendation as call |
| exam            | `FINAL ANSWER:` last line              | memo headings                |

`gate_violations()` already used by the prose repair loop grows this axis.
A desk_chat row with `Finding:` is a repair, then a dead letter, not an
SFT row. That is the collapse detector you can run at generation time
instead of waiting for `09_assistant_eval`.

Pack computers assign register from a table in `work_types.yaml`, not
from the renderer. One family may emit more than one register; it may
not emit desk_chat that looks like ic_memo.

---

## D. Pack contract: one official number per quantity

`packs/base.py` `assemble_numbers` today dumps every rounding twin into
`allowed_numbers` so 373 and 372.6 both pass. Stop.

Each computer declares:

```
canonical: { "active_bp": 372.6, "port_ret_pct": 3.31, ... }
aliases:   { "active_bp": ["373"] }   # question prose only, not answer
display:   { "shares": "430,567", "arrival": "296.61" }
```

`invented_numbers` checks answers against `canonical` (+ tiny whitelist
0–10, 100, 252). Questions may use `aliases`. Answers that emit an alias
when a canonical exists fail with `tag=rounding_drift`.

Integers in answers may not carry `.0`. That is a format gate in
`verification/prose.py`, one regex, applied before the teacher repair
so the repair request can say “write 430,567 not 430567.0”.

Judgement the numeric gate cannot see (EV quoted as a price) stays a
human gold-bar problem. Do not pretend a filter will catch it. Catch it
by reading 100 and by putting those misses into `gold_bar_v3.jsonl`.

---

## E. Inventory and holdout

`inventory.expand_jobs` already has `holdout` on `Job`. Amendment:

- `packs` computes **all** families, including holdout, into
  `shards/v3/fact_packs/`.
- `render` / `prefer` write holdout families to
  `shards/v3/eval/{kind}/`, never `sft/`.
- `01_prepare_data.py` drops any sft row whose `holdout` is true
  (`dropped_holdout_leak`). This is defense in depth.

`max_share: 0.03` stays a planning cap. With five work types it cannot
be 3% *each* of a 3k pool and also sum to 100%. Read it as: **no
scenario family > 3% of SFT**, which with ~10 train families is
automatically ~10% each today. Either add work types before scaling
toward 45k, or change the constant to `FAMILY_MAX_SHARE = 1.25 / n_train_families`
and stop advertising 0.03 as if 15 families existed.

Do not emit 120 analysis variants of one family until 20 have been
read. Inventory grows with `variants_per_family` in YAML; WIP YAML
should ship `analysis: 20` not `analysis: 120`. Full counts are a
publish-bar edit, not the working tree.

---

## F. Control plane: script and Make are modes, not a firehose

`dataset_build.sh` is the operator entry. It must not export
`COSIMO_V3_LIVE=1` at the top.

Modes: `smoke | packs | slice | live-slice | prefer-slice | full`.
`full` requires `CONFIRM_FULL=1` and refuses if:

- `gold_bar_v3.jsonl` missing
- last live-slice invented-number rate > 0 on the fixture+live sample
- any student row in the last slice contains `"v3 teacher"`
- think-off analysis has not been measured

Makefile gains the flags the CLI already has:

```
TYPES LIMIT WORK LIVE OUT QUICK
```

Airflow stays a BashOperator over the same CLI. Mapped tasks already
pass `--types` and `--work-type`. The DAG default for render must be
fixture unless `COSIMO_V3_LIVE=1` is set on the Airflow Variable, not
in the repo script.

Stage order for any live path:

```
test → smoke → inventory → packs → render(limit, types) → verify --quick
                                                      ↘ human read
prefer is not on the happy path until register_ok is boring.
publish is last and still needs the gold bar.
```

Verify before prefer. That is the amendment to spec §7.

---

## G. Harness: prepare must not train the factory

`jobs/fine-tune/cosimo_ft/data_schema.py` / `01_prepare_data.py`:

1. Build SFT from `question` + `answer` + student identity in
   `configs/base.yaml`. Ignore `messages` except for `agentic`, and
   even then drop role=system and any content matching the teacher
   brief fingerprint.
2. Drop rows with `verified != true`.
3. Drop rows with `holdout == true`.
4. Cap exam share in the *prepared* mix at `EXAM_SHARE_BAND`, even if
   the shard tree is exam-heavy from an old run.
5. Persist `fact_pack` into the eval sidecar, not into the chat turns.
6. Pin `model.revision` to a SHA. `null` on `main` is still open.

Chat-template policy (unchanged intent, now enforceable):

| record_type                         | think in *target* |
|-------------------------------------|-------------------|
| exam, critique, implementation      | short think ok    |
| analysis, memo, grounded, abstention| no                |
| agentic planner turn                | optional          |
| agentic tool-call turn              | no                |

`09_assistant_eval.py` grows, in this order:

- `invented_number_rate` against eval fact packs
- `register_match`
- `teacher_leak_rate` (factory fingerprint in generations)
- existing `exam_shape_rate` / `mean_new_tokens`

The promotion bar from “WIP slice” to “raise LIMIT” is those four on
a 200-row LoRA smoke, not a green `v3-verify` on teacher logs.

---

## H. Gold bar and example artifacts

- `dataset/goldbar/gold_bar_v3.jsonl` — 100 human-read rows, disjoint
  ids, mix of registers and work types. Publish stays red without it.
- `dataset/pipelines/v3/example_generation.jsonl` is replaced by
  `examples/v3/{analysis,memo,critique,grounded,abstention,exam,agentic,implementation}.jsonl`
  — one student row each, no teacher `messages`. The current 9-row
  teacher log moves to `examples/v3/_teacher_logs/` or is deleted.

---

## I. What does not change

- Two subsystems, joined on Hub, no `dataset` ↔ `jobs` imports.
- Shared tool schemas live in `cosimo.tools`.
- Student stays Qwen3.8-27B, Apache 2.0, Spark QLoRA.
- Teacher stays OpenAI-v1. Default routing in `config.py` is already
  the right pair; stop the shell script from flattening it.
- v1/v2 `generate.py` stays frozen.
- Atomic shards, seed from `(work_type, family, record_type, variant)`.

---

## J. Implementation order on current `main` (do not open a v4 folder)

PR-A  student row schema + leak fingerprint in prepare  
      `prose.py`, `write.py`, `data_schema.py`, tests on the 9-row file
      (those rows must *fail* the new prepare gate)

PR-B  think-off for analysis/memo/grounded/abstention  
      `routing.py`, timeout 120s, 20-row bake-off recorded in
      `dataset/progress/v3_think_ablation.md`

PR-C  register gate in `gate_violations`  
      dead-letter the Finding:/Call: desk_chat rows; update briefs

PR-D  canonical numbers + integer format  
      `assemble_numbers`, attribution 373/372.6, TCA `.0`

PR-E  `dataset_build.sh` modes + Makefile `TYPES/LIMIT/WORK/LIVE`  
      delete top-level `COSIMO_V3_LIVE=1`

PR-F  holdout → `eval/` not `sft/`; shrink YAML analysis 120 → 20
      until 100 gold-bar reads exist

PR-G  pin 27B revision; 200-row LoRA smoke; four eval numbers

No `full` live render before PR-G. The 2,810-job plan stays a file on
disk, not a command you run.
