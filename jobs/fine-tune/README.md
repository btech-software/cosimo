# Cosimo fine-tuning harness

Post-training for **Cosimo**, a financial domain assistant, on an **NVIDIA DGX Spark**.

The end goal is not a model that passes CFA/FRM exams. It is an assistant to a Head of
Quantitative Asset Management — one that reasons about valuation, risk, market microstructure and
research papers, and is honest about what it does not know. **Exam accuracy is the milestone this
harness measures, not the objective it optimises for.** A run that lifts exam accuracy while
flattening the model into a five-step calculator has failed, and the evaluation section below tells
you how to notice that before you ship.

Base model: `unsloth/Phi-4-mini-reasoning` (3.8 B, bf16).
Corpus: **`btech-software/cosimo-quant-reasoning-v2`** (113 574 records across five record types,
7 500 standalone preference pairs), mixed with `btech-software/cosimo-cfa-frm-71k` capped at 12 %
for exam depth. See [The corpus is a mix](#the-corpus-is-a-mix-of-two-published-datasets).
Default pipeline: **LoRA SFT → DPO**, with a complete **ORPO** single-stage alternative.

---

## Contents

| Path | What it is |
| --- | --- |
| `configs/` | Layered YAML: `base.yaml` plus one file per stage |
| `configs/chat_template.jinja` | The chat template the whole harness uses (see [Chat template](#the-chat-template-is-overridden-on-purpose)) |
| `cosimo_ft/` | The library: config, chat rendering, schema, splits, grading, generation, evaluation, reporting |
| `scripts/` | The numbered pipeline, `00` … `08`. `05b_train_orpo.py` is the alternative to stage `05`, not a step of its own (see [ORPO](#the-orpo-alternative-path)) |
| `cosimo_ft/tools.py` | The tool-calling wire format, shared by the chat template and the data generator (see [Tool calling](#tool-calling-and-the-langgraph-flow)) |
| `suites/` | Hand-written assistant-quality prompt sets and the terminology glossary (see [Assistant-quality evaluation](#assistant-quality-evaluation-09_assistant_evalpy)) |
| `tests/` | CPU-only unit tests (no GPU, no network, no torch) |
| `run_all.sh` | Thin orchestrator that chains the documented commands |
| `../../docker/fine-tune/` | `Dockerfile`, `build.sh`, `run.sh`, `torch_arch_guard.py` — the only supported environment |
| `../../docker/serve/run.sh` | Serves a merged checkpoint on an OpenAI-compatible API for manual testing (see [Serving](#serving)) |

Generated and gitignored: `data/` (prepared JSONL + `split_manifest.json`) and `runs/`
(adapters, checkpoints, merged weights, generations, metrics, TensorBoard logs).

---

## Prerequisites

**Hardware.** NVIDIA DGX Spark — GB10 Grace Blackwell, aarch64, compute capability **sm_121**,
**128 GB unified (CPU+GPU shared) memory**, CUDA 13.0. Nothing here is expected to run unchanged on
x86 + discrete GPU; `00_check_env.py` warns loudly when the capability is not `(12, 1)`.

**Disk.** ~40 GB for the image, ~10 GB for the base model in the HF cache, and a few GB per run.

**Software.** Docker with the NVIDIA container runtime (`--gpus all` must work). Nothing else is
installed on the host: **Docker is the only supported path**, there is no host `uv`/`pip` variant.

**Access.** A Hugging Face account able to read `unsloth/Phi-4-mini-reasoning`,
`btech-software/cosimo-quant-reasoning-v2` and `btech-software/cosimo-cfa-frm-71k` — the corpus is
a mix, and `01_prepare_data.py` fails if either dataset is unreadable. Export `HF_TOKEN` on the
host before `run.sh` and it is
forwarded into the container. `WANDB_API_KEY` is optional; when it is set, the training scripts add
`wandb` to their reporters alongside TensorBoard.

**Version pins** live in the repository-root `pyproject.toml`, in the `[dependency-groups]`
`fine-tune` group — **not** in a `requirements.txt`. The Dockerfile extracts that group with stdlib
`tomllib` and installs it, so the group is the single source of truth. `torch` is deliberately
absent from it: it comes from the NGC base image and a PyPI torch would destroy the aarch64 CUDA 13
build (the Dockerfile has a guard that fails the build if `torch.__version__` changes).

Two pins deliberately contradict upstream guidance, and both are correct here:

* **`trl==0.24.0`.** NVIDIA's DGX Spark playbook installs `trl==0.26.1`, which is outside
  unsloth 2026.8.1's supported range (`<=0.24.0`).
* **`bitsandbytes==0.49.2`.** Unsloth's own DGX Spark Dockerfile pins `0.48.0`, but unsloth's own
  metadata *excludes* exactly `0.46.0` and `0.48.0`; the two cannot be co-resolved. `0.49.2`
  satisfies unsloth and has a published `manylinux_2_24_aarch64` wheel.

`unsloth` and `unsloth_zoo` are installed with `--no-deps` and are intentionally outside the locked
group, because their declared dependencies conflict with the NGC-tuned stack.

---

## How to run on DGX Spark

One sequence, in order. Every command is runnable exactly as written; the ones after
`docker/fine-tune/run.sh` run **inside** the container, whose working directory is already
`/workspace/cosimo/jobs/fine-tune`.

```bash
# --- on the DGX Spark host ---------------------------------------------------
git clone git@github.com:btech-software/cosimo.git
cd cosimo

export HF_TOKEN=hf_xxx                       # needed for the model and the dataset

bash docker/fine-tune/build.sh               # builds cosimo-fine-tune:latest (long, first time)
bash docker/fine-tune/run.sh                 # interactive shell in the harness directory

# --- inside the container (/workspace/cosimo/jobs/fine-tune) -----------------
python scripts/00_check_env.py               # GPU, sm_121, pins, unsloth, bnb 4-bit smoke test

python scripts/01_prepare_data.py            # -> data/processed/*.jsonl + split_manifest.json
python scripts/02_prepare_tool_data.py       # -> data/processed/tool_{train,val}.jsonl

python scripts/03_baseline_eval.py           # -> runs/baseline/eval/  (the reference measurement)

python scripts/04_train_sft.py --dry-run     # builds everything, trains nothing. Read its output.
python scripts/04_train_sft.py               # -> runs/sft/adapter

python scripts/06_evaluate.py --run-name sft --adapter runs/sft/adapter

python scripts/05_train_dpo.py --sft-adapter runs/sft/adapter    # -> runs/dpo/adapter

python scripts/06_evaluate.py --run-name dpo --adapter runs/dpo/adapter

python scripts/07_compare.py --runs baseline sft dpo             # -> runs/comparisons/*.md

# Is it still an assistant? Run it for the baseline too — every number below is
# only meaningful as a base-vs-tuned delta.
python scripts/09_assistant_eval.py --run-name baseline
python scripts/09_assistant_eval.py --run-name sft --adapter runs/sft/adapter

python scripts/08_export_merge.py --run-name sft                 # -> runs/sft/merged (bf16)
```

A one-shot command works too, without an interactive shell:

```bash
bash docker/fine-tune/run.sh python scripts/00_check_env.py
```

### Do not skip step `04_train_sft.py --dry-run`

It loads the data, builds the model, attaches LoRA, constructs the trainer and applies
response-only masking — then stops before `.train()`. It prints the first rendered example, the
resolved LoRA target modules, the trainable-parameter count, and a **masking report** that asserts
the supervised span is non-empty, contains `FINAL ANSWER:`, and does **not** contain the question.
That is the check that catches chat-template drift before you spend a day of GPU time on a run that
was training on the wrong tokens.

### `run_all.sh`

The same sequence, chained, echoing each command before it runs:

```bash
./run_all.sh --dry-run       # print the plan, run nothing
./run_all.sh                 # full pipeline
./run_all.sh --eval-only     # evaluate existing checkpoints and compare; never trains
./run_all.sh --limit 20 --suites "cosimo_test gsm8k"   # fast smoke pass
```

It refuses to overwrite an existing `runs/baseline` (skips the step and says so; `--force-baseline`
overrides), and `--eval-only` never invokes a training or export script. It orchestrates only — it
reimplements nothing, so anything it does you can also do by hand from the list above.

### Smoke-testing the whole pipeline first

Everything below assumes you want the real thing. To prove the plumbing on a fresh machine in
minutes rather than days:

```bash
python scripts/01_prepare_data.py --limit 200 --force
python scripts/02_prepare_tool_data.py --force \
    --set tools.train_records=200 --set tools.val_records=20
python scripts/03_baseline_eval.py --suites cosimo_test --limit 20
python scripts/04_train_sft.py --run-name sft_smoke --set sft.max_steps=20
python scripts/06_evaluate.py --run-name sft_smoke --adapter runs/sft_smoke/adapter \
    --suites cosimo_test --limit 20
python scripts/07_compare.py --runs baseline sft_smoke --suite cosimo_test
```

`--limit` takes a seeded sample of N rows **per split of each source**, not a head slice, so a
smoke run still sees every program, all five record types and the holdout families. It is 2 000
rows here, not 200: five splits × two corpora.

Then delete `runs/` and `data/` and start the real run, because a `--limit 200` split assignment is
not the split assignment of the full corpus.

---

## Expected wall clock and memory

> **The training rows below were measured on the v1-only corpus** (DGX Spark, 2026-08-03/04) and are
> now **lower bounds**: the mixed corpus is 2.4× the SFT rows, so scale accordingly — see the
> arithmetic. Evaluation figures were always estimates. The authoritative token statistics arrive
> when you run `01_prepare_data.py`: it writes real p50/p95/p99/max token-length percentiles into
> `data/processed/split_manifest.json`.

| Step | Wall clock | Estimated peak resident memory |
| --- | --- | --- |
| `00_check_env.py` | < 1 min | negligible |
| `01_prepare_data.py` | 30–70 min (download + render + tokenize ~135 k rows from two Hub repos) | a few GB host |
| `03_baseline_eval.py` (default suites) | 2–4 h *(estimate; longer now, see below)* | ~10–15 GB |
| `04_train_sft.py --dry-run` | 3–8 min | ~15–25 GB |
| `04_train_sft.py` (1 epoch) | **~12 h estimated** on the mixed corpus (6.3 h measured on v1-only: 22 574 s, 2 147 steps) | ~15–30 GB |
| `06_evaluate.py` (default suites) | 2–4 h *(estimate)* | ~10–15 GB |
| `05_train_dpo.py` (1 epoch) | **~1.7 h estimated** on 7 425 pairs (5.1 h measured on 22 048: 18 396 s, 1 378 steps) | ~20–35 GB |
| `07_compare.py` | seconds | negligible |
| `08_export_merge.py` | 5–15 min | ~20 GB (+15 GB written) |

Both training stages came in well under the original estimate, which assumed 25–40 % MFU; the
measured SFT run did 3.04 samples/s and 9.37 × 10¹⁷ FLOP. **Evaluation now costs more than the
table says**: `eval.max_new_tokens` went from 768 to 2048 (see below), and the base model actually
uses that budget. Batches also shrink to stay inside `eval.max_batch_tokens`, so throughput drops
with the larger budget. Budget generously for `03_baseline_eval.py` in particular.

**SFT wall clock is the number that moved most, and it moved against you.** The corpus went from
63 500 SFT rows to **125 939**, so a single epoch is now ~12 h rather than 6.3 h at the same
throughput, and non-exam targets are longer than exam ones (an analysis answer is ~900 tokens
against ~250), which pushes it further. If that is too long, `data.max_train_records` is the knob —
**not** a smaller `dataset.mix` share and not a shorter persona, because either changes what the
model is rather than how long it takes to make.

The arithmetic, so you can disagree with it:

* **Corpus.** 113 574 rows from v2, plus v1 capped at 12 % of the merged *trainable* pool
  (21 211 of its 71 000 kept, held-out records exempt from the cap). Cross-source duplicate
  questions remove 2 683. Six held-out stem families remove 6 910 (~5 %). **SFT train = 125 939
  rows** (measured), of which 29.9 % are exam.
  Preference: v2's 7 500 standalone pairs, minus the ones whose split assignment put them in val —
  **DPO train = 7 425 pairs** (measured).
* **Tokens per example.** p50 **794**, p95 1 689, max 1 829 (measured, mixed corpus). The identity
  block is ~650 of that on every row — see
  [The system prompt](#the-system-prompt-is-two-blocks). Nothing exceeds `max_seq_length: 8192`.
* **SFT steps.** 125 939 at effective batch 32 = **~3 936 optimizer steps**, against 2 147 measured
  for v1-only. At the measured 3.04 samples/s that is ~11.5 h, and longer with the heavier non-exam
  targets.
* **DPO steps.** 7 425 at effective batch 16 = **~464 steps**, against 1 378 measured. Each step
  still forwards four sequences (chosen/rejected × policy/reference), so it is expensive per step,
  but the stage is now a third of its former length.
* **Evaluation.** Default suites are 657 (`cosimo_test`) + **6 910** (`cosimo_unseen_stems`) +
  250 (GSM8K) + 250 (MATH-500) ≈ **8 070 items** at up to 2048 new tokens each. Decoding is
  memory-bandwidth bound (7.6 GB of bf16 weights read per step against ~273 GB/s), so batching helps
  but does not rescue you. **`cosimo_unseen_stems` dominates**, which is why
  `eval.samples.cosimo_unseen_stems` defaults to `1000` rather than the whole ~6 900-item slice.
  To cut it further while iterating:

  ```bash
  python scripts/06_evaluate.py --run-name sft --adapter runs/sft/adapter \
      --set eval.samples.cosimo_unseen_stems=500
  ```

  Do the final comparison at full size, and use the *same* setting for every run you compare —
  `07_compare.py` compares only the intersection of item sets and warns when they differ.

Memory is not the binding constraint on 128 GB; **host page-cache pressure is**. See
[Troubleshooting](#out-of-memory-on-a-128-gb-machine).

---

## The corpus is a mix of two published datasets

`dataset.hub_id` is the primary corpus and each entry of `dataset.mix` adds another, merged into
one pool by `01_prepare_data.py`. The default is v2 uncapped plus v1 at 30 %:

| | `cosimo-quant-reasoning-v2` (primary) | `cosimo-cfa-frm-71k` (mixed, 30 %) |
| --- | --- | --- |
| Published rows | 113 574 supervised, 7 500 pairs | 71 000 supervised, 24 711 pairs |
| Record types | exam, analysis, abstention, agentic, implementation | exam only |
| Nested columns | JSON-encoded strings | Arrow structs |
| Preference config | `preference`, ids **disjoint** (`cosimopref_`) | `preference_pairs`, ids **shared** |
| Generators | 299 | 71 (all also in v2) |
| Used for pairs here | yes | **no** (`preference_config: null`) |
| Trainable rows kept | all | 12 % of the merged trainable pool |

**Why mix at all.** v2 is deliberately majority non-exam — exam is 21 % of it, and each of the six
held-out stem families has only 174 rows against v1's 1 000. v1 supplies the exam depth and a
usable `cosimo_unseen_stems` slice; the cap is what stops it re-creating the exam-only corpus that
flattened the first run. At `max_share: 0.12` exam lands at **30 % of SFT**, so roughly 70 % of
what the model trains on is the material v2 was built to supply. The trade-off against exam depth
is smooth — 0.20 gives 36.6 %, 0.30 gives 44.5 % — and the table is in `configs/base.yaml`.

**`max_share` is a share of the merged *trainable* pool, and held-out records are exempt.** A
held-out family never trains; it is the `unseen_stems` measuring instrument. Subsampling it would
shrink the evaluation slice and widen its confidence interval as a side effect of retuning the
training mix, so the cap does not touch it. The holdout is 6 910 items at every setting.
`data.test_frac` is coupled the same way and for the same reason — see below.

**What preparation produces at the defaults** (measured, full corpus):

| | rows |
| --- | --- |
| `sft_train.jsonl` | 125 939 — analysis 31.9 %, exam 29.9 %, agentic 15.3 %, abstention 12.7 %, implementation 10.2 % |
| `sft_val.jsonl` | 1 279 |
| `eval_cosimo_test.jsonl` | 657 (exam only) |
| `eval_cosimo_unseen_stems.jsonl` | 6 910 (exam only, six families) |
| `pref_train.jsonl` / `pref_val.jsonl` | 7 425 / 75, SFT overlap **0** |

Read `by_record_type` in `split_manifest.json` first. A corpus that has drifted back to
majority-exam is the style collapse of the first run waiting to happen again, and it is visible
there before a GPU-hour is spent.

### `FINAL ANSWER:` belongs to exam rows and nothing else

Each record type renders differently, and this is the load-bearing part of the whole change:

| Record type | Supervised target | System block |
| --- | --- | --- |
| `exam` | reasoning trace + `FINAL ANSWER: <value>` | identity **+ exam protocol** |
| `analysis` | the long-form answer verbatim (~900 tokens) | identity only |
| `abstention` | the calibration response verbatim | identity only |
| `implementation` | fenced ```python``` code, the test block, then the result | identity only |
| `agentic` | the whole conversation from the first assistant turn, tool schemas bound | identity only |

The exam protocol carries the grading contract, so attaching it to an open-ended analysis
instructs the exam format into an answer that is supposed to demonstrate its absence. Two gates
enforce the split: `01_prepare_data.py`'s validation gate checks every written row in both
directions (exam rows must carry the tag, no other type may), and `04_train_sft.py`'s masking check
re-derives the same fact from the tokenized row — the tag appears in the *masked* prompt span if
and only if the row is an exam row, which needs no column the trainer's text-only view has dropped.

Agentic records are rendered through `chat.render_tool_example` with their `tool_schemas` bound as
the template's top-level `tools` variable. Their interior tool results render as `<|user|>` turns,
so `train_on_responses_only` masks them and supervises every assistant turn, with no change to the
masking configuration.

### Two properties of the mix that had to be handled explicitly

* **1 840 exam questions are byte-identical across the two corpora under different ids.** Splits
  are keyed by `id`, so without intervention the same question could sit in v2's test slice and
  v1's training set. Rows whose question text already appeared in a higher-priority source are
  dropped (2 683 rows at the defaults, since v1 repeats some of them internally). Duplicates
  *within* one corpus are left alone — v1 has 15 366 of them and that is a pre-existing property of
  that dataset, not something the mix introduced.
* **7 500 of the published v2's 13 000 `implementation` records ship a `test_code` field that does
  not parse** — a stray-indent `SyntaxError` (`"f = g(x)\n    assert len(f) == 4"`), because the
  generator ran `.strip()` before `textwrap.dedent()` and only the first line lost its indent.
  Fixed at the source in `dataset/pipelines/templates/v2_implementation.py`, but the published
  corpus still carries it, so `normalize_python_block` undoes the damage by re-dedenting the
  continuation lines. The repair runs **only** on a block that fails to parse and is accepted
  **only** if the result parses, so a legitimately indented block can never be mangled; anything
  still unparseable is dropped rather than rendered. Both counts are in the manifest
  (`implementation_test_code_reindented`, `..._unparseable`). All 7 500 currently repair cleanly,
  and the reindented count should fall to 0 once v2 is regenerated from the fixed generator.

### Only exam records are graded

`grading.grade_cosimo` reads the value after the last `FINAL ANSWER:` line and matches it
numerically or by option letter. An analysis or an abstention response has no such value, so
scoring one would count it wrong for every model. Both evaluation slices are therefore exam-only,
and non-exam records are split into `train`/`val` with `test_frac = 0` — they never reach an
evaluation file.

Holdout is still by *family* and still applies to **all** record types: a held-out family's
analysis and agentic records are excluded from training too, or the family leaks straight back in
through its other types. Those non-exam held-out rows are excluded from training and never
evaluated; the manifest reports the count as `holdout_records_not_evaluated` so the gap between
`holdout_records` and the file size is explained rather than looking like rows going missing.

Assistant quality across the four non-exam types is measured by
[`09_assistant_eval.py`](#assistant-quality-evaluation-09_assistant_evalpy), which is the honest
instrument for it.

### The preference stage no longer depends on a holdout

v2's pairs live in the `cosimopref_` id namespace, **disjoint from every supervised id by
construction**, so the SFT/DPO chosen-side overlap that made the first run's preference stage a
zero-gradient no-op is now impossible rather than merely avoided. They are also about judgement
rather than arithmetic — 2 750 `false_confidence`, 1 750 `wrong_assumption`, 1 750
`answers_different_question`, 1 250 `invented_term` — and neither side carries a `FINAL ANSWER:`
line.

`data.preference_holdout_frac` is consequently **inert at the defaults**: it only bites on a
shared-id corpus, and v1's pairs are not used. The machinery, the manifest's `sft_pref_overlap`
field and the validation gate that fails on a non-zero overlap all remain, so setting
`preference_config: preference_pairs` on the v1 entry re-enables the v1 pairs safely.

---

## The system prompt is two blocks

`configs/base.yaml` defines the prompt in two pieces, and the split is deliberate.

* **`prompt.identity`** — the ~2 300-character persona. Present on **every** example, in training
  and at inference. This is what binds "being Cosimo" to the model rather than to a system prompt
  someone might forget to send.
* **`prompt.exam_protocol`** — ~180 characters, appended **only** for exam-format items (Cosimo
  records, GSM8K, MATH-500). It carries the grading contract:
  `Finish with a single final line in exactly this form:\nFINAL ANSWER: <value>`.
* **`prompt.identity_short`** — a one-line identity used on a deterministic
  `prompt.variation_rate` = **15 %** of *training* examples, selected by hashing the record `id`
  (so data preparation is reproducible). It exists so the model does not become brittle to one exact
  string. **Evaluation never uses it**: every evaluated item, for every suite and every model, gets
  the full identity.

The composed system message is `identity + "\n\n" + exam_protocol` — **2 494 characters**, which at
the ~4 chars/token typical of English prose under this tokenizer is **~600–630 tokens on every
example**.

**That cost is real and it is not hidden.** With response-only loss those tokens are unsupervised
*context* — the model is never asked to reproduce them — but they are still forward- and
backward-propagated, so **tokens per step roughly double** versus a bare instruction. DPO pays it
four times per step (chosen/rejected × policy/reference). This is the accepted price of training
under the same prompt the model is served with. If wall clock is the problem, reach for
`data.max_train_records` or `sft.max_steps`, **not** for a shorter persona — a persona trimmed for
throughput is a different model than the one you evaluated.

Why not bind the exam protocol to everything? Because attaching the full persona to terse exam
traces on every item is exactly how you teach a model that being Cosimo *means* answering in five
formulaic steps. The identity is universal; the task block is not.

---

## The chat template is overridden on purpose

The stock `unsloth/Phi-4-mini-reasoning` chat template hardcodes this ahead of every system message:

```
<|system|>Your name is Phi, an AI math expert developed by Microsoft.
```

That directly contradicts the identity being trained. `configs/chat_template.jinja` is a
structurally identical template with that sentence removed — same `<|user|>` / `<|assistant|>` /
`<|end|>` markers, same trailing `eos_token` behaviour, so response-only masking and the
`text == prompt + completion` invariant are unaffected.

It is applied in **every** entry point: data preparation, SFT, DPO, ORPO, evaluation and export.
In particular **the base model is evaluated through the same template as the fine-tuned model** —
comparability comes from both sides seeing an identical prompt surface, not from using the vendor
default. The exported adapter and merged checkpoint carry the template too, and
`08_export_merge.py` reads it back off disk and fails the export if the vendor preamble survived,
so serving matches training.

`chat.template_path: null` falls back to the vendor template. The training, evaluation and export
scripts all refuse to run in that state rather than silently produce incomparable numbers. The
template's SHA-256 is recorded in `split_manifest.json` and in every `metrics.json`, so you can
prove two artifacts were produced through the same prompt surface.

---

## Tool calling and the LangGraph flow

The served target is `cosimo/agents/react_agent/agent.py`, a LangGraph `create_react_agent`. Its
model comes from `agent_lab`'s `get_chat_model`, which for `integration_type: openai_api_v1`
returns a `ChatOpenAI` pointed at an OpenAI-compatible endpoint. So the chain is:

```
create_react_agent -> bind_tools -> ChatOpenAI -> OpenAI `tools=[...]`
                                               -> vLLM
                                               -> tokenizer.apply_chat_template(messages, tools=...)
```

**The server does the tool templating and parsing, not LangChain.** That makes the chat template
shipped inside the merged checkpoint load-bearing, and it is why the template handles three things
the vendor one did not:

| Concern | What the template does |
| --- | --- |
| Tool schemas | Reads the **top-level** `tools` variable. The vendor template read `message['tools']`, a per-message key nothing sets, so schemas were silently dropped and the model answered as though no tools existed. |
| Assistant tool calls | Renders `<tool_call>{"name": …, "arguments": {…}}</tool_call>` — the Hermes format, so vLLM's stock parser reads it back with no custom plugin. |
| Tool results | Renders a `<|user|>` turn wrapping `<tool_response>`. Deliberate: `train_on_responses_only` splits on `<|user|>` / `<|assistant|>`, so every assistant turn in a multi-turn tool conversation stays supervised and every tool result stays masked, with no change to the masking config. |

`cosimo_ft/tools.py` is the single owner of that wire format, and
`tests/test_tools.py::test_rendered_tool_call_matches_the_template` asserts the module and the
template emit byte-identical strings. A training target that differs from the served rendering by
one space teaches a format the runtime cannot parse back, and nothing else would catch it.

### Serving

`../../docker/serve/run.sh` serves a merged checkpoint over an OpenAI-compatible API on
`http://127.0.0.1:8000/v1`, for manual testing. It uses vLLM's own published image from Docker Hub
rather than the training image, and builds nothing.

```bash
bash docker/serve/run.sh                                 # runs/sft/merged on :8000
bash docker/serve/run.sh --run-name dpo --port 8001      # a different run, a different port
bash docker/serve/run.sh -- --gpu-memory-utilization 0.7 # extra args go to vllm serve
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cosimo","messages":[{"role":"user","content":"Compute the Sharpe ratio for a portfolio returning 11% with 14% vol and a 3% risk-free rate."}]}'
```

It binds to loopback and has **no authentication** — it is a testing harness, not a deployment.
On the Spark it needs the `-aarch64` image tag, which is the default; override with
`COSIMO_VLLM_IMAGE` to pin a version instead of tracking `latest`.

The equivalent by hand, if you want to run vLLM yourself:

```bash
vllm serve runs/sft/merged \
    --chat-template runs/sft/merged/chat_template.jinja \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --dtype bfloat16 \
    --max-model-len 8192
```

`--tool-call-parser hermes` is **required**. Without it vLLM returns the raw `<tool_call>` text as
message content and LangGraph never sees a tool call, so the ReAct loop terminates on the first
step with the JSON as its answer. `--dtype bfloat16` is explicit because checkpoints exported
before the `torch_dtype` fix in `08_export_merge.py` carry `"torch_dtype": null`, which `auto`
resolves to float16.

**`--max-model-len 8192`, not the architecture's 131072.** The config declares a 128 K LongRoPE
window, but the LoRA was trained at 8192 and adapts attention (`qkv_proj`, `o_proj`), so
long-context behaviour is untested and nothing in this harness measures it.

**Send the trained persona.** The model was trained with `prompt.identity` on every example, so it
carries the identity in its weights, but the app's default execution prompt
(`cosimo/agents/react_agent/default_execution_system_prompt.txt`) is a generic four-line ReAct
instruction the model has never seen. Serving it that prompt is a needless distribution shift.

### What the model was actually trained on

Two sources, and they do different jobs.

**The corpus supplies 19 503 real `agentic` records** — multi-turn conversations with tool schemas,
Hermes-format calls, tool results (including failures the assistant recovers from) and a final
answer. 15 % of them chain two rounds of calls. These are the realistic trajectories.

**`scripts/02_prepare_tool_data.py` writes 2 000 synthetic tool rows** (`tools.train_records`,
reduced from 5 000 now that the corpus carries the volume) for the two things the corpus records do
*not* teach: 2–5 competing schemas per example, resampled per row — 18 912 of the agentic records
offer a single schema, which teaches "call the tool" rather than "choose the right tool" — and the
20 % `tools.no_call_rate` fraction where none of the offered tools fit and the correct move is to
answer directly. **Every one of the corpus's agentic records calls a tool**, so without that last
group the model calls one for every question it is ever asked.

These rows render with `exam=False`, so they carry the persona but **not** the `FINAL ANSWER:`
protocol — that contract is an exam-grading artefact and has no place in a ReAct answer.

There are deliberately **no tool-calling preference pairs**; see the note at the end of
`configs/dpo.yaml` for why.

The application's own tool surface is still a placeholder (`cosimo/mcp/tools.py`), so
`ReactAgent.get_react_tools()` returns `[]`. Everything downstream of it is wired; bind real tools
there when they exist.

---

## Configuration

Every script takes the same override surface: `--config PATH` (repeatable, layers an extra YAML
file) and `--set dotted.key=value` (repeatable, value parsed as YAML). Merge order is
`base.yaml` → `<stage>.yaml` → `--config` files → `--set`. Unknown `--set` keys are **rejected**, so
a typo costs you a second rather than a training run. The fully resolved config is written to every
run directory as `resolved_config.yaml` and hashed into `metrics.json` as `config_hash`.

```bash
python scripts/04_train_sft.py --set sft.per_device_train_batch_size=2 --set sft.max_steps=200
```

### `configs/base.yaml` — shared by every stage

| Knob | Default | What it does |
| --- | --- | --- |
| `seed` | `3407` | Seeds splitting, subsampling, shuffling, training and generation. |
| `model.base_id` | `unsloth/Phi-4-mini-reasoning` | The base checkpoint. Its projections are **fused** (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`). |
| `model.revision` | `null` | Pin a commit SHA to make training *and* evaluation reproducible against a moving Hub repo. |
| `model.max_seq_length` | `8192` | Sequence budget. Sized for the agentic path, not the exam corpus: a typical exam row is ~1200 tokens, but a tool conversation adds a JSON schema list and a call/result round-trip, and the served ReAct loop stacks more on top. Raise it further if `split_manifest.json` shows meaningful truncation. |
| `model.load_in_4bit` | `false` | bf16 LoRA is the default: 128 GB unified memory makes quantization unnecessary, and bf16 has no quantization error. Toggle for a 4-bit run. |
| `model.dtype` | `bfloat16` | Native on Blackwell; the base checkpoint is already bf16. |
| `dataset.hub_id` / `.revision` | `btech-software/cosimo-quant-reasoning-v2` / `main` | The primary corpus. Pin a SHA for a frozen result; the manifest records the resolved SHA of **every** source. |
| `dataset.preference_config` | `preference` | Which Hub config carries the pairs, or `null` for none. v2 names it `preference` and its ids are disjoint from the supervised rows; v1 names it `preference_pairs` and shares them. |
| `dataset.mix` | v1 at `max_share: 0.12` | Additional corpora merged into the same pool. Same keys plus `max_share`, a ceiling as a fraction of the merged **trainable** pool (`null` = uncapped, held-out records exempt), enforced by a seeded subsample. See [The corpus is a mix](#the-corpus-is-a-mix-of-two-published-datasets). |
| `paths.*` | `data/`, `data/processed/`, `runs/` | Resolved against this directory, never the CWD. |
| `prompt.*` | see above | identity / identity_short / exam_protocol / variation_rate / final_answer_tag. |
| `chat.template_path` | `configs/chat_template.jinja` | The prompt surface. `null` reinstates the vendor identity preamble and is refused by the scripts. |
| `chat.instruction_part` / `.response_part` | `<|user|>` / `<|assistant|>` | The markers `train_on_responses_only` masks on. Tool results render as `<|user|>` turns so they are masked by the same rule. |
| `tools.enabled` | `true` | Whether `02_prepare_tool_data.py` generates rows. `false` writes empty files, which `04_train_sft.py` skips with a log line. |
| `tools.train_records` / `.val_records` | `5000` / `100` | ~7% of the SFT corpus: enough for the format to survive an epoch against 68k rows ending in `FINAL ANSWER:`, small enough not to displace the financial reasoning. |
| `tools.schemas_per_example` | `[2, 5]` | Schemas offered per example. More than one is essential — with a single schema the model learns "call the tool" rather than "choose the right tool". |
| `tools.no_call_rate` | `0.2` | Fraction where no offered tool fits and the model must answer directly. Without it, the model calls a tool for every question. |

### `configs/data.yaml` — `01_prepare_data.py`

| Knob | Default | What it does |
| --- | --- | --- |
| `data.val_frac` | `0.01` | ~1 280 rows: enough for a stable eval-loss curve, small enough to keep mid-training evaluation to minutes. Applies to every record type. |
| `data.test_frac` | `0.017` | 657 rows. Higher than `val_frac` because it is taken from the **exam** records alone (~30 % of the corpus) — the other types have no gradeable answer. At 1 % it would be ~390, widening the headline Wilson interval as a side effect of a training-mix decision. |
| `data.max_train_records` | `null` | Cap on SFT training rows (seeded subsample). The knob to reach for when wall clock is the problem. Does not affect DPO. |
| `data.preference_holdout_frac` | `0.5` | Fraction of preference-carrying records **reserved** for the preference stage: excluded from `sft_*.jsonl`, written to `pref_*.jsonl`. **Inert at the defaults** — it only applies to a corpus whose pairs share ids with its supervised rows, and v2's do not. Kept, with its validation gate, for the case where v1's pairs are re-enabled. |
| `data.holdout_families` | six families | Excluded from **all** training, across **every** record type; see [`unseen_stems`](#why-unseen_stems-exists). |
| `data.drop_unverified` | `true` | Drop rows that failed the generator's own answer-recomputation check. The dropped count is logged and recorded in the manifest. |

### `configs/sft.yaml` — `04_train_sft.py`

`lora.r` = 32, `lora_alpha` = 32 (scale 1.0), `lora_dropout` = 0.0, `bias: none` (the only setting
that stays mergeable for `08_export_merge.py`), `use_rslora: false`,
`use_gradient_checkpointing: unsloth`.

`lora.target_modules: auto` is **required, not a convenience**: this checkpoint's projections are
fused, so the usual seven-module list (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, …)
matches nothing at all. `auto` walks the live module tree and resolves
`["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]`. Unsloth prints a "custom modules … might be
slower" notice for these — expected, not an error.

`per_device_train_batch_size: 4` × `gradient_accumulation_steps: 8` = **effective batch 32**, which
is what `learning_rate: 2.0e-4` was chosen for. Change the micro-batch freely for memory; keep the
product at 32 or retune the LR. `optim: adamw_torch_fused` — pure CUDA, no bitsandbytes in the
default path. `num_train_epochs: 1`: the corpus is synthetic and built from only 71 stems, so a
second epoch memorises stem wording instead of teaching method. Eval and checkpoints every 250
steps, `save_total_limit: 3`.

### `configs/dpo.yaml` — `05_train_dpo.py`

`beta: 0.1`, `loss_type: ["sigmoid"]` (TRL 0.24.0 types this as `list[str]`, not a bare string),
`learning_rate: 5.0e-6` — two orders of magnitude below SFT, because DPO starts from a competent
policy and only needs to shift relative likelihoods. Effective batch 16 (1 × 16).

`max_prompt_length: 6144` deserves a note: TRL truncates an over-long prompt with `keep_end`, which
slices `prompt[-max_prompt_length:]` — it drops the **start** of the prompt, and the start is exactly
where the identity block lives. At the more obvious 768 every long-vignette pair would train against
a persona-less prompt. Check the percentiles in `split_manifest.json` if you change the persona.

The SFT adapter is attached as trainable and `ref_model=None`: with PEFT, TRL uses the
adapter-disabled base model as the implicit reference, so no second copy of the weights is loaded.

### `configs/eval.yaml` — `03_baseline_eval.py` and `06_evaluate.py`

`suites` (the four below), `samples` per suite (`null` = all), `max_new_tokens: 2048`,
`max_batch_tokens: 24576`,
`temperature: 0.0` (greedy — a base-vs-tuned delta must be a model difference, not sampling noise),
`top_p: 1.0`, `batch_size: 16` (lower this first on OOM), `rel_tol: 1.0e-3`.

**`max_new_tokens` was 768 and that was wrong.** The base model is a long chain-of-thought reasoner
and 768 truncated it on 90–97 % of items, grading it wrong for running out of budget rather than for
being wrong. `summarize_suite()` now warns when a suite's `truncation_rate` exceeds 10 %. Read that
warning: an accuracy measured under it is a lower bound, and any delta against a short-form model is
an overstatement of the gain.

---

## Evaluation methodology

**Protocol.** Every model — base and tuned alike — is prompted with the same composed system message
(full identity + exam protocol), through the same chat template, with greedy decoding, and graded by
the same code path (`cosimo_ft/evalrun.py` is the single implementation shared by
`03_baseline_eval.py` and `06_evaluate.py`). The `FINAL ANSWER: <value>` contract is stated in the
system prompt given to the **base** model too, which is what makes the comparison fair.

**Suites.**

| Suite | What it measures |
| --- | --- |
| `cosimo_test` | Held-out IID slice (657 items, **exam records only**). The headline in-domain number. |
| `cosimo_unseen_stems` | Six stem families excluded from all training (6 910 items, **exam records only**). Generalisation to question structures never trained on. |
| `gsm8k` | 250 grade-school math items. **Regression check**, not a target. |
| `math500` | 250 competition math items. **Regression check**, not a target. |

**Metrics** (`runs/<name>/eval/metrics.json`, per suite):

* `accuracy` and `accuracy_ci95` — a Wilson score interval, so small-n differences are not
  over-read.
* `format_compliance` — fraction of generations that actually emitted the `FINAL ANSWER:` tag.
  Reported **separately from accuracy on purpose**: when the tag is missing the grader falls back to
  the last non-empty line, so a right answer in the wrong shape is scored right and shows up as a
  formatting problem rather than a reasoning one. A base model typically has low format compliance
  and a tuned model high; that difference is a formatting gain, not a reasoning gain, and keeping
  them separate is what stops you confusing the two.
* `distractor_rate` — fraction of wrong answers that match one of the record's distractors, i.e.
  the "fell for the pitfall" rate. This is the headline metric for the preference stage: DPO's whole
  job is to push probability mass off the pitfall answers.
* `mean_new_tokens`, `p95_new_tokens`, `truncation_rate` — the response-length distribution. Watch
  these: a large drop in mean length after SFT is the fingerprint of style collapse.
* Breakdowns by program, question type, topic and difficulty.

Per-item generations are streamed to `<suite>_generations.jsonl` as they are produced, so a crash
loses at most one batch, and `--resume` skips ids already on disk.

**Grading** (`cosimo_ft/grading.py`) reads the value after the last `FINAL ANSWER:` line.
Formatting never decides correctness: `$1,234.00`, `1234` and `1,234.0` are the same answer,
`12.5%` equals `12.5`, `(1,234)` is `−1234`. MCQ items are graded by option letter **or** by a
numeric match against the gold value — the corpus contains duplicate option values (both `A. 20` and
`B. 20`), so letter-only matching would score an equally correct answer wrong. Prose gold requires
every number in the gold to appear in the prediction, agreement on a leading `Yes`/`No`, and is not
fooled by negation ("will not increase" does not answer "Increase").

**Comparison.** `07_compare.py` joins runs **per item id** and reports a paired delta with an exact
McNemar p-value — not two independent accuracies subtracted. It compares only the intersection of
item sets, warns when they differ, and warns when two runs were measured under different decoding
settings or config hashes (`--strict` turns those warnings into a non-zero exit).

### Why `unseen_stems` exists

The corpus is generated from templates — **299 generators** across the mix, 71 of them v1's. A
random train/test split therefore leaks: the same generator, the same formula, the same phrasing
skeleton appears on both sides, and in-domain test accuracy measures "did it memorise these
templates" as much as "can it do the finance".

So six stem **families** are excluded from training entirely — `fi_modified_duration`,
`port_sharpe_treynor`, `deriv_bsm_call`, `attr_carino`, `mkt_cvar`, `credit_el`, chosen to span all
five programs — and reported as their own metric. Holding out by *family* matters: some generators
are `v_` (vignette), `cr_` (constructed response) or `m_` (MCQ) wrappers over a base stem, and
excluding only the wrapper would leak the identical question structure straight back into training.
The holdout applies to **every** record type, so a family's analysis and agentic records are kept
out of training too — but only its exam records are evaluated, because the other types cannot be
graded.

**The slice does not move when the training mix does.** Held-out records are exempt from
`dataset.mix`'s `max_share` cap, so it stays 6 910 items (v1's ~1 000 per family plus v2's 174) at
every setting. That is deliberate: a knob that balances what the model learns must not quietly
widen the confidence interval of what measures it. `data.test_frac` is raised to 1.7 % for the same
reason — it applies to exam records only, which are ~30 % of the corpus, so 1 % would have yielded
~390 items instead of 657.

**Read the two numbers like this.** `cosimo_test` accuracy is optimistic — treat it as an upper
bound that includes template familiarity. `cosimo_unseen_stems` accuracy is the honest one for
"question types it has never seen". A large gap between them is not a bug; it is the measurement
working. A gap that *widens* over training stages means you are buying in-distribution accuracy with
memorisation.

### What the first full run showed

One complete `baseline → SFT → DPO` pass has been executed (DGX Spark, 2026-08-03/04). Three
results are worth carrying forward, because two of them are about the harness rather than the model.

**1. The baseline was truncation-bound and its numbers are not usable.** At the old
`max_new_tokens: 768`, baseline `truncation_rate` was 0.895 / 0.929 / 0.524 / 0.968 across
`cosimo_test` / `cosimo_unseen_stems` / `gsm8k` / `math500`. Inspecting the generations shows the
base model solving items correctly and being cut off while writing the answer. Every reported
base-vs-tuned delta from that run is inflated by an unknown amount. The default is now 2048 with a
KV-cache bound (`eval.max_batch_tokens`); the whole run needs re-measuring before any delta is
quoted. 4096 was tried first and OOM-killed the machine — see
[Troubleshooting](#the-machine-swaps-or-the-oom-killer-runs-during-evaluation).

**2. Style collapse is real and large.** `mean_new_tokens` went from ~750 (baseline, against the
cap) to **120** after SFT, p95 179 — roughly a 6× compression, on GSM8K and MATH-500 as well as on
the exam suites. Format compliance went 4 % → 100 %. The model now answers everything in the shape
of a four-step exam trace.

That run trained on an exam-only corpus where every target ended in `FINAL ANSWER:`. **The
`cosimo-quant-reasoning-v2` mix is the direct answer to it**: 70 % of SFT rows are now analysis,
abstention, agentic or implementation records, none of which carry the grading contract or the
exam protocol. Whether that is *enough* is unmeasured — no run has been executed on the mixed
corpus, and `exam_shape_rate` from
[`09_assistant_eval.py`](#assistant-quality-evaluation-09_assistant_evalpy) is the number that will
say.

**3. The DPO stage was a no-op, and structurally so.** Train loss was exactly `0.0` from step 10,
eval loss 1e-9 to 1e-10, and the adapter moved 0.16 % in relative Frobenius norm
(`‖DPO−SFT‖_F / ‖SFT‖_F = 0.0016`). Every suite delta was within noise (McNemar p = 1, 1, 0.5, 1)
and `distractor_rate` did not improve.

The cause was in the data, not the code: the corpus `reasoning_trace` **is** the `chosen` side of
the preference pair, and SFT trained on those same rows — all 22 048 of them, a 100 % overlap. By
the time DPO started, the policy already assigned overwhelming relative likelihood to chosen over
rejected, the implicit reward margin was hundreds of nats, the sigmoid saturated, and the gradient
was zero.

**This is now structural rather than merely avoided.** The preference stage trains on v2's 7 500
standalone pairs, whose `cosimopref_` ids are disjoint from every supervised id **by construction**
— no reservation is needed and no overlap is possible. `01_prepare_data.py` still writes the
overlap into `split_manifest.json` and the validation gate still **fails** if it is non-zero, so
the failure cannot recur silently even if a shared-id corpus is re-enabled.
`data.preference_holdout_frac`, the earlier fix, is inert at the defaults and kept for that case.

Runs produced before this change still have the old behaviour baked in; a re-prepared dataset is
required to benefit, and re-preparing changes the split assignment.

The in-domain / held-out gap was `cosimo_test` 62.9 % vs `cosimo_unseen_stems` 18.0 %. Quote the
second number, as this document has always said.

### Style collapse: check for it before you ship

Training on terse exam traces compresses response style. The model learns that answers are five
steps and a `FINAL ANSWER:` line — which is correct for exams and wrong for the job. **This is not
hypothetical: the first run compressed mean response length from ~750 tokens to 120.**

GSM8K and MATH-500 detect regression in **general reasoning**. They do **not** measure financial
assistant quality — [`09_assistant_eval.py`](#assistant-quality-evaluation-09_assistant_evalpy)
does, and it is the step to read before shipping. Alongside it, still do this by hand:

1. Ask the merged checkpoint and the base model the same **open-ended** financial questions — "walk
   me through how you'd hedge a convexity mismatch in this book", "what breaks in this factor model
   during a liquidity crisis", "read this paper's model and implement it" — and compare side by side.
2. Watch `mean_new_tokens` across runs in `metrics.json`. A sharp drop with flat accuracy is style
   collapse, not efficiency.
3. If the tuned model has become a calculator, prefer the SFT checkpoint over the DPO one, reduce
   epochs or `data.max_train_records`, or mix in non-exam data before another attempt.

---

## Assistant-quality evaluation (`09_assistant_eval.py`)

The exam suites answer *was the number right*. This answers *is it still an assistant* — the gap
that let the first run raise exam accuracy while collapsing mean response length from ~750 tokens
to 120 and answering an open-ended hedging question in `Step 1./Step 2.` form, inventing a
"Durbin-Watson duration" on the way.

```bash
python scripts/09_assistant_eval.py --run-name baseline                          # do this first
python scripts/09_assistant_eval.py --run-name sft --adapter runs/sft/adapter
python scripts/09_assistant_eval.py --run-name sft --merged runs/sft/merged
```

**Every number here is only meaningful as a base-vs-tuned delta.** There is no gold answer and no
absolute standard; a 40 % exam-shape rate means nothing until you know the base model's.

| Suite | Items | What it measures |
| --- | --- | --- |
| `open_ended` | 30 | Real Head-of-Quant questions — hedging, factor breakdowns, execution, paper implementation. Judgement questions with no single number. |
| `calibration` | 20 | Underspecified, unanswerable and false-premise prompts. Does the model ask, or answer anyway. |
| `agentic` | 16 | Mock-tool ReAct trajectories: single-call, **multi-call**, and no-call-appropriate. |

Metrics (`runs/<name>/assistant_eval/metrics.json`):

* **`exam_shape_rate`** — fraction of answers carrying `ASSUMPTIONS:`, a run of three or more
  `Step N.` lines, or `FINAL ANSWER:`. The direct read on style collapse, and the headline number.
  A single enumerated step is normal writing and does not count.
* **`abstention_rate`** — fraction that ask for missing information or decline, measured on the
  *opening* of the response so that committing first and hedging later does not count. The persona
  claims honesty about what it does not know while every supervised target is a confident
  computation; this is where that contradiction shows up.
* **`unknown_terms`** — technical terms absent from the curriculum taxonomy plus `suites/glossary.txt`,
  ranked by frequency. **A triage aid, not a hallucination detector**: the vocabulary is incomplete,
  so a real term it has not heard of is reported exactly like an invented one. Read the list; do not
  threshold on it. It is here because scanning twenty flagged phrases is tractable and reading four
  hundred responses is not.
* **`multi_step_accuracy`** — broken out from overall agentic accuracy because
  `02_prepare_tool_data.py` generates exactly **one** tool round-trip per example. Anything longer is
  extrapolation, and this is the number that says whether chaining generalised.
* **`no_call_precision`** — did it decline to call a tool when none fit.
* **`hallucinated_tool_rate`** — did it invent a tool that was never offered.

The prompts are hand-written and small on purpose: they are meant to be read and argued with, and a
generated suite would inherit the same template bias as the training corpus. `configs/assistant.yaml`
deliberately does **not** append `prompt.exam_protocol` — instructing the `FINAL ANSWER:` contract
into the prompt would manufacture the exact format being measured. The persona is still sent,
because it is sent at serving time.

## The ORPO alternative path

ORPO (`scripts/05b_train_orpo.py`, `configs/orpo.yaml`) folds the supervised NLL term and an
odds-ratio preference term into one loss. It needs no reference model and no preceding SFT stage:
one run, one adapter, straight from the base model.

```bash
python scripts/05b_train_orpo.py --dry-run
python scripts/05b_train_orpo.py --run-name orpo
python scripts/06_evaluate.py --run-name orpo --adapter runs/orpo/adapter
python scripts/07_compare.py --runs baseline sft dpo orpo
```

It is the documented **alternative**, not the default. It trains on the ~22 000 preference pairs —
roughly a third of the SFT corpus — so it sees far less supervised signal than SFT → DPO does. Its
LoRA geometry is deliberately identical to `configs/sft.yaml` (`r`, `alpha`, targets), because
changing it would confound the comparison. TRL 0.24.0 prints a deprecation warning when
`ORPOTrainer` is constructed; it is informational, and `TRL_EXPERIMENTAL_SILENCE=1` suppresses it.

`microsoft/Phi-4-mini-flash-reasoning` comes up as an alternative base model. It is a
`Phi4FlashForCausalLM` SambaY hybrid with custom modelling code and is **not trainable with
Unsloth/PEFT**. Treat it as an inference-only comparison point, never as a training target.

---

## Troubleshooting

### `00_check_env.py` reports a hard failure

It exits 1 only on things that make the rest pointless: no CUDA, a torch build carrying no
architecture a GB10 can run, a Triton kernel that will not run, `import unsloth` failing, or a
version mismatch on transformers / trl / unsloth. Fix those before anything else; the report is
written to `runs/env_check.json` either way.

On the "arch list" check specifically: a GB10 is sm_121, but the torch does **not** need literal
`sm_121` cubins. CUDA guarantees binary compatibility from one minor revision to the next within a
major architecture, so `sm_120` cubins execute on an sm_121 device, with `compute_120` PTX as the
JIT fallback. `nvcr.io/nvidia/pytorch:25.11-py3` ships `sm_80/86/90/100/110/120 + compute_120` and
no literal `sm_121`, and it runs bf16 matmuls, Triton kernels and bitsandbytes 4-bit linears on a
Spark. The check accepts any of `sm_121`, `sm_120`, `compute_121`, `compute_120`.

### The bitsandbytes 4-bit smoke test fails

It is a **warning, not a failure**, and the default path does not use bitsandbytes at all
(`model.load_in_4bit: false`, `optim: adamw_torch_fused`). Version `0.49.2`'s aarch64 wheel *does*
ship sm_121 cubins — verified by parsing the `.nv_fatbin` sections of `libbitsandbytes_cuda130.so`,
which carries cubins for sm_75/80/90/100/110/120/**121** plus sm_121 PTX — so 4-bit is expected to
work on this hardware. The smoke test tells you whether it actually does *on your machine* before you
rely on it. If it fails, stay on bf16; you lose nothing, because 128 GB of unified memory makes
quantizing a 3.8 B model pointless.

### `04_train_sft.py` fails in `attach_lora` with "incompatible version of torchao"

```
ImportError: Found an incompatible version of torchao. Found version 0.14.0+git,
but only versions above 0.16.0 are supported
```

The NGC base image ships torchao 0.14.0+git; `peft==0.20.0` wants >0.16.0 and *raises* rather than
returning False, from a dispatcher PEFT walks for every LoRA target module — so adapter injection
dies even though the default path uses no quantization at all. The Dockerfile removes torchao for
this reason. **Rebuild the image** (`bash docker/fine-tune/build.sh`) if you built it before that
change; the fix is not in an already-built image.

### The machine swaps, or the OOM killer runs during evaluation

Symptoms: the desktop session dies, `dbus`/`pipewire` are killed, and the kernel log shows

```
NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY] from _memdescAllocInternal
Out of memory: Killed process NNNN (python) total-vm:158493688kB anon-rss:1000kB
```

**Read the `anon-rss`.** 158 GB of virtual address space and ~1 MB resident, with swap almost
untouched, means the memory went to **CUDA/NVRM driver allocations** — which on unified memory are
host RAM, pinned and unswappable. The kernel has nothing to page out, so it kills whatever it can
instead of raising a catchable `torch.cuda.OutOfMemoryError`. `free` will not show it as the
Python process's usage.

Two causes, both real:

1. **Something else is holding the GPU.** `docker/serve/run.sh` runs vLLM, which reserves
   `gpu_memory_utilization` (0.9 by default) up front — ~109 GB of the 121 GB the host also lives
   in. **Stop the server before running the pipeline**, or start it with
   `-- --gpu-memory-utilization 0.35`. Check with `docker ps` first.

2. **The generation budget outgrew the batch.** `batch_size` does not bound memory; every sequence
   reserves `prompt + max_new_tokens` of KV cache, so raising `max_new_tokens` at a fixed count
   multiplies the reservation. Measured on this hardware:

   | `batch × (prompt + max_new_tokens)` | Outcome |
   | --- | --- |
   | 16 × (575 + 768) = **21 488** | completed |
   | 16 × (575 + 4096) = **74 736** | exhausted 121 GB, OOM-killed |

   `eval.max_batch_tokens` (default **24 576**) now caps that product. Batches shrink automatically
   as `max_new_tokens` rises, so a larger budget costs wall clock rather than stability, and the
   planned batching is logged before generation starts:

   ```
   2150 prompts in 239 batches (max 9/batch, peak 23607 token slots, budget 24576)
   ```

   Read that line. If `peak` is at the budget and the run still dies, lower `eval.max_batch_tokens`
   — not `batch_size`, which is only the count ceiling.

`docker/fine-tune/run.sh` also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which lets
the caching allocator release segments instead of pinning its high-water mark. Rebuilding is not
required, but you must start the container through `run.sh` for it to apply.

### Out of memory on a 128 GB machine

Host and GPU **share** the 128 GB. Host page cache counts against you, so a training run that fits
on a fresh boot can OOM after a large dataset download has filled the cache. NVIDIA's DGX Spark
troubleshooting recipe, on the **host** (not in the container):

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

Then, in order: lower `sft.per_device_train_batch_size` (and raise
`gradient_accumulation_steps` to keep the effective batch at 32), lower `eval.batch_size`, lower
`model.max_seq_length` if `split_manifest.json` says you can. Close other GPU processes; the harness
assumes it owns the device.

### Hugging Face authentication

`export HF_TOKEN=hf_xxx` on the host **before** `run.sh` — it is forwarded only when set.
The model cache is bind-mounted from `~/.cache/huggingface`, so weights are downloaded once and
survive container restarts. For a fully offline evaluation, skip the Hub-backed suites:

```bash
python scripts/06_evaluate.py --run-name sft --adapter runs/sft/adapter \
    --suites cosimo_test cosimo_unseen_stems
```

### Resuming

* Training: `--resume` on `04_train_sft.py` / `05_train_dpo.py` / `05b_train_orpo.py` resumes from
  the latest checkpoint in `runs/<name>/checkpoints`.
* Evaluation: `--resume` skips item ids already present in the suite's generations file. Without
  `--resume`, an evaluation clears stale generations first so a half-finished older run cannot be
  read as current.
* Baseline: `03_baseline_eval.py` refuses to overwrite `runs/baseline` — pass `--resume` to continue
  it or `--force` to deliberately re-measure it. It is the reference for every comparison; losing it
  invalidates every delta you have already computed.

### "chat.template_path is not set"

Training, evaluation and export all refuse to run with the vendor template, because it would
reintroduce the Microsoft identity preamble and make the numbers incomparable. Restore
`chat.template_path: configs/chat_template.jinja` in `configs/base.yaml`.

### `07_compare.py` warns that runs are not comparable

Different config hashes, decoding settings or item sets between two runs. Re-evaluate the odd one
out with the same `--suites` / `--set eval.*` settings. Use `--strict` in automation so this is an
error rather than a note.

### The Docker build fails with "the … install replaced the NGC torch"

Working as intended: something in the dependency group tried to pull a PyPI torch, which would
destroy the aarch64 CUDA 13 build for a GB10. Do not remove the guard — fix the pin.

The guard is `docker/fine-tune/torch_arch_guard.py`, run after every stage that could disturb
torch. It reads `torch._C._cuda_getArchFlags()` rather than `torch.cuda.get_arch_list()`: the
public wrapper returns `[]` when `torch.cuda.is_available()` is False, and there is no CUDA driver
inside `docker build`, so a `get_arch_list()` check can never pass at build time.

---

## Reproducibility

* **Seeds.** One `seed: 3407` in `configs/base.yaml` drives split assignment, benchmark
  subsampling, dataset shuffling, trainer seeding and generation.
* **Deterministic splits.** `assign_splits` derives a per-stratum RNG from the seed and the stratum
  key, so the assignment is stable even when unrelated strata change size. It is keyed by `id` and
  reused for the preference config, so no question can be in DPO training and in the test set.
* **Deterministic prompts.** The 15 % short-identity variation is chosen by hashing the record `id`,
  not by an RNG draw, so re-preparing the data reproduces the same prompts.
* **Manifests.** `data/processed/split_manifest.json` records counts per split/program/question
  type, held-out families, seed, config hash, dataset revision, tokenizer id, chat-template SHA-256
  and real token-length percentiles. Every run directory gets `resolved_config.yaml`, `env.json`
  (interpreter, package versions, GPU, git commit) and `manifest.json`.
* **Pinned versions.** The `[dependency-groups] fine-tune` group plus the NGC base image tag pin the
  stack; `00_check_env.py` compares what is installed against that group and reports a mismatch
  table.
* **Pin the moving parts.** `model.revision` and every source's `revision` default to `null`/`main`.
  For a result you intend to defend months from now, set all of them to commit SHAs. The manifest's
  `dataset.resolved_shas` records the SHA each source actually resolved to, so a run made against
  `main` can at least be traced afterwards.
* **Not reproducible:** exact floating-point training trajectories across driver/library versions,
  and anything downloaded fresh from the Hub without a pinned revision.

---

## Tests

CPU-only, no GPU, no network, no torch — they exercise the pure-logic modules and the shipped
config/template assets. **`pytest` is not installed in the fine-tuning image**; the suite is meant to
run on a host venv. From the repository root:

```bash
.venv/bin/python -m pytest jobs/fine-tune/tests -q
```

That venv needs only `pytest`, `pyyaml` and (optionally) `jinja2` — the tests that render the real
`configs/chat_template.jinja` skip cleanly when Jinja is absent. Coverage: grading edge cases
(unicode minus, hyphenated ranges, accounting negatives, MCQ letter/numeric/bare-letter paths,
negation in prose gold, distractor matching), the `prompt == text` prefix invariant under the
shipped template, the identity/exam composition and its 15 % variation rate, the absence of
"Microsoft" from every rendered prompt, split determinism and holdout integrity, row normalisation
for all three preference-pair shapes, and the config merge/override surface.

For the mixed corpus specifically: JSON-encoded nested columns resolving to a real generator (the
failure that would silently collapse the split stratification), the per-record-type supervised
targets, the `FINAL ANSWER:` contract landing on exam rows and nowhere else, agentic conversations
rendering with their schemas bound and their tool results masked, the unparseable-test-block guard,
and the standalone preference shape.

---

## Known limitations

1. **299 generator stems bound the novelty of the corpus.** 162 000 rows are 299 templates with
   randomised numbers — better than v1's 71, but the ceiling is still the generator count, not the
   row count. In-distribution test accuracy is therefore optimistic; `cosimo_unseen_stems` is the
   number to quote, and even that is six families and ~5 200 items wide.
2. **The data is synthetic.** It was generated and verified by recomputation, not written by CFA/FRM
   charterholders. It teaches calculation procedure, not judgement, and it inherits whatever biases
   the generators have. v2's long-form analysis and abstention prose is composed from parameterized
   paragraph builders — topically keyed and numerically grounded, but not written by a human either,
   and `abstention` answers in particular share a structural skeleton.
3. **MCQ items contain duplicate option values.** The grader accepts a numeric match against the
   gold value, so when both `A. 20` and `B. 20` exist, answering either is scored correct. That is
   the right call for correctness but it makes `distractor_rate` on those items noisier.
4. **Known grading edge case.** An MCQ prediction beginning with a lower-case letter followed by a
   space (`"a portfolio worth 51"`) is currently read as option A. There is a strict-`xfail` test in
   `tests/test_grading.py` pinning the intended behaviour; remove the marker when it is fixed.
5. **GSM8K and MATH-500 measure regression, not finance skill.** They tell you whether general
   reasoning survived. They say nothing about whether the model is still a good assistant.
6. **Assistant quality is measured behaviourally, not for correctness.**
   [`09_assistant_eval.py`](#assistant-quality-evaluation-09_assistant_evalpy) scores response
   *shape*, abstention, terminology and tool trajectories — all without a gold answer. Nothing
   grades whether an open-ended answer is actually **good**: there is still no rubric, no LLM judge
   and no human comparison. A model can score perfectly on every metric in that script and still be
   wrong about finance.
7. **Memory figures and the evaluation wall clock are still estimates**, and the SFT training
   figure is now an estimate too: the 6.3 h measurement was taken on the v1-only corpus, which is
   half the size of the mixed one.
8. **One epoch, one seed, no sweep.** The hyperparameters are defensible defaults with reasons
   attached, not tuned values. Nobody has run this twice. In particular, `learning_rate: 2.0e-4`
   and `num_train_epochs: 1` were chosen against a 63 500-row corpus and have not been revisited
   for a 125 939-row one.
9. **The preference stage is 7 425 pairs against 125 939 SFT rows** — 5.9 %, down from ~17 %. It
   covers four failure modes and five programs, but it is a much narrower slice of topics than SFT
   sees, and `FRM_Part_1` has only 250 pairs in the published set.
10. **Nothing has been run on the mixed corpus.** The preparation is verified end to end and every
    gate passes, but no baseline, SFT, DPO or assistant evaluation has been executed against it.
    That the corpus is now 70 % non-exam does not prove style collapse is solved — `exam_shape_rate`
    from `09_assistant_eval.py` is the measurement that will say, and it has not been taken.
11. **The tool suite tests format, not judgement.** `agentic` uses mock results and checks that the
    right tool was chosen and its output reached the answer. Nothing checks whether calling that
    tool was the right analytical move. This applies to the corpus's own 19 503 agentic records as
    much as to the suite: their tool results are mocked and their conversations are single-digit
    turns.
12. **The results currently on disk were measured at `max_new_tokens: 768`** on the v1-only corpus,
    and are truncation-bound on the baseline. They need re-measuring at the new 2048 default before
    any delta is quoted, and their training corpus no longer exists in this configuration.
13. **The published v2 corpus predates the `v2_implementation.py` dedent fix.** Its 7 500 broken
    `test_code` blocks are repaired in the harness rather than discarded, which is sound but is a
    workaround: regenerating and republishing v2 is what actually removes it, and until then the
    manifest's `implementation_test_code_reindented` count stays at 7 500.
