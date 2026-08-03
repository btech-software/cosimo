# Cosimo fine-tuning harness

Post-training for **Cosimo**, a financial domain assistant, on an **NVIDIA DGX Spark**.

The end goal is not a model that passes CFA/FRM exams. It is an assistant to a Head of
Quantitative Asset Management — one that reasons about valuation, risk, market microstructure and
research papers, and is honest about what it does not know. **Exam accuracy is the milestone this
harness measures, not the objective it optimises for.** A run that lifts exam accuracy while
flattening the model into a five-step calculator has failed, and the evaluation section below tells
you how to notice that before you ship.

Base model: `unsloth/Phi-4-mini-reasoning` (3.8 B, bf16).
Corpus: `btech-software/cosimo-cfa-frm-71k` (71 000 synthetic CFA/FRM items, 24 711 preference pairs).
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
| `tests/` | CPU-only unit tests (no GPU, no network, no torch) |
| `run_all.sh` | Thin orchestrator that chains the documented commands |
| `../../docker/fine-tune/` | `Dockerfile`, `build.sh`, `run.sh`, `torch_arch_guard.py` — the only supported environment |

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

**Access.** A Hugging Face account able to read `unsloth/Phi-4-mini-reasoning` and
`btech-software/cosimo-cfa-frm-71k`. Export `HF_TOKEN` on the host before `run.sh` and it is
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

python scripts/08_export_merge.py --run-name dpo                 # -> runs/dpo/merged (bf16)
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

Then delete `runs/` and `data/` and start the real run, because a `--limit 200` split assignment is
not the split assignment of the full corpus.

---

## Expected wall clock and memory

> **These are estimates, not measurements.** No GPU run has been executed against this harness. They
> are order-of-magnitude arithmetic from record counts and token budgets, shown below so you can
> redo them with your own numbers. The authoritative token statistics arrive when you run
> `01_prepare_data.py`: it writes real p50/p95/p99/max token-length percentiles into
> `data/processed/split_manifest.json`. Check those before trusting anything in this table.

| Step | Estimated wall clock | Estimated peak resident memory |
| --- | --- | --- |
| `00_check_env.py` | < 1 min | negligible |
| `01_prepare_data.py` | 15–40 min (download + render + tokenize 71 k rows) | a few GB host |
| `03_baseline_eval.py` (default suites) | 2–4 h | ~10–15 GB |
| `04_train_sft.py --dry-run` | 3–8 min | ~15–25 GB |
| `04_train_sft.py` (1 epoch) | **10–16 h** | ~15–30 GB |
| `06_evaluate.py` (default suites) | 2–4 h | ~10–15 GB |
| `05_train_dpo.py` (1 epoch) | **11–18 h** | ~20–35 GB |
| `07_compare.py` | seconds | negligible |
| `08_export_merge.py` | 5–15 min | ~20 GB (+15 GB written) |

The arithmetic, so you can disagree with it:

* **Corpus.** 71 000 rows. Six held-out stem families remove ~6 000 (~8.5 %). `val_frac` and
  `test_frac` are 1 % each of the remainder, ~650 rows each. **SFT train ≈ 63 500 rows.**
  Preference pairs: 24 711, of which the ones whose `id` landed in val/test/`unseen_stems` are
  dropped — **DPO train ≈ 22 000 pairs.**
* **Tokens per example ≈ 1 200.** identity ~600 + exam protocol ~45 + question 50–500 +
  completion 150–400. See [The system prompt](#the-system-prompt-is-two-blocks) — the identity block
  roughly doubles this versus a bare instruction.
* **SFT steps.** 63 500 / (4 × 8) = **~1 990 optimizer steps** at effective batch 32.
  76 M tokens/epoch × ~6 FLOP-multiples × 3.8 B params ≈ 1.7 × 10¹⁸ FLOP. Assuming ~125 TFLOP/s
  dense bf16 and 25–40 % MFU (≈ 31–50 TFLOP/s), that is 10–16 h. The MFU assumption is the weakest
  link here.
* **DPO steps.** 22 000 / (2 × 8) = **~1 375 steps** at effective batch 16. Each step forwards four
  sequences (chosen/rejected × policy/reference), so despite being a third of the row count it is
  *more* expensive per epoch than SFT.
* **Evaluation.** Default suites are ~650 (`cosimo_test`) + **~6 000** (`cosimo_unseen_stems`) +
  250 (GSM8K) + 250 (MATH-500) ≈ **7 150 items** at up to 768 new tokens each. Decoding is
  memory-bandwidth bound (7.6 GB of bf16 weights read per step against ~273 GB/s), so batching helps
  but does not rescue you. **`cosimo_unseen_stems` dominates**: `eval.samples.cosimo_unseen_stems`
  defaults to `null` (all of them). While iterating, cut it:

  ```bash
  python scripts/06_evaluate.py --run-name sft --adapter runs/sft/adapter \
      --set eval.samples.cosimo_unseen_stems=500
  ```

  Do the final comparison at full size, and use the *same* setting for every run you compare —
  `07_compare.py` compares only the intersection of item sets and warns when they differ.

Memory is not the binding constraint on 128 GB; **host page-cache pressure is**. See
[Troubleshooting](#out-of-memory-on-a-128-gb-machine).

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

```bash
vllm serve runs/dpo/merged \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --max-model-len 8192
```

`--tool-call-parser hermes` is **required**. Without it vLLM returns the raw `<tool_call>` text as
message content and LangGraph never sees a tool call, so the ReAct loop terminates on the first
step with the JSON as its answer.

### What the model was actually trained on

`scripts/02_prepare_tool_data.py` writes ~5 000 synthetic tool rows (`tools.train_records`) mixed
into the SFT corpus. They teach the **format**, not a tool set: eight tool families with several
name variants each, 2–5 schemas per example resampled per row, and a 20% `tools.no_call_rate`
fraction where none of the offered tools fit and the correct move is to answer directly. Without
that last group the model calls a tool for every question it is ever asked.

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
| `dataset.hub_id` / `.revision` | `btech-software/cosimo-cfa-frm-71k` / `main` | Pin a SHA for a frozen corpus. |
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
| `data.val_frac` / `data.test_frac` | `0.01` each | ~650 rows each; enough for a stable eval-loss curve and a usable Wilson interval, small enough to keep evaluation affordable. |
| `data.max_train_records` | `null` | Cap on SFT training rows (seeded subsample). The knob to reach for when wall clock is the problem. Does not affect DPO. |
| `data.holdout_families` | six families | Excluded from **all** training; see [`unseen_stems`](#why-unseen_stems-exists). |
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

`suites` (the four below), `samples` per suite (`null` = all), `max_new_tokens: 768`,
`temperature: 0.0` (greedy — a base-vs-tuned delta must be a model difference, not sampling noise),
`top_p: 1.0`, `batch_size: 16` (lower this first on OOM), `rel_tol: 1.0e-3`.

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
| `cosimo_test` | Held-out IID slice (~650 items). The headline in-domain number. |
| `cosimo_unseen_stems` | Six stem families excluded from all training (~6 000 items). Generalisation to question structures never trained on. |
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

The corpus is generated from only **71 stems**. A random train/test split therefore leaks: the same
generator, the same formula, the same phrasing skeleton appears on both sides, and in-domain test
accuracy measures "did it memorise these 71 templates" as much as "can it do the finance".

So six stem **families** are excluded from training entirely — `fi_modified_duration`,
`port_sharpe_treynor`, `deriv_bsm_call`, `attr_carino`, `mkt_cvar`, `credit_el`, chosen to span all
five programs — and reported as their own metric. Holding out by *family* matters: seven generators
are `v_` (vignette), `cr_` (constructed response) or `m_` (MCQ) wrappers over a base stem, and
excluding only the wrapper would leak the identical question structure straight back into training.

**Read the two numbers like this.** `cosimo_test` accuracy is optimistic — treat it as an upper
bound that includes template familiarity. `cosimo_unseen_stems` accuracy is the honest one for
"question types it has never seen". A large gap between them is not a bug; it is the measurement
working. A gap that *widens* over training stages means you are buying in-distribution accuracy with
memorisation.

### Style collapse: check for it before you ship

Training on terse exam traces compresses response style. The model learns that answers are five
steps and a `FINAL ANSWER:` line — which is correct for exams and wrong for the job.

GSM8K and MATH-500 detect regression in **general reasoning**. They do **not** measure financial
assistant quality, and nothing in this harness does. Before shipping any checkpoint:

1. Ask the merged checkpoint and the base model the same **open-ended** financial questions — "walk
   me through how you'd hedge a convexity mismatch in this book", "what breaks in this factor model
   during a liquidity crisis", "read this paper's model and implement it" — and compare side by side.
2. Watch `mean_new_tokens` across runs in `metrics.json`. A sharp drop with flat accuracy is style
   collapse, not efficiency.
3. If the tuned model has become a calculator, prefer the SFT checkpoint over the DPO one, reduce
   epochs or `data.max_train_records`, or mix in non-exam data before another attempt.

---

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
* **Pin the moving parts.** `model.revision` and `dataset.revision` default to `null`/`main`. For a
  result you intend to defend months from now, set both to commit SHAs.
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

---

## Known limitations

1. **Only 71 generator stems bound the novelty of the corpus.** 71 000 rows are 71 templates with
   randomised numbers. In-distribution test accuracy is therefore optimistic;
   `cosimo_unseen_stems` is the number to quote, and even that is six families wide.
2. **The data is synthetic.** It was generated and verified by recomputation, not written by CFA/FRM
   charterholders. It teaches calculation procedure, not judgement, and it inherits whatever biases
   the generators have.
3. **MCQ items contain duplicate option values.** The grader accepts a numeric match against the
   gold value, so when both `A. 20` and `B. 20` exist, answering either is scored correct. That is
   the right call for correctness but it makes `distractor_rate` on those items noisier.
4. **Known grading edge case.** An MCQ prediction beginning with a lower-case letter followed by a
   space (`"a portfolio worth 51"`) is currently read as option A. There is a strict-`xfail` test in
   `tests/test_grading.py` pinning the intended behaviour; remove the marker when it is fixed.
5. **GSM8K and MATH-500 measure regression, not finance skill.** They tell you whether general
   reasoning survived. They say nothing about whether the model is still a good assistant.
6. **Nothing here measures assistant quality.** There is no open-ended eval, no rubric, no LLM
   judge, no human comparison. The manual spot-check described under
   [Style collapse](#style-collapse-check-for-it-before-you-ship) is currently the only defence, and
   it is a weak one.
7. **All wall-clock and memory figures in this document are estimates**, derived from record counts
   and token budgets, not from a run. No GPU execution has been performed against this harness.
8. **One epoch, one seed, no sweep.** The hyperparameters are defensible defaults with reasons
   attached, not tuned values. Nobody has run this twice.
9. **The DPO preference pairs cover only ~35 % of the corpus** (the rows that carry a
   `preference_pair`), so the preference stage sees a narrower slice of topics than SFT does.
