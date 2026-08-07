# Cosimo Architecture

This document maps the Cosimo repository — components, data flow, responsibilities
— across its two build subsystems: the synthetic **corpus pipeline** that produces
training data, and the **post-training harness** that consumes it.

For agent behavioural guidelines (including parallel agent loops) see
[`AGENTS.md`](AGENTS.md); for the record schema see
[`dataset/FORMAT.md`](dataset/FORMAT.md); for the dataset overview see
[`dataset/README.md`](dataset/README.md); for harness operations — every command,
config knob, troubleshooting entry and known limitation — see
[`jobs/fine-tune/README.md`](jobs/fine-tune/README.md), which is authoritative and
deliberately not restated here.

**Path convention:** every path below is relative to the **repository root**.

**Counts and measured results are not documented here.** Record, shard, template
and coverage numbers are generated artifacts — read them from
`dataset/progress/progress.md`. Training wall clock, accuracy and token-length
figures live in the harness README and in each run's `metrics.json`. Numbers
written into prose go stale silently.

---

## 0. Subsystems at a glance

| Path | Subsystem | Documented in |
| --- | --- | --- |
| `dataset/` | Synthetic corpus generation + verification | Part I below |
| `jobs/fine-tune/` | Post-training: SFT → DPO/ORPO, evaluation, export | Part II below |
| `docker/` | The only supported runtime environments | §13 |
| `cosimo/` | LangGraph ReAct application — the serving target | *not yet documented* |
| `tests/` | Application tests (`Makefile`: `make test`) | *not yet documented* |

The corpus and the harness are separate programs joined by published artifacts,
not by imports:

```
dataset/                     generate → verify → publish
   │   publish/push_to_hub.py
   ▼
Hugging Face Hub             btech-software/cosimo-quant-reasoning-v2   (primary)
                             btech-software/cosimo-cfa-frm-71k          (mixed, capped)
   │   scripts/01_prepare_data.py   (config: dataset.hub_id + dataset.mix)
   ▼
jobs/fine-tune/              prepare → SFT → DPO/ORPO → evaluate → merge
   │   scripts/08_export_merge.py → runs/<name>/merged  (bf16 + chat template)
   ▼
docker/serve/run.sh          vLLM, OpenAI-compatible API on :8000
   │
   ▼
cosimo/agents/react_agent/   LangGraph create_react_agent (serving target)
```

Four narrower contracts cross the boundary directly, without going through the
Hub — see §14.

---

# Part I — Corpus pipeline (`dataset/`)

Sections 1–8 describe the synthetic dataset pipeline: how records are generated,
verified and published. Everything in this part is scoped to `dataset/`.

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

---

# Part II — Post-training harness (`jobs/fine-tune/`)

Post-training for Cosimo on an NVIDIA DGX Spark: LoRA SFT → DPO by default, with
a single-stage ORPO alternative. Base model `unsloth/Phi-4-mini-reasoning` (3.8 B,
bf16).

The objective is an assistant to a Head of Quantitative Asset Management, not an
exam solver — exam accuracy is the milestone the harness *measures*, not the thing
it optimises for. That distinction drives the whole evaluation design (§12).

**Operational detail lives in [`jobs/fine-tune/README.md`](jobs/fine-tune/README.md)**
— every command, every config knob and its rationale, wall-clock and memory
figures, troubleshooting, and twelve known limitations. This part maps the
components and their contracts; it does not restate the README.

---

## 9. Stage pipeline (`jobs/fine-tune/scripts/`)

Numbered scripts run in order. Each takes the same override surface (§11) and
writes into its own run directory (§10.3).

| Script | Reads | Writes |
| --- | --- | --- |
| `00_check_env.py` | the installed stack | `runs/env_check.json` |
| `01_prepare_data.py` | Hub datasets (`dataset.hub_id` + `dataset.mix`) | `data/processed/{sft,pref}_{train,val}.jsonl`, `eval_cosimo_{test,unseen_stems}.jsonl`, `split_manifest.json` |
| `02_prepare_tool_data.py` | tool families in-script | `data/processed/tool_{train,val}.jsonl` |
| `03_baseline_eval.py` | prepared eval slices, base model | `runs/baseline/eval/` |
| `04_train_sft.py` | `sft_*.jsonl` + `tool_*.jsonl` | `runs/sft/adapter` |
| `05_train_dpo.py` | `pref_*.jsonl`, SFT adapter | `runs/dpo/adapter` |
| `05b_train_orpo.py` | `pref_*.jsonl`, base model | `runs/orpo/adapter` |
| `06_evaluate.py` | an adapter or merged checkpoint | `runs/<name>/eval/` |
| `07_compare.py` | two or more runs' metrics | `runs/comparisons/*.md` |
| `08_export_merge.py` | an adapter | `runs/<name>/merged` (bf16 + chat template) |
| `09_assistant_eval.py` | `suites/*.jsonl` | `runs/<name>/assistant_eval/` |

`05b_train_orpo.py` is the **alternative** to stage `05`, not a step of its own:
ORPO folds the supervised and preference terms into one loss, needs no reference
model and no preceding SFT. Its LoRA geometry deliberately matches `sft.yaml` so
the comparison is not confounded.

`run_all.sh` chains the same commands and reimplements nothing — `--dry-run`,
`--eval-only` (never trains), `--limit`/`--suites` for a smoke pass, and it
refuses to overwrite an existing `runs/baseline` without `--force-baseline`.

Two stage-specific notes worth knowing before reading the code:

- **`04_train_sft.py --dry-run` is a real gate, not a preview.** It builds data,
  model, LoRA and trainer, applies response-only masking, then stops before
  `.train()` — asserting the supervised span is non-empty, contains
  `FINAL ANSWER:`, and excludes the question. It is what catches chat-template
  drift before a day of GPU time is spent training on the wrong tokens.
- **`03_baseline_eval.py` refuses to overwrite `runs/baseline`.** It is the
  reference for every delta; losing it invalidates comparisons already computed.

---

## 10. Harness library (`jobs/fine-tune/cosimo_ft/`)

`__init__.py` deliberately imports no submodules: the pure-logic modules must stay
importable on a CPU-only machine with stdlib + pyyaml, so the tests can run
without torch or a GPU. Import submodules explicitly.

### 10.1 Pure-logic modules (CPU, unit-tested)

- `config.py` — layered YAML merge, `--set` overrides, `config_hash`,
  `harness_path` (resolves against the harness root, never the CWD).
- `data_schema.py` — reconciles **both** published corpus shapes (§14.1) onto one
  `CosimoRecord` and renders `to_sft_row` / `to_pref_row` / `to_eval_row`, with
  the supervised target dispatched on `record_type` (§14.6). Owns
  `stem_family()`, which strips the `v_` / `cr_` / `m_` wrapper prefixes — the
  reason holdout is by *family* rather than by generator (§12.2) — and
  `is_exam()`, the single place the `FINAL ANSWER:` contract is decided.
- `splits.py` — deterministic `assign_splits` into `train`/`val`/`test`/
  `unseen_stems`, with a per-stratum RNG so assignment is stable when unrelated
  strata change size.
- `chat.py` — system-prompt composition (`identity` / `identity_short` /
  `exam_protocol`), chat-template loading, override and SHA-256, prompt rendering,
  and the `id`-hashed 15 % short-identity variation.
- `tools.py` — **the single owner of the tool-calling wire format**: schema,
  call and response rendering plus `parse_tool_calls`.
- `grading.py` — `FINAL ANSWER:` extraction and answer equivalence (currency,
  percent, accounting negatives, MCQ letter *or* numeric, prose gold with negation
  handling).
- `assistant.py` — behavioural metrics: exam-shape detection, abstention,
  unknown-term harvesting, tool-trajectory grading.
- `report.py` — Wilson intervals and the exact McNemar test, implemented from
  their definitions so no scipy/numpy is needed at report time.

### 10.2 GPU-touching modules

- `modeling.py` — model/tokenizer loading, `resolve_target_modules` (the `auto`
  path that discovers this checkpoint's **fused** projections — `qkv_proj`,
  `o_proj`, `gate_up_proj`, `down_proj` — because the conventional seven-module
  LoRA list matches nothing here), `attach_lora`, `model_fingerprint`.
- `generation.py` — batched greedy generation with left padding, length-bucketed
  prompts, and `plan_batches` enforcing the `max_batch_tokens` KV-cache bound.
- `evalrun.py` — **the single evaluation implementation** shared by
  `03_baseline_eval.py` and `06_evaluate.py`, so base and tuned models are
  measured by identical code. `summarize_suite` warns above a 10 % truncation
  rate.
- `benchmarks.py` — suite loading; maps `cosimo_test` / `cosimo_unseen_stems` to
  prepared JSONL and fetches GSM8K / MATH-500 from the Hub.

### 10.3 Run directories (`runlog.py`)

`RunDir` fixes the layout of `runs/<name>/`: `adapter/`, `checkpoints/`,
`merged/`, `eval/`, `tb/`. Alongside it each run records `resolved_config.yaml`,
`env.json` (interpreter, package versions, GPU, git commit) and `manifest.json`.

`data/` and `runs/` are gitignored — everything in them is reproducible from the
pinned configs, the manifest and the seed.

---

## 11. Configuration and reproducibility

Layered YAML under `configs/`, merged in a fixed order:

```
base.yaml → <stage>.yaml → --config FILE (repeatable) → --set dotted.key=value
```

`base.yaml` carries what every stage shares: `seed: 3407`, model id and
`max_seq_length`, the Hub dataset id, `paths.*`, the `prompt.*` block, `tools.*`
generation parameters, and `chat.*`. Stage files (`data`, `sft`, `dpo`, `orpo`,
`eval`, `assistant`) add only their own keys.

Unknown `--set` keys are **rejected**, so a typo costs a second rather than a
training run. The fully resolved config is written to every run directory as
`resolved_config.yaml` and hashed into `metrics.json` as `config_hash`.

Reproducibility rests on four things, all mechanised:

1. **One seed** (`3407`) drives split assignment, subsampling, shuffling, trainer
   seeding and generation.
2. **Deterministic splits**, keyed by record `id` and reused for the preference
   config, so no question can be in DPO training and in the test set.
3. **Deterministic prompts** — the 15 % short-identity variation is chosen by
   hashing the `id`, not by an RNG draw.
4. **Manifests** — `split_manifest.json` records counts per split, held-out
   families, seed, config hash, dataset revision, tokenizer id, chat-template
   SHA-256 and real token-length percentiles.

`model.revision` and `dataset.revision` default to `null`/`main`; pin both to
commit SHAs for a result you intend to defend later.

### 11.1 The system prompt is two blocks

`prompt.identity` (~2 300 chars) is present on **every** example, training and
inference — it binds "being Cosimo" to the weights rather than to a system prompt
someone might forget to send. `prompt.exam_protocol` (~180 chars) is appended
**only** to exam-format items and carries the `FINAL ANSWER:` grading contract.

The split is deliberate and load-bearing: attaching the exam protocol to
everything is precisely how a model learns that being Cosimo *means* answering in
five formulaic steps. The identity is universal; the task block is not.

### 11.2 The chat template is overridden on purpose

The stock `unsloth/Phi-4-mini-reasoning` template hardcodes
`<|system|>Your name is Phi, an AI math expert developed by Microsoft.` ahead of
every system message, contradicting the identity being trained.
`configs/chat_template.jinja` is structurally identical with that sentence
removed, and is applied in **every** entry point — preparation, SFT, DPO, ORPO,
evaluation, export — so the base model is evaluated through the same prompt
surface as the tuned one.

`chat.template_path: null` reinstates the vendor template; training, evaluation
and export all **refuse to run** in that state rather than silently produce
incomparable numbers. `08_export_merge.py` reads the template back off disk and
fails the export if the vendor preamble survived.

---

## 12. Evaluation surfaces

Two distinct measurement systems, answering different questions.

### 12.1 Exam suites (`03_baseline_eval.py`, `06_evaluate.py`)

Was the number right. Four suites: `cosimo_test` (held-out IID slice),
`cosimo_unseen_stems` (families excluded from all training), and `gsm8k` /
`math500` as **regression checks on general reasoning, not targets**.

Every model is prompted identically, decoded greedily, and graded by the same code
path. Metrics per suite include `accuracy` with a Wilson interval,
`format_compliance` (reported *separately* from accuracy, so a right answer in the
wrong shape reads as a formatting problem rather than a reasoning one),
`distractor_rate` (the "fell for the pitfall" rate — the headline number for the
preference stage), and `mean/p95_new_tokens` + `truncation_rate`.

`07_compare.py` joins runs **per item id** and reports a paired delta with an
exact McNemar p-value — not two independent accuracies subtracted. It compares
only the intersection of item sets and warns when runs differ in decoding settings
or config hash (`--strict` makes that an error).

### 12.2 Why `unseen_stems` exists

A random split leaks: the same generator, formula and phrasing skeleton appear on
both sides, so in-domain accuracy measures template memorisation as much as
finance. Six stem *families* spanning all five programs are therefore excluded
from training entirely and reported separately.

Holding out by **family** is the point — `v_` / `cr_` / `m_` wrappers over a base
stem would otherwise leak the identical question structure straight back into
training. Read `cosimo_test` as an upper bound and `cosimo_unseen_stems` as the
honest number; a gap that *widens* over training stages means in-distribution
accuracy is being bought with memorisation.

### 12.3 Assistant quality (`09_assistant_eval.py`)

Is it still an assistant. Three hand-written suites — `open_ended` (30),
`calibration` (20, underspecified/unanswerable/false-premise), `agentic` (16 mock
ReAct trajectories including multi-call and no-call-appropriate).

Metrics: `exam_shape_rate` (the direct read on style collapse, and the headline),
`abstention_rate` (measured on the response *opening*, so committing first and
hedging later does not count), `unknown_terms` (**a triage aid, not a
hallucination detector** — the vocabulary is incomplete), `multi_step_accuracy`,
`no_call_precision`, `hallucinated_tool_rate`.

Two design constraints that are easy to break by accident:

- **Every number here is only meaningful as a base-vs-tuned delta.** There is no
  gold answer; run the baseline too.
- **`configs/assistant.yaml` deliberately omits `prompt.exam_protocol`.**
  Instructing the `FINAL ANSWER:` contract into the prompt would manufacture the
  exact format being measured. The persona *is* still sent, because it is sent at
  serving time.

The suites are hand-written and small on purpose — a generated suite would inherit
the same template bias as the training corpus.

---

## 13. Runtime environment (`docker/`)

| Path | Purpose |
| --- | --- |
| `docker/fine-tune/Dockerfile` | The training image, from `nvcr.io/nvidia/pytorch:25.11-py3` |
| `docker/fine-tune/build.sh` | Builds `cosimo-fine-tune:latest` from the repo root |
| `docker/fine-tune/run.sh` | Interactive shell (or one-shot command) with the repo and HF cache mounted |
| `docker/fine-tune/torch_arch_guard.py` | Build-time guard: fails if anything replaced the NGC torch |
| `docker/serve/run.sh` | Serves a merged checkpoint on vLLM, OpenAI-compatible, loopback only |
| `docker/app/Dockerfile` | The application image (not part of the harness) |

**Docker is the only supported path** for the harness; there is no host
`uv`/`pip` variant. `run.sh` mounts the repo at `/workspace/cosimo`, sets the
working directory to `jobs/fine-tune`, bind-mounts `~/.cache/huggingface`, and
forwards `HF_TOKEN` / `WANDB_API_KEY` when set.

Version pins live in the **repository-root `pyproject.toml`**, in the
`[dependency-groups] fine-tune` group — not a `requirements.txt`. The Dockerfile
extracts that group with stdlib `tomllib`, so the group is the single source of
truth. `torch` is deliberately absent: it comes from the NGC base image, and a
PyPI torch would destroy the aarch64 CUDA 13 build. `unsloth` / `unsloth_zoo` are
installed `--no-deps` because their declared dependencies conflict with the
NGC-tuned stack.

Serving requires `--tool-call-parser hermes`; without it vLLM returns raw
`<tool_call>` text as message content and the ReAct loop terminates on the first
step. `--max-model-len 8192` matches what the LoRA was trained at, not the
architecture's declared 128 K window.

### 13.1 Tests

`jobs/fine-tune/tests/` is CPU-only — no GPU, no network, no torch — and
**`pytest` is not installed in the fine-tuning image**. Run it from a host venv:

```bash
.venv/bin/python -m pytest jobs/fine-tune/tests -q
```

This is separate from the application's own suite (`make test`, `tests/`).

---

# Part III — Contracts between subsystems

## 14. Where the corpus and the harness touch

Five couplings. The first is the main data path; the rest are narrow, easy to
break silently, and each has a gate.

### 14.1 The Hub is the handoff

`01_prepare_data.py` loads every source in `dataset.hub_id` + `dataset.mix` via
`load_dataset` — by default `btech-software/cosimo-quant-reasoning-v2` (configs
`default` and `preference`) plus `btech-software/cosimo-cfa-frm-71k` capped at
12 % of the merged **trainable** pool (config `default` only; its
`preference_pairs` are not used). Held-out records are exempt from the cap —
they never train, so subsampling them would only shrink the `unseen_stems`
measurement. **The harness does not read `dataset/shards/`.** Regenerating the corpus
locally has no effect on training until it is published
(`dataset/publish/push_to_hub.py`) and a source's `revision` points at it.

Consequence: `revision: main` means the corpus can move under a training run.
Pin a SHA for anything you intend to defend; the manifest records the resolved
SHA of every source either way.

Three shape differences between the two corpora are load-bearing, and
`cosimo_ft/data_schema.py` is the single place that reconciles them:

| | v2 | v1 |
| --- | --- | --- |
| `metadata` / `verification` / `conversation` / `tool_schemas` | JSON-encoded **strings** | Arrow **structs** (first two only) |
| Record types | five, discriminated by `record_type` | exam only (no such column) |
| Preference ids | `cosimopref_`, **disjoint** from supervised | **shared** with supervised |

The first is the dangerous one: reading a JSON string as an empty mapping
resolves every generator to `unknown`, which collapses the split stratification
to a single stratum and makes every configured holdout family match nothing.
`01_prepare_data.py`'s gate turns that into a hard failure rather than a silent
corpus.

### 14.2 Held-out suites must not be contaminated

`dataset/verification/suite_overlap.py` reads
`jobs/fine-tune/suites/{open_ended,calibration,agentic}.jsonl` and **fails
generation** when a generated record's token-set Jaccard against any suite prompt
exceeds 0.6, recording the result to `verification/suite_overlap.json`.

These suites are the only evaluation measuring the actual objective rather than
exam accuracy, and a contaminated instrument cannot be un-contaminated — every
cross-round comparison built on it becomes meaningless. The similarity check is
crude and deliberately over-sensitive: a false positive costs one reworded
generator, a false negative costs the evaluation.

### 14.3 The taxonomy is the terminology vocabulary

`configs/assistant.yaml` lists `../../dataset/taxonomy/taxonomy.json` alongside
`suites/glossary.txt` as `vocabulary_files`, and `09_assistant_eval.py`'s
`unknown_terms` metric flags every technical term absent from their union.

So a term the corpus teaches but the taxonomy never names is reported as unknown.
Extending the taxonomy is what keeps that signal readable — and it is why the
metric is a triage aid rather than a threshold.

### 14.4 One tool-calling wire format, two repositories

`jobs/fine-tune/cosimo_ft/tools.py` owns the format;
`configs/chat_template.jinja` renders it at training and serving time;
`tests/test_tools.py::test_rendered_tool_call_matches_the_template` asserts the
two emit byte-identical strings. `dataset/pipelines/generate.py` generates
`agentic` records against that same rendering, and
`dataset/scripts/smoke_generate.py` checks every agentic record survives it.

A training target differing from the served rendering by one space teaches a
format the runtime cannot parse back, and nothing else would catch it.

### 14.5 The shared failure mode

Both subsystems encode the same lesson from the first full run, in different
places: **response-shape uniformity is a training failure, not a quality signal.**
The corpus side enforces it through `FORMAT.md`, the length gate, and
`FINAL ANSWER:` being restricted to `exam` records (§8); the harness side measures
it through `exam_shape_rate` and `mean_new_tokens` (§12.3). A change on one side
that ignores the other will not be caught by either.

### 14.6 The record type decides the prompt surface

The harness half of §14.5, made mechanical. `FINAL ANSWER:` and
`prompt.exam_protocol` belong to `exam` records and nothing else. The corpus
guarantees it at generation time; the harness re-checks it twice, because a
corpus change and a harness change can each break it alone:

- `01_prepare_data.py`'s validation gate checks **every written row** in both
  directions: an exam row must carry the protocol in its system block and the
  tag in its target, and no other record type may carry either.
- `04_train_sft.py`'s masking check re-derives the same fact from the tokenized
  row — the tag appears in the *masked* prompt span if and only if the row is an
  exam row — so it holds even for a hand-supplied `--train-file`.

The four non-exam types render as: the answer verbatim (`analysis`,
`abstention`), fenced code plus the result (`implementation`), or the whole
conversation from the first assistant turn with `tool_schemas` bound
(`agentic`, via `chat.render_tool_example` — the same wire format as §14.4).

Only `exam` records are gradeable — `grading.grade_cosimo` reads a final-answer
value — so the two evaluation slices are exam-only and non-exam records are
split with `test_frac = 0`. The holdout still applies to every record type, or a
family leaks back into training through its non-exam rows.
