# Cosimo fine-tuning harness — live progress

*Status of the **harness itself**, not of any training run. No training, no GPU job, no image build
and no model download was executed while building this — it is scaffolding only, by design.*

**Updated**: 2026-08-03 · **Overall readiness**: ✅ ready to clone and run on a DGX Spark

---

## End goal

Turn a mathematics-specialised base model (`unsloth/Phi-4-mini-reasoning`, 3.8B) into an outstanding
**financial domain assistant**. Passing CFA/FRM exam items is the measurable milestone, not the
destination — a harness that maximises exam accuracy while degrading general assistant behaviour has
failed. See `README.md` § *Evaluation methodology* and its style-collapse warning.

## Pipeline

```
00_check_env → 01_prepare_data → 02_baseline_eval → 03_train_sft → 05_evaluate
                                                  → 04_train_dpo → 05_evaluate → 06_compare
                                                                              → 07_export_merge
                                     (alternative single-stage path: 04b_train_orpo)
```

## Components

| Component | Files | Status | Note |
|---|---|---|---|
| Environment (container) | `docker/fine-tune/{Dockerfile,build.sh,run.sh}`, root `.dockerignore` | ✅ | Base-image guard asserts sm_121 + aarch64 + CUDA 13; torch guard before and after each install |
| Dependency pins | root `pyproject.toml` `[dependency-groups] fine-tune` + `uv.lock` | ✅ | Single source of truth; Dockerfile extracts it with `tomllib`. `uv sync --frozen` still works for the app image |
| Environment check | `scripts/00_check_env.py` | ✅ | Real Triton kernel launch, bnb 4-bit smoke test, pin/marker parsing, dual-filesystem disk check |
| Core config/logging | `cosimo_ft/{config,runlog}.py` | ✅ | Typo'd `--set` paths rejected; torn JSONL lines tolerated on resume |
| Prompt + chat template | `cosimo_ft/chat.py`, `configs/chat_template.jinja` | ✅ | Two-block persona prompt; vendor identity preamble removed; persona byte-identical to source |
| Dataset schema | `cosimo_ft/data_schema.py` | ✅ | Struct/string/null `preference_pair`, MCQ, missing generator; trailing EOS stripped for DPO/ORPO parity |
| Splits | `cosimo_ft/splits.py`, `configs/data.yaml` | ✅ | Deterministic; stem-family holdout verified leak-free |
| Data preparation | `scripts/01_prepare_data.py` | ✅ | Config-vs-output gate; 13 adversarial corpora fail loudly or are counted |
| Grading | `cosimo_ft/grading.py` | ✅ | 7 defects fixed and verified case by case |
| Generation | `cosimo_ft/generation.py` | ✅ | Batching, length-bucketing, order restoration and token accounting verified correct |
| Model loading / LoRA | `cosimo_ft/modeling.py` | ✅ | Fused-projection auto-detection; `revision` pinned for training *and* evaluation |
| Benchmarks | `cosimo_ft/benchmarks.py` | ✅ | Cosimo test + unseen-stems + GSM8K + MATH-500 |
| Evaluation runner | `cosimo_ft/evalrun.py`, `scripts/{02_baseline_eval,05_evaluate}.py` | ✅ | Provenance isolated from training runs; template mismatch fatal before the model loads |
| Comparison + statistics | `cosimo_ft/report.py`, `scripts/06_compare.py` | ✅ | Wilson CI + exact McNemar (cross-checked against statsmodels and R); comparability guards incl. template hash |
| SFT training | `scripts/03_train_sft.py`, `configs/sft.yaml` | ✅ | Structural response-only masking assertion; truncation scan; TRL 0.24.0 API verified against source |
| Preference training | `scripts/04_train_dpo.py`, `04b_train_orpo.py`, `configs/{dpo,orpo}.yaml` | ✅ | Prompt/completion truncation scans; reference-free via adapter disable |
| Export / merge | `scripts/07_export_merge.py` | ✅ | Merges via `PeftModel.merge_and_unload`; reads saved keys back and rejects any `lora_`/`base_layer` |
| README (DGX Spark runbook) | `README.md` | ✅ | Single command sequence; every flag verified against real argparse |
| Tests | `tests/` | ✅ | 159 CPU-only tests, 0.24s, no network |
| Orchestration | `run_all.sh`, `.gitignore` | ✅ | `--dry-run`, `--eval-only`, `--limit`, `--suites`, baseline-clobber guards |

## Issues found and resolved

Every item below was found by an independent critic reviewing the real files, then fixed and
re-verified. Nothing here is speculative.

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | `07_export_merge.py` exported unmerged weights: `hasattr` dispatch resolved through PEFT delegation onto the unpatched base model, so Unsloth took its "no adapter to merge" branch and wrote `lora_A`/`base_layer` keys that reload as **random weights** — exit code 0 | Critical | ✅ Fixed + post-export key assertion |
| 2 | Evaluating a checkpoint overwrote that run's training provenance | High | ✅ Fixed |
| 3 | `06_compare.py` never verified two runs shared a base model, decoding budget or item set | High | ✅ Fixed — `NOT COMPARABLE` banner, `--strict` |
| 4 | Stale suite generation files survived `--force` and were silently reused | High | ✅ Fixed |
| 5 | `--resume` reported metrics over items the config did not request | High | ✅ Fixed |
| 6 | Vendor chat template asserts a Microsoft/Phi identity on every turn | High | ✅ Fixed — harness template everywhere; mismatch fatal; hash in every artifact |
| 7 | `max_prompt_length: 768` with left-side truncation silently deleted the identity block from long DPO prompts | High | ✅ Fixed — 1408 + truncation scans |
| 8 | DPO prompt truncation is unconditionally left-sided in TRL 0.24 regardless of `truncation_mode` | High | ✅ Comments corrected, scans added |
| 9 | Env check reported "ready" while the CUDA device query raised `no kernel image available` | High | ✅ Fixed — hard failure |
| 10 | Nothing proved a GPU kernel compiles for sm_121 (Unsloth *is* Triton kernels, JIT-compiled hours in) | High | ✅ Fixed — real kernel launch + arch-list assertion |
| 11 | Data prep verified what it produced, never that it matched its configuration | High | ✅ Fixed — mistyped/prefixed holdout family, empty split, unresolved revision all fail loudly |
| 12 | `--limit` head-sliced a generator-sorted corpus: the smoke run covered 2 generators, not 71 | High | ✅ Fixed — seeded sample; preference join 200/200 (was 8/200) |
| 13 | MCQ preference pairs: `chosen` letter-prefixed in 100/100, `rejected` in 0/100 — a one-token cue DPO could exploit instead of the reasoning | High | ✅ Fixed — letter resolved onto `rejected`; unresolvable pairs dropped and counted |
| 14 | `--dry-run` never took an optimizer step, so `compute_loss` failures were invisible | High | ✅ Fixed — pre-flight is dry-run **plus** a short real run |
| 15 | Seven grading defects (unicode case-expansion mis-index, hyphen-as-minus, bare-letter distractors, negation acceptance, prose-phrased MCQ, math last-number bias, lower-case article read as option A) | Medium | ✅ Fixed |
| 16 | DPO appended a second EOS while ORPO did not — different targets for identical pairs | Medium | ✅ Fixed at prep + gate assertion |
| 17 | Masking leak detector could abort a correct run and missed ≤5-word leaks | Medium | ✅ Replaced with an exact structural assertion |
| 18 | Records with null question/answer/trace rendered a bare `FINAL ANSWER:` target and passed the gate | Medium | ✅ Fixed |
| 19 | Base-model `revision` never pinned at evaluation time | Medium | ✅ Fixed |
| 20 | `--force` did not overwrite; stale checkpoints could be resumed under different hyperparameters | Medium | ✅ Fixed |
| 21 | `HF_TOKEN` passed as `-e NAME=value`, exposing it in `ps` | Medium | ✅ Fixed — passed by name |
| 22 | DPO docstring claimed the reference model is the SFT policy; it is the pre-SFT base | Medium | ✅ Documentation corrected |
| 23 | Env check: dev-version strings satisfied exact pins; unwritable `runs/` was only a warning; hub `<1.0` unchecked; markers silently ignored | Low–Med | ✅ All fixed |

## Open items — operator judgement, not defects

| # | Item | Why it is left open |
|---|---|---|
| A | Style collapse from terse exam traces | Mitigated by the two-block system prompt and identity variation, but it cannot be measured by any automated suite here. Spot-check open-ended financial questions against the baseline before shipping — the README gives the procedure. |
| B | Persona costs ~600 tokens/example, roughly doubling tokens per step | Accepted deliberately. Batch/accumulation rebalanced to hold effective batch at 32 so the tuned learning rate stays valid. |
| C | The published dataset's MCQ preference asymmetry | The harness normalises it at prep time, but the cue exists in `btech-software/cosimo-cfa-frm-71k` itself and is worth fixing at the generator. |
| D | `eval.samples.cosimo_unseen_stems` defaults to 1000, not the full ~6000 | The full slice adds hours per model across three models. Set it to `null` for a final published number. |
| E | Runs are not bit-reproducible despite `seed: 3407` | bf16 reduction order is nondeterministic; `torch.use_deterministic_algorithms` was not imposed. Documented in the README's reproducibility notes. |

## Verification evidence

- **159 CPU-only tests pass** in 0.24s with no network. They render through the *shipped*
  `configs/chat_template.jinja` via real jinja2, and keep the vendor template as a negative control
  so the "no Microsoft" assertions can genuinely fail.
- Split assignment verified on a 6,000-record synthetic corpus: deterministic across runs and input
  order, no id in two splits, zero held-out-family leakage, preference rows inherit their id's split.
- Data preparation verified end-to-end against synthetic corpora including every published-schema
  edge case; all validation gates were deliberately triggered and fired; two full runs produced
  byte-identical SHA256 for all six output files.
- TRL 0.24.0 / transformers 4.56.2 / peft 0.20.0 / unsloth 2026.8.1 call sites checked
  argument-by-argument against the **actual pinned sources**, not from memory.
- Wilson intervals cross-checked against statsmodels; exact McNemar against R's `binom.test`.
- Dependency set proven co-installable via `uv lock`; two real upstream version conflicts found and
  resolved rather than papered over.
- `bitsandbytes==0.49.2`'s aarch64 wheel confirmed to ship sm_121 cubins by parsing the library's
  fatbin sections — so the earlier "4-bit probably won't work" framing was wrong and was corrected.
- Every README command verified against the scripts' real argparse.
- `ruff check` clean across the harness; all scripts executable with valid shebangs; all shell
  scripts pass `bash -n`; all six YAML configs parse.

## What has NOT been verified

- **Nothing has run on a GPU.** No image built, no weights downloaded, no training step taken.
- Wall-clock and memory figures in the README are estimates with the arithmetic shown, not
  measurements.
- The `compute_loss` interaction between Unsloth's compiled forward and TRL's logits access
  (issue 14) is reasoned from source but unproven on hardware — which is exactly why the pre-flight
  now includes a short real training run before the full one.
- Token-length percentiles come from the real corpus only when the operator runs
  `01_prepare_data.py`; the manifest reports them, and they are the numbers to trust.

## First actions for the operator

1. `bash docker/fine-tune/build.sh`, then `bash docker/fine-tune/run.sh python scripts/00_check_env.py`
   — do not proceed past a red verdict.
2. `python scripts/01_prepare_data.py`, then **read `data/processed/split_manifest.json`**: check the
   token percentiles against `max_seq_length`, the `unseen_stems` share, and the dropped-row counts.
3. `python scripts/02_baseline_eval.py` — the reference measurement everything else is compared to.
4. `python scripts/03_train_sft.py --dry-run`, then a short real run
   (`--run-name sft_smoke --set sft.max_steps=20`) before committing to the full epoch.
