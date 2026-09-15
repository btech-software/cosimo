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
`dataset/progress/progress.md` for v1/v2 and from `verify_v3`'s board and
`dataset/tools/slice_audit.py` for v3. Training wall clock, accuracy and
token-length figures live in the harness README and in each run's
`metrics.json`. Numbers written into prose go stale silently.

**Two corpus generations live in this repository.** `dataset/pipelines/v3/` is
the current one and is where new work happens; the template generator that
produced v1 and v2 is frozen and documented in §3 for provenance. They share no
code.

---

## 0. Subsystems at a glance

| Path | Subsystem | Documented in |
| --- | --- | --- |
| `dataset/pipelines/v3/` | **Current** corpus: fact packs → briefs → gated render | §2 |
| `dataset/pipelines/` (v1/v2) | Frozen template generator, kept for provenance | §3–§9 |
| `dataset/tools/` | Operator instruments: slice audit, probes, suite builders | §2.8 |
| `jobs/fine-tune/` | Post-training: SFT → DPO/ORPO, evaluation, export | Part II below |
| `docker/` | The only supported runtime environments | §13 |
| `cosimo/` | LangGraph ReAct application — the serving target | *not yet documented* |
| `tests/` | Application tests (`Makefile`: `make test`) | *not yet documented* |

The corpus and the harness are separate programs joined by published artifacts,
not by imports:

```
dataset/pipelines/v3/        inventory → packs → render → verify → prefer → publish
   │   publish certifies the board and writes a card; it does NOT push yet
   ▼
dataset/shards/v3/           sft/ eval/ preference/ dead_letter/ fact_packs/
   │   scripts/01_prepare_data.py --set dataset.local_dir=<tree>     ← the normal path today
   │   (dataset.hub_id + dataset.mix is the eventual path; see §14.1)
   ▼
jobs/fine-tune/              prepare → tool rows → SFT → DPO/ORPO → evaluate → merge
   │   scripts/08_export_merge.py → runs/<name>/merged  (bf16 + chat template)
   ▼
docker/serve/run.sh          vLLM, OpenAI-compatible API on :8000
   │
   ▼
cosimo/agents/react_agent/   LangGraph create_react_agent (serving target)
```

The student is **`Qwen/Qwen3.8-27B`**, pinned by SHA, trained as QLoRA (4-bit,
`max_seq_length` 8192). The teacher that writes the corpus is a separate,
swappable endpoint whose answering model is stamped into every row.

Narrower contracts cross the boundary directly, without going through the
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

## 2. The v3 pipeline — current (`dataset/pipelines/v3/`)

v3 replaced "run a template a thousand times" with "compute a scenario, then
commission prose against it". The unit is the **fact pack**: a frozen,
JSON-serialisable scenario carrying its own verification contract. Every row —
exam, prose, agentic, implementation — is rendered over one, and the verifier
recomputes the pack from `(work_type, family, variant)` and requires equality.
That is why the pack is data and never prose.

```
dataset/taxonomy/work_types.yaml     the plan: families, registers, record types, caps
        │  inventory.py              → deterministic job list, family-capped, illegal pairs refused
        ▼
packs/<computer>.py                  fact computers (TCA, Brinson-Carino, VaR, FCFF, multiples)
        │  a scenario that cannot be computed raises PackError and is skipped, never shipped
        ▼
teacher/prompts.py                   system (short) + kind × register × work-type brief
teacher/routing.py                   lane, think flag, budget            teacher/client.py → one POST
        ▼
render/{prose,exam,agentic,implementation}.py
        │  gate → repair → gate, three strikes, then a dead letter carrying the whole ladder
        ▼
dataset/shards/v3/{sft,eval}/<record_type>.jsonl      + dead_letter/, teacher_logs/
        │
        ▼
verify_v3.py (16 axes)  ·  tools/slice_audit.py  ·  prefer.py  ·  publish.py
```

### 2.1 The plan and the inventory

`taxonomy/work_types.yaml` is law: which families exist, which are holdout,
which **registers** each family may be written in, how many variants each record
type gets, and the ceiling any one family may take of the supervised pool.
`inventory.py` expands it into `[(work_type, family, record_type, variant)]`,
applies the family cap by largest-remainder apportionment (so truncation
preserves a family's planned *shape* rather than keeping whichever record type
sorts first), and **refuses illegal pairs at load**: a `memo` needs a family
whose registers are all memo-legal, because one pack serves every record type of
a variant and the register is drawn before the record type is known.

### 2.2 Fact packs (`packs/`)

Each computer owns one work type and returns a `FactPack`. Beyond the scenario's
inputs and computed figures it carries the contract the answer is graded
against:

| field | what it decides |
| --- | --- |
| `canonical` | the one official value per named quantity — what an *answer* may claim |
| `allowed_numbers` | the union any surface may print (oracle results, distractors, pinned tests) |
| `aliases` | spellings the *question* prints, so a rounding it introduces is not the answer's |
| `display` | how the desk spells a figure — `430,567`, `4.86%`, `295.77` |
| `conventions` | named constants of the discipline the answer may cite unprompted |
| `must_mention` / `forbidden_claims` | the points to engage and the claims to refuse |
| `conditional_mentions` / `conditional_forbids` | points whose truth is an inequality *in this pack* |
| `abstention_question` / `abstention_missing` | the question this pack cannot answer, and what is absent |

Three of these exist because a single list could not answer two questions at
once. `canonical` versus `allowed_numbers` separates "what an answer may claim"
from "what may appear anywhere". `conventions` separates knowledge from facts —
a Sharpe band or a Basel multiplier is not a figure of this scenario, and a
corpus where every number must come from the pack can teach arithmetic and
refusal but never judgement against a benchmark. `abstention_question` exists
because an abstention rendered over a question the pack *answers* is not a
refusal; it is an analysis with a caveat.

Conditional entries are folded at build time against the pack's own numbers, so
the teacher never sees a slogan its arithmetic has already ruled out — a TCA
pack below its participation cap forbids "the schedule itself becomes the risk";
a VaR pack with a negative daily mean forbids "positive drift".

### 2.3 The brief (`teacher/prompts.py`)

The system turn is short and says only what is true of every row: use the pack's
figures in their display form, invent no entity, say what is missing if the pack
cannot support the question, never name the machinery. Everything about *this*
row lives in the user turn, composed from four sources:

- **kind** — the job: what the first sentence does, what carries it, what would
  overturn it, whether it ends in a decision, and its word and sentence caps;
- **register** — the shape, taken from the gate's own words
  (`verification/register.py`), so the brief and the refusal cannot disagree;
- **work type** — what this arithmetic makes wrong (participation below a cap is
  an impact bill; a negative mean is never a gain; EV is not a share price);
- **policies** — number spelling from `display`, the points to engage, and the
  conventions this work type licenses.

The rule this encodes is that **kind is the job and register is the shape**.
Differentiating record types by word budget alone is how `analysis` and
`grounded` collapse into one instruction wearing two names.

### 2.4 Rendering and the repair loop (`render/`)

`render_prose_row` asks the routed teacher, runs the gate, and on a violation
appends the draft plus a repair turn naming every fault, cooling the temperature
one notch — three strikes, then the whole exchange goes to `dead_letter/` with
its attempt ladder, never silently. A truncated reply (no text, `finish_reason:
length`) is *not* a strike: it is answered with more room, and the lane's
observed floor rises so later rows open where this one ended.

Two coordinates are skipped before a call is ever made: ids the **gold bar**
holds (a certified row is not regenerated) and packs the **eval reservation**
names (§2.7). Both are reported per stage rather than silently dropped.

Rows are written to `sft/` or `eval/` by the family's holdout flag — two
directories, because a glob cannot confuse them the way a boolean on a row can.
The student row carries the question and the visible answer and nothing of the
factory; the teacher transcript is a separate surface written only when the
operator asks for it.

### 2.5 The gate (`verification/`)

`prose.gate_violations` is the whole difference between "the teacher wrote
something" and "the corpus may carry it". One policy, three call sites — the
renderer's repair loop, the verify board, and the publish-time slice audit — so
the generator and the auditor cannot disagree about clean:

| axis | refuses |
| --- | --- |
| `invented_numbers` | a figure no canonical value or declared convention explains, at the precision it is written |
| `rounding_drift` | the question's rounding of a figure the answer is reporting |
| `display_form_offenders` | the raw float where the pack publishes a desk spelling |
| `integer_format_offenders` | a count with a decimal tail |
| `overprecise_numbers` | figures past desk precision |
| `contract_leaks` | naming the pack, the gate, the brief — the factory talking about itself |
| `malformed_prose` | unbalanced brackets, a word fused to a figure, a word printed twice |
| `missing_mentions` | points the answer does not engage (the gap, for an abstention) |
| `forbidden_hits` | a forbidden claim asserted rather than warned against |
| `contradiction_violations` | claims this pack's own arithmetic refutes (§2.6) |
| `register_violations` | the shape the register promises, plus the kind's sentence cap |
| word budget | the band this kind allows *in this register* |

### 2.6 Contradiction gates (`verification/contradictions.py`)

Every other axis asks where a figure came from. These ask whether the answer
contradicts the arithmetic it is quoting — the class of failure that shipped
board-green because none of it invents a number:

- `schedule_risk_below_cap` — the schedule called the risk while participation
  sits under the pack's cap;
- `drift_sign` — a negative daily mean read as a gain;
- `unreconciled_call` — an effect named worth acting on while the pieces do not
  add to the active return;
- `ev_as_price` — an ownership call, or an EV-versus-price comparison, from a
  valuation carrying no market price.

Each is a comparison the pack can make against itself, and each is scoped: a
convention or a claim legal in one work type is invented in another. Denials are
read in the claim's own clause, and a quoted question is not a claim — both
because the sentence the corpus *wants* contains the same words as the one it
refuses.

### 2.7 Evaluation reservations and the gold bar

Two fences, answering different questions:

- **`dataset/goldbar/gold_bar_v3.jsonl`** — rows a human certified. `verify_v3`
  axis 13 fails a training row that near-duplicates one, and the renderer skips
  their ids so the corpus cannot regenerate them.
- **`dataset/eval/reserved_coordinates.json`** — whole *packs* an evaluation
  interrogates, written by `tools/trap_suite.py`. Broader than the gold bar by
  design: a suite that asks about a scenario poisons any training row rendered
  on it, whatever its record type.

### 2.8 Operator instruments (`dataset/tools/`)

- `slice_audit.py` — is this slice worth scaling from: invented-number rate,
  contradiction tags, the factory fingerprint, holdout leakage, near-duplicates,
  whether `analysis` and `grounded` open on the same sentence, and whether the
  think ablation has actually been run. `dataset_build.sh full` greps its last
  line.
- `probe_teacher.py` — what this endpoint costs before a slice is paid for:
  whether the think flag is honoured, and the completion-token spread. The
  budgets in `config.py` are its findings, not estimates.
- `think_ablation.py` — the §B bake-off, think-off versus think-on on identical
  briefs. `--arm` runs one side when a brief change needs re-measuring;
  only a two-arm run writes the verdict.
- `trap_suite.py` — generates `jobs/fine-tune/suites/traps.jsonl` **from the
  packs**, so the figures are the corpus's own, and writes the reservation those
  packs then get.
- `smoke_corpus.py` — assembles a small training corpus from committed rows,
  excluding the gold bar, the reserved packs, and the scripted examples.

### 2.9 The board, preference and publish

`verify_v3.py` runs sixteen axes over the written shards, recomputing every pack
rather than trusting the copy stored on the row: schema, pack recompute,
invented numbers, `must_mention` / forbidden claims, the exam-only tag, three
share bands, tool schemas and replay, hidden tests, preference disjointness,
gold-bar near-duplication, teacher pinning, and two slice-level axes — corpus
near-duplication and register separation — that no per-row gate can ask.

`prefer.py` builds preference pairs as a second telling plus a named defect;
`publish.py` certifies the board and writes a dataset card. Publishing refuses
without the gold bar.

---

## 3. Legacy v1/v2 generator (frozen)

Everything in this section describes the **frozen** template generator that
produced v1 and v2. It is kept for provenance and for the exam corpus it
published; new work happens in §2, and nothing here is on the v3 path. The v3
pipeline shares none of its code — no templates, no `generate.py`, no
`verify_all.py`.

### 3.1 Data flow at a glance


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

### 3.2 Component map

#### 3.2.1 Taxonomy (`dataset/taxonomy/taxonomy.json`)

Topic/subtopic scaffolding that informs template organization. It is the
curriculum source of truth; templates map topics to question stems.

#### 3.2.2 Templates (`dataset/pipelines/templates/*.py`)

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

#### 3.2.3 Core helpers (`dataset/pipelines/core.py`)

Shared utilities consumed by the generator: content-hashed record IDs, the
deterministic `RNG` wrapper over `random.Random(seed)`, number/percent formatting,
and shard-path helpers. Defines `BASE_DIR`, `SHARDS_DIR` and `PROGRESS_DIR` —
see the hard-rules section below for how these resolve.

#### 3.2.4 Generator driver (`dataset/pipelines/generate.py`)

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

#### 3.2.5 Preference pairs

Built inline in `generate.py` (`build_preference`) whenever a template returns
`flawed` **and** `rng.r.random() < PAIR_RATIO`, plus the dedicated generators in
`templates/v2_preference.py`. The legacy `pipelines/preference.py` helper was
**removed**; inline + `v2_preference` only.

#### 3.2.6 Verification (`dataset/verification/`)

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

#### 3.2.7 Progress (`dataset/pipelines/progress.py`)

Scans shards and emits the live report (`dataset/progress/progress.md` and
`progress.html`) of counts, coverage, and known gaps. This is the single source
of truth for corpus numbers.

#### 3.2.8 Evaluation and gold bar (`dataset/eval/`, `dataset/goldbar/`)

- `goldbar/gold_bar.jsonl` — the curated assistant-transcript gold bar that
  defines the quality target.
- `goldbar/validate.py` — structural validation of the gold bar, per record type.
  Exits non-zero on failure.
- `eval/ab_eval.py` — blind A/B of generated records vs. the gold bar. Where the
  gold bar overlaps the shards, this is calibration-vs-gold, not an independent
  oracle.
- `eval/diversity.py` — structural-novelty report (distinct stems, per-topic
  coverage).

#### 3.2.9 Config (`dataset/config/seed.json`)

Central seed configuration consumed by generation, including
`preference_pair_ratio`.

#### 3.2.10 Scripts and publishing (`dataset/scripts/`, `dataset/publish/`)

- `scripts/smoke_generate.py` — Phase A verification: one variant per generator
  into a scratch directory (`dataset/.smoke/shards`). Proves the pipeline is whole
  without a bulk run — checks every generator produces a record, all record types
  are present, per-type required fields exist, numeric recomputation passes,
  agentic records render through the chat template, seeds are process-stable, and
  a second run is idempotent. Exits non-zero on any failure.
- `scripts/publish_dataset.py`, `publish/push_to_hub.py`, `publish/dataset_card.md`
  — Hub publishing and the dataset card.

---

### 3.3 Output artifacts

- **Shards** (`dataset/shards/<program>/<program>_shard_XXXX.jsonl`) — append-only,
  atomic, resumable. Gitignored.
- **Progress report** (`dataset/progress/progress.md`, `.html`) — live counts,
  coverage heat-map, response-length distribution, honest gaps.
- **Gold bar** (`dataset/goldbar/gold_bar.jsonl`) — curated exemplars defining the
  quality target.

---

### 3.4 Hard rules / conventions

1. **Paths are anchored to `dataset/`, not to your shell — with two exceptions.**
   Almost every script resolves `BASE_DIR` from `__file__`
   (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`) and builds
   absolute paths from it, so those commands work from **any** working directory.
   The exceptions glob a hardcoded relative `shards/` and **must be run with
   `dataset/` as the working directory**:
   - `eval/diversity.py` (line 7)
   - `scripts/publish_dataset.py` (line 142)

   Run from anywhere else they silently see **zero records** — `diversity.py` then
   crashes in `min()` on the empty counter. Either `cd dataset` first (see Commands below) or
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

### 3.5 Commands

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

# Must run with dataset/ as CWD — globs a relative 'shards/' (see the hard-rules section).
# diversity.py additionally assumes the v1 record shape and currently crashes (see Gotchas).
(cd dataset && python3 eval/diversity.py)
(cd dataset && python3 scripts/publish_dataset.py)
```

---

### 3.6 Adding a new question stem

1. Add a template function to the relevant module under
   `dataset/pipelines/templates/` — an exam program module or the `v2_*` module
   for the record type — matching the contract in the component map above.
2. Register it in that module's `TEMPLATES` dict (stem name → fn).
3. Generate it in isolation first:
   `TEMPLATE=<stem> python3 dataset/pipelines/generate.py`.
4. Run `python3 dataset/scripts/smoke_generate.py` — it must exit 0.
5. Run `python3 dataset/verification/verify_all.py` — it must exit 0.
6. Refresh `python3 dataset/pipelines/progress.py`.

---

### 3.7 Gotchas

Check these before editing.

- **Two scripts are CWD-dependent** — `eval/diversity.py` and
  `scripts/publish_dataset.py` glob a relative `shards/`, so from the repo root
  they report an empty corpus instead of erroring usefully. Everything else is
  `__file__`-anchored. See the hard-rules section.
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
a single-stage ORPO alternative. Student **`Qwen/Qwen3.8-27B`**, pinned by commit
SHA, **QLoRA** (4-bit NF4, bf16 compute) at `max_seq_length` 8192 — 27 B
parameters at bf16 leave no room for an 8 k sequence and its activations inside
128 GB of unified memory.

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
| `01_prepare_data.py` | a local v3 tree (`dataset.local_dir`) or Hub sources | `data/processed/{sft,pref}_{train,val}.jsonl`, `eval_cosimo_{test,unseen_stems}.jsonl`, `split_manifest.json` |
| `02_prepare_tool_data.py` | tool families in-script, **sized against the prepared corpus** | `data/processed/tool_{train,val}.jsonl` |
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
- **`02_prepare_tool_data.py` sizes itself against the corpus.** The synthetic
  tool rows teach a *format* the corpus contains no examples of — declining to
  call, and reading 2–5 competing schemas. Their count is `tools.train_share` of
  whatever `01_prepare_data.py` actually wrote, not a constant: a fixed 2,000
  was chosen when a far larger corpus supplied the real agentic rows, and left
  unchanged it would make format drills a large minority of the supervised
  signal. The resolved count and the share it will occupy are printed.
- **`04_train_sft.py` refuses a training set outside its declared band.**
  `sft.min_train_rows` / `sft.max_train_rows` exist because the smoke path and
  the full path are the same script reading whatever was prepared last: without
  a band, a forgotten `--config configs/sft_smoke.yaml` turns a smoke into a
  multi-hour run, and a forgotten re-prepare turns a full run into a handful of
  rows that looks like a bad model rather than a mistake.

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
  path that discovers what this checkpoint actually exposes; Qwen3.8 carries the
  conventional seven — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj` — where the previous Phi student carried fused ones, and
  a hardcoded list would have matched nothing across the swap), `attach_lora`,
  `model_fingerprint`. `model.use_exact_model_name` keeps unsloth from silently
  substituting its own pre-quantized mirror, which loads with no quantization
  state.
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
`max_seq_length`, the dataset source, `paths.*`, the `prompt.*` block, `tools.*`
sizing and generation parameters, and `chat.*`. Stage files (`data`, `sft`,
`sft_smoke`, `dpo`, `orpo`, `eval`, `assistant`) add only their own keys.

`configs/sft_smoke.yaml` is a named overlay rather than a pile of `--set` flags,
so "run the cheap experiment" is reproducible and the row-count band has a
ceiling to enforce. It changes the corpus size and the cadence and deliberately
**not** the learning rate, LoRA geometry or batch shape: a smoke that trains
differently from the real run measures the smoke.

Two budgets are derived rather than pinned, because a literal that outlives the
run it was chosen for is silently wrong. `sft.eval_steps` / `save_steps` accept
`auto` and resolve from the number of optimizer steps this run will actually
take — a literal 250 fires zero times on a corpus that takes far fewer, so the
eval-loss curve the data config describes would not exist. And the escalation
ceiling for a truncated teacher call is an absolute token count, not a multiple
of the lane cap: raising the cap once moved a ceiling nobody had chosen and cost
a render two hours of wall clock on two rows.

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

`prompt.identity` is present on **every** example, training and inference — it
binds "being Cosimo" to the weights rather than to a system prompt someone might
forget to send. `prompt.exam_protocol` is appended **only** to exam-format items
and carries the `FINAL ANSWER:` grading contract.

The identity block shrank by roughly an order of magnitude with v3 (four lines,
not the v2 essay). That is a training decision, not tidying: the block is paid
for on every example of every epoch, and a long persona spends the sequence
budget teaching the model to recite itself.

The split is deliberate and load-bearing: attaching the exam protocol to
everything is precisely how a model learns that being Cosimo *means* answering in
five formulaic steps. The identity is universal; the task block is not.

### 11.2 The chat template is overridden on purpose

`configs/chat_template.jinja` is the one template the whole harness renders
through — preparation, SFT, DPO, ORPO, evaluation, export — so the base model is
evaluated through the same prompt surface as the tuned one.

It began as a vendor template with one sentence removed: the Phi student's stock
template hardcoded `<|system|>Your name is Phi…` ahead of every system message,
contradicting the identity being trained. Under Qwen3.8 the turn markers are
`<|im_start|>role\n` / `<|im_end|>`, and the file is maintained rather than
inherited for a second reason: it defines where the supervised span begins
(`chat.response_part`), it passes `<think>` blocks through as ordinary content
(nothing here adds, strips or requires them), and it renders the tool wire
format shared with `cosimo/tools/wire.py` (§14.4).

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

Is it still an assistant. Seven suites. Six are hand-written — `open_ended`,
`calibration` (underspecified / unanswerable / false-premise), `agentic` (mock
ReAct trajectories including multi-call and no-call-appropriate), and the three
v3 record types that had no bucket: `grounded`, `memo`, `critique`.

The seventh, **`traps`, is generated from the fact packs themselves**
(`dataset/tools/trap_suite.py`) and is the exception that proves the
hand-written rule. Its figures must be the corpus's own to the last decimal, so
hand-typing them would measure the typo; and its questions are the four this
corpus exists because the *teacher* got wrong — a clip below its participation
cap called a pacing problem, an effect named worth acting on out of pieces that
do not reconcile, a negative drift read as a gain, an enterprise value turned
into an ownership call. Every other suite asks a competent desk question that a
capable base model answers, so a green board on them says nothing about any of
those.

Because the suite interrogates whole scenarios, its packs are **reserved against
generation** (§2.7). Train on them and the measurement is recall.

Metrics: `exam_shape_rate` (the direct read on style collapse, and the headline),
`abstention_rate` (measured on the response *opening*, so committing first and
hedging later does not count), `unknown_terms` (**a triage aid, not a
hallucination detector** — the vocabulary is incomplete), `multi_step_accuracy`,
`no_call_precision`, `hallucinated_tool_rate`, and on rows that declare the
contract they measure: `invented_numbers`, `must_mention`, `register_match`,
`conventions_cited`.

`invented_numbers` here grades against the row's own figures **plus the
standards of the field it declares**, and counts convention use separately as a
feature rather than a fault. The distinction is the difference between a metric
and a coin: a correct Sharpe answer that places 0.43 against the conventional
bands is doing the job, and scoring it as invention trains an assistant that
will not contextualise a number. The same scorer mirrors the corpus's rounding
rule — a figure rounded for the reader is the same figure — because a generator
and an evaluator that disagree about what counts as the same number certify a
corpus by a rule it was not written to.

One ceiling is load-bearing and easy to get wrong: `assistant.max_new_tokens`
must hold a chain of thought *plus* an answer. The student thinks by default and
the template passes `<think>` through, so a ceiling sized for the answer alone
clips every generation mid-reasoning and every metric above is then computed
over truncated reasoning rather than over a reply.

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

Seven couplings. The first is the main data path; the rest are narrow, easy to
break silently, and each has a gate.

### 14.1 The Hub is the handoff

`01_prepare_data.py` loads every source in `dataset.hub_id` + `dataset.mix` via
`load_dataset` — by default `btech-software/cosimo-quant-assistant-v3` (configs
`default` and `preference`), with `dataset.mix: []`. **The harness does not read
`dataset/shards/` by default.** Regenerating the corpus locally has no effect on
training until it is published and a source's `revision` points at it.

Consequence: `revision: main` means the corpus can move under a training run.
Pin a SHA for anything you intend to defend; the manifest records the resolved
SHA of every source either way.

**The one exception, and it is currently the normal path.** `dataset.local_dir`
makes `01_prepare_data.py` read a local v3 shard tree — `<dir>/sft/*.jsonl` and
`<dir>/preference/pairs.jsonl` — instead of the Hub. It exists because v3's
`publish` command certifies the verification board and writes a dataset card but
**does not push**: there is no v3 Hub repo yet for `load_dataset` to read. This
is a deliberate hole in §14.1's own rule, and it closes when the repo is
published (spec §10, PR6). While it is open, provenance is the `local_dirs`
entry in the manifest plus the per-file content fingerprint in `row_sets` — the
rows actually read, rather than a commit that might contain them.

Shape differences across the three corpora are load-bearing, and
`cosimo_ft/data_schema.py` is the single place that reconciles them:

| | v3 | v2 | v1 |
| --- | --- | --- | --- |
| Nested columns | native lists/dicts (raw JSONL) | JSON-encoded **strings** | Arrow **structs** |
| Record types | eight, discriminated by `record_type` | five | exam only (no such column) |
| Taxonomy axis | `work_type` / `scenario_id` / `register` | `program` / `topic` / `subtopic` | same as v2 |
| Holdout axis | scenario family (`<work_type>.<family>`) | `v_`/`cr_`/`m_` stem family | same as v2 |
| Preference ids | `cosimov3pref_`, **disjoint** | `cosimopref_`, **disjoint** | **shared** with supervised |
| Certification | a `verification` stamp; no `verified` column | boolean `verified` | boolean `verified` |
| Exam close | `FINAL ANSWER: <letter> -- <value> <unit>` | `FINAL ANSWER: <value>` | `FINAL ANSWER: <value>` |

Two of these are the dangerous ones.

For v1/v2, reading a JSON string as an empty mapping resolves every generator to
`unknown`, which collapses the split stratification to a single stratum and makes
every configured holdout family match nothing. `01_prepare_data.py`'s gate turns
that into a hard failure rather than a silent corpus.

For v3 the equivalent is the axis mapping. `normalize_v3_record` maps
`work_type` → `program` and `scenario_id` → both `generator` and `stem_family`,
which is precisely why `splits.py` needs no v3 branch — it strata on
`(program, generator)` and holds out on `stem_family`. Leave any of the three
blank and the same single-stratum collapse follows, from a different cause.

### 14.1.1 Three holdout axes, and they are not the same axis

The corpus and the harness each hold things out, for different reasons, and a
reader who collapses them will misread every generalisation number:

1. **Plan holdout** — `work_types.yaml` marks one family per work type
   `holdout: true`. These *are* rendered now (the earlier behaviour dropped them
   outright, which left nothing downstream to hold out and pushed the
   generalisation claim onto families the model had trained on). A holdout job
   renders exactly like a train one and lands in `eval/` rather than `sft/` —
   two directories, because a glob cannot confuse them the way a boolean on a
   row can. `01_prepare_data.py` reads only `sft/`, so the plan's holdout
   families never reach training at all.
2. **Harness holdout** — `data.holdout_scenario_families` names *shipped*
   families excluded from train/val/test and reported as `unseen_stems`. This is
   a holdout *inside* the training distribution, and it costs real training
   rows; it is the honest number to read beside `cosimo_test`.
3. **Evaluation reservations** — packs a suite interrogates (§14.4.1), excluded
   from generation entirely rather than from training.

Only the first is free. The second buys its measurement with corpus, and the
third with coverage.

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

### 14.4 One tool-calling wire format, three repositories

`cosimo/tools/wire.py` owns the format since v3 (spec §8.5);
`jobs/fine-tune/cosimo_ft/tools.py` re-exports it, `dataset/pipelines/v3`
imports it, and `configs/chat_template.jinja` renders it at training and serving
time. `tests/test_tools.py::test_rendered_tool_call_matches_the_template` and
`tests/test_chat_template_qwen.py` assert the renderings are byte-identical.
`dataset/scripts/smoke_generate.py` checks every agentic record survives it.

A training target differing from the served rendering by one space teaches a
format the runtime cannot parse back, and nothing else would catch it.

This is why the Qwen3.8 template swap moved the **turn** markers
(`<|system|>`/`<|user|>`/`<|end|>` → `<|im_start|>role\n`/`<|im_end|>`) and
deliberately left the **tool** markers alone. Under Phi, `<|tool|>` was a real
special token; under Qwen it is ordinary text, costing a few tokens per example.
Paying that keeps `cosimo/tools/wire.py` — a contract shared across the
`dataset` ↔ `jobs` boundary that AGENTS.md §5.3 says must be changed alone —
untouched by a harness-only change.

### 14.4.1 The trap suite is generated from the packs, and reserves them

`dataset/tools/trap_suite.py` writes `jobs/fine-tune/suites/traps.jsonl` by
computing the packs it asks about, so every figure in a trap prompt is the
corpus's own. It writes `dataset/eval/reserved_coordinates.json` in the same
run, and `render/prose.py` skips any job on a reserved coordinate.

This is the second fence, and it answers a different question from the gold
bar's. The bar says *do not regenerate this row*; the reservation says *do not
generate any row on this scenario*, because a suite that interrogates a pack is
poisoned by any record type rendered on it. The two are easy to conflate and the
consequence is invisible: a training corpus that overlaps the trap packs turns a
generalisation test into a recall test, and a recall result reads as success
exactly when it should not.

`dataset/tools/smoke_corpus.py` applies both fences plus one more — it excludes
`examples/v3/<record_type>.jsonl` by name. Those files were scripted prose for
most of the project's life, and when a smoke corpus swept them up the adapter
reproduced the fixture harness's opening line verbatim. They are real rendered
rows now (§2.9's artifacts), and the exclusion stays because an example is
documentation, not training data.

### 14.5 The shared failure mode

Both subsystems encode the same lesson from the first full run, in different
places: **response-shape uniformity is a training failure, not a quality signal.**
The corpus side enforces it through `FORMAT.md`, the length gate, and
`FINAL ANSWER:` being restricted to `exam` records (§3.7); the harness side measures
it through `exam_shape_rate` and `mean_new_tokens` (§12.3). A change on one side
that ignores the other will not be caught by either.

v3 adds three more paired measurements of the same kind, and they pair the same
way: the corpus *refuses to publish* a row whose answer invents a number or
misses a `must_mention` term, and `09_assistant_eval.py` measures whether the
student learned those constraints (`invented_numbers`, `must_mention`,
`register_match`). The number gate is reimplemented in `cosimo_ft/assistant.py`
rather than imported, because `jobs` must not import `dataset` — so the two
policies can drift, and if they do, the eval quietly stops measuring what
generation enforced.

They have drifted twice, and both times the evaluator was the looser one: it
scored a figure rounded for the reader as invented, and it had no notion of the
`conventions` a pack declares. Both are corrected, and the drift is the thing to
watch on any change to either side — the corpus gate is authoritative and the
assistant scorer must be read as tracking it.

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

The seven non-exam types render as: the answer verbatim (`analysis`,
`abstention`, and v3's `memo` / `critique` / `grounded`), fenced code plus the
prose (`implementation` — v3's `reference_code` and `public_tests`, closing on
the teacher-authored `limitations`), or the whole conversation from the first
assistant turn with `tool_schemas` bound (`agentic`, via
`chat.render_tool_example` — the same wire format as §14.4).

One v3 subtlety on both ends of the exam contract. The options live in
`messages[1]`, not in the row's `question` column, so preparing the bare column
would ask the model for a letter without ever showing it the letters. And
`prompt.exam_protocol` had to move to the `<letter> -- <value> <unit>` form:
under the v2 wording the system block would have instructed a close that every
v3 supervised target contradicts, and the existing gate — which only checks that
the tag is *present* on exam rows — would not have noticed.

Only `exam` records are gradeable — `grading.grade_cosimo` reads a final-answer
value — so the two evaluation slices are exam-only and non-exam records are
split with `test_frac = 0`. The holdout still applies to every record type, or a
family leaks back into training through its non-exam rows.

### 14.7 The teacher is not the student

The endpoint that writes the corpus and the model being trained are different
systems, and nothing in the repository assumes otherwise. `teacher/routing.py`
sends an explicit `thinking` disable on prose lanes and stamps `think_present`
from the returned reasoning text rather than from the request flag, because a
serving stack that ignores the field would otherwise be invisible.
`dataset/tools/probe_teacher.py` measures what a given endpoint actually costs
before a slice is paid for, and the budgets in `pipelines/v3/config.py` are its
findings.

Every row records the model that answered it, which is what makes a mid-run
teacher swap auditable rather than a silent style drift — `verify_v3` axis 14
fails a corpus whose rows disagree about who wrote them.
