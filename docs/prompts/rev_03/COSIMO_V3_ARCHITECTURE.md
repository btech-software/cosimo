# Cosimo v3 software architecture — implement in-repo

This is the build plan for the v3 corpus and the Qwen3.8-27B student, written against
the existing two-subsystem layout in `ARCHITECTURE.md`. It does not replace v1/v2
generators. It adds a third generation backend that publishes a new Hub repo and
points the harness at it.

Invariant from the current architecture, kept:

> The corpus and the harness are separate programs joined by published artifacts,
> not by imports.

The one exception: **tool schemas** move into the already-existing `cosimo/`
package so generation, training, and serving cannot drift.

---

## 1. Target runtime picture

```
                    ┌─────────────────────────────────────────┐
                    │  OpenAI-v1 teacher                      │
                    │  default: deepseek-v4-flash             │
                    │  bake-off: qwen3.8-flash-next           │
                    └──────────────────▲──────────────────────┘
                                       │ HTTPS
┌──────────────────────────────────────┴───────────────────────────────────┐
│  CORPUS  dataset/                                                        │
│                                                                          │
│  taxonomy + work_types.yaml                                              │
│       │                                                                  │
│       ▼                                                                  │
│  v3 inventory ──► fact computers ──► fact_packs/*.jsonl                  │
│       │                                                                  │
│       ├─ exam renderer (local, optional teacher phrasing)                │
│       ├─ prose renderer (teacher + fact lock)                            │
│       ├─ agentic loop (teacher ↔ tool oracle)                            │
│       └─ impl renderer (local code + teacher limitations)                │
│       │                                                                  │
│       ▼                                                                  │
│  verify_v3 ──► shards/v3/ ──► preference ──► push Hub                    │
│                btech-software/cosimo-quant-assistant-v3                  │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │ Hub artifact only
┌──────────────────────────────────────▼───────────────────────────────────┐
│  HARNESS  jobs/fine-tune/                                                │
│                                                                          │
│  base.yaml: model.base_id = Qwen/Qwen3.8-27B                             │
│             dataset.hub_id = btech-software/cosimo-quant-assistant-v3    │
│                                                                          │
│  01_prepare → 03_baseline → 04_sft → 05_dpo → 06/09_eval → 08_merge      │
│                                                                          │
│  docker/fine-tune  (DGX Spark, Unsloth, aarch64)                         │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │ merged weights
┌──────────────────────────────────────▼───────────────────────────────────┐
│  SERVE  cosimo/ + docker/serve                                           │
│  LangGraph ReAct app, tools from cosimo.tools, vLLM OpenAI-v1            │
└──────────────────────────────────────────────────────────────────────────┘
```

Airflow is an **optional scheduler** around the corpus CLI. The CLI must remain
the source of truth so a Spark box and CI can run without Airflow.

---

## 2. What you add vs what you freeze

| Path | Action |
|---|---|
| `dataset/pipelines/templates/cfa_*.py`, `v2_*.py`, `generate.py` | **Freeze.** v1/v2 stay reproducible. Do not add teacher calls here. |
| `dataset/pipelines/v3/` | **New** generation backend. |
| `dataset/verification/verify_v3.py` | **New** gates; reuse `nums.py`. |
| `dataset/taxonomy/work_types.yaml` | **New** primary axis. Keep `taxonomy.json` as LOS metadata. |
| `dataset/publish/` | Add v3 card + hub id. Do not overwrite v2 card. |
| `cosimo/tools/` | **Extract** canonical tool schemas from `jobs/fine-tune/cosimo_ft/tools.py`. |
| `jobs/fine-tune/configs/base.yaml` | Point student + dataset at v3. ~~Keep a `base.phi4.yaml` archive.~~ (dropped -- decision log #7) |
| `jobs/fine-tune/configs/chat_template.jinja` | Qwen3.8 template (thinking + tool call). |
| `jobs/fine-tune/scripts/01_prepare_data.py` | Teach it v3 record types + `fact_pack`. |
| `ops/airflow/` | Optional DAG. Not required to generate. |
| `docker/corpus/` | Slim CPU image for teacher ETL. Separate from Spark fine-tune image. |

Do not mix v1 rows into v3 at `dataset.mix` once v3 exam exists. The 12% v1 cap
was a crutch.

---

## 3. In-repo layout

```
cosimo/                            # serving app + shared contracts
  tools/
    schemas.py                     # get_fundamentals, compute_metrics, ...
    registry.py                    # name → schema, versioned
  graph/                           # existing LangGraph (still thin)

dataset/
  taxonomy/
    taxonomy.json                  # LOS (unchanged)
    work_types.yaml                # NEW primary key
  pipelines/
    core.py                        # KEEP: rng, ids, shard paths
    generate.py                    # KEEP: v1/v2 driver
    templates/                     # KEEP: frozen
    v3/
      __init__.py
      config.py                    # teacher endpoint, mix caps, family cap
      inventory.py                 # scenario_family × record_type × variant plan
      seed.py                      # hash(work_type, family, variant, record_type)
      packs/
        __init__.py
        base.py                    # FactPack dataclass + allowed_numbers
        valuation_fcff.py
        valuation_multiples.py
        risk_var.py
        portfolio_attribution.py
        execution_tca.py           # start with 5; 15 is the first publish bar
      render/
        exam.py
        prose.py                   # analysis, memo, grounded, critique, abstention
        agentic.py                 # teacher ↔ oracle loop
        implementation.py
      teacher/
        client.py                  # OpenAI-v1
        prompts.py
        routing.py                 # record_type → model + think/no-think
      oracle/
        runtime.py                 # execute tool against fact_pack + dirty layer
        faults.py                  # stale, missing, schema drift, rate limit
      prefer.py                    # contrastive rejected + paraphrase chosen
      write.py                     # append-only shards under shards/v3/
      cli.py                       # `python -m pipelines.v3.cli ...`
  verification/
    nums.py                        # KEEP
    terms.py                       # KEEP
    verify_all.py                  # KEEP for v1/v2
    verify_v3.py                   # NEW
    invented_numbers.py            # or import from v3.verify
  shards/v3/                       # gitignored
    fact_packs/
    supervised/
    preference/
    dead_letter/
  goldbar/
    gold_bar.jsonl                 # KEEP, freeze
    gold_bar_v3.jsonl              # NEW, human, disjoint
  publish/
    dataset_card_v3.md
    push_to_hub.py                 # branch on --corpus v3

jobs/fine-tune/
  configs/
    base.yaml                      # Qwen3.8-27B + v3 hub
    (base.phi4.yaml was archived here, then dropped -- decision log #7)
    data.yaml                      # holdout_families → holdout_scenario_families
    chat_template.jinja            # Qwen3.8
  cosimo_ft/
    tools.py                       # re-export cosimo.tools
    data_schema.py                 # accept fact_pack, memo, critique, grounded
    chat.py                        # thinking / tool tokens
  scripts/                         # same 00–09 numbers

ops/airflow/
  dags/cosimo_v3_corpus.py
  requirements.txt                 # apache-airflow optional extra

docker/
  corpus/Dockerfile                # python 3.12, openai, no torch
  fine-tune/                       # unchanged Spark path
  serve/                           # point at merged Qwen

docs/
  ARCHITECTURE.md                  # add Part III: v3 (do not rewrite Part I/II)
  V3.md                            # this document, or a short pointer
```

Python import rule:

- `dataset.pipelines.v3` may import `cosimo.tools` and `dataset.pipelines.core`.
- `jobs.fine-tune.cosimo_ft` may import `cosimo.tools`.
- `dataset` must not import `jobs`.
- `jobs` must not import `dataset`.

That is the current architecture with one shared contract package.

---

## 4. Core types

Keep records JSONL. Discriminator is still `record_type`. Add fields; do not
break v2 loaders until `01_prepare_data.py` understands both.

```python
# dataset/pipelines/v3/packs/base.py

@dataclass(frozen=True)
class FactPack:
    schema_version: str          # "v3.0"
    scenario_id: str             # "valuation.equity.dcf.mature_consumer"
    work_type: str
    seed: int
    variant: int
    entities: list[dict]
    inputs: dict
    computed: dict
    formulas: list[str]
    allowed_numbers: list[float]
    forbidden_claims: list[str]
    must_mention: list[str]
    register: str                # desk_chat | ic_memo | ...
    as_of: str                   # ISO date
    question: str
    stimulus: str | None = None
    program: str | None = None   # LOS leftover, metadata only
    topic: str | None = None
    subtopic: str | None = None
```

Supervised row (published):

```
id, record_type, work_type, scenario_id, program, topic, subtopic,
difficulty, question_type, register, question, answer,
conversation, tool_schemas, code, test_code, defect,
fact_pack,           # JSON string, same Hub compromise as v2
verified, verification, metadata
```

`id` format: `cosimov3_{record_type}_{seed:016x}`  
Preference: `cosimov3pref_{seed:016x}` — disjoint by prefix, enforced in verify.

---

## 5. Corpus control plane

### 5.1 Inventory, not “run every template 1000 times”

`work_types.yaml` is the plan:

```yaml
valuation.equity.dcf:
  families:
    mature_consumer: {holdout: false}
    cyclical_industrial: {holdout: false}
    fade_required: {holdout: true}      # unseen_scenario_family
  record_types: [exam, analysis, memo, critique, grounded, abstention, agentic, implementation]
  variants_per_family:
    exam: 80
    analysis: 120
    memo: 40
    # ...
  max_share: 0.03
```

`inventory.py` expands this into a deterministic list of jobs
`(work_type, family, record_type, variant)`. Re-running the same inventory
resumes: existing shard ids are skipped.

Family cap is enforced here, not after the fact. If a family would exceed 3%
of planned SFT, inventory refuses to emit more variants.

### 5.2 Seed

```
seed = blake2s(f"{work_type}|{family}|{record_type}|{variant}|{schema_version}")
rng  = random.Random(seed)
```

No wall clock. Same contract as v1/v2, different tuple.

### 5.3 Fact computers

One module per work type. Pure Python. No network. Returns `FactPack`.

A computer that cannot produce a consistent pack (WACC ≤ g, empty table) raises
`PackError`; inventory marks the variant skipped, not dead-lettered.

First milestone: five computers.

```
packs/valuation_fcff.py
packs/valuation_multiples.py
packs/risk_var.py
packs/portfolio_attribution.py
packs/execution_tca.py
```

Each file owns its formulas. Verification re-imports the computer and
recomputes from `(seed, variant)`.

### 5.4 Teacher client

```python
# teacher/client.py
class Teacher:
    def complete(self, messages, *, temperature, max_tokens, extra=None) -> TeacherResult:
        ...
```

`TeacherResult` stores `model`, `finish_reason`, `usage`, raw text, think-block
if present. That metadata is written onto `verification.teacher`.

`routing.py`:

| record_type | model env | think |
|---|---|---|
| exam, implementation, agentic, critique | `TEACHER_REASONING` (deepseek-v4-flash) | on |
| analysis, memo, grounded | `TEACHER_PROSE` (same or qwen flash-next) | on or off after bake-off |
| abstention | `TEACHER_REASONING` | off |
| preference rejected | same as parent type | off |

One client, two model names, OpenAI-v1 only. No SDK per vendor.

### 5.5 Renderers

**Exam** — pack in, local trace builder out. Optional teacher rewrite of the
trace with `allowed_numbers` locked and liturgy cap 25%. Numbers still from pack.

**Prose** — system + brief from `teacher/prompts.py`. Repair loop: if
`invented_numbers` or `must_mention` fail, one repair call with the violations
listed, then dead-letter.

**Agentic** — state machine, not a filled conversation template:

```
user goal (from pack.question + scenario wrapper)
loop up to max_turns:
    teacher proposes assistant turn (text and/or tool_calls)
    validate tool_calls against cosimo.tools.registry
    oracle executes with optional injected fault
    append tool role
final assistant must only use numbers in pack ∪ tool results
```

Fault schedule comes from `oracle/faults.py` and the inventory (some families
are “clean tools”, some are “dirty”). 15–20% of agentic jobs are `no_call`.

**Implementation** — reference function + hidden tests in the pack module.
Teacher writes `limitations` only. Row dies if hidden tests fail.

### 5.6 Preference

After a supervised row is verified:

- probability `p_pref` by type (analysis 0.25, abstention 0.40, …)
- paraphrase `chosen` with the same pack (temperature 0.7)
- rejected under a named pitfall
- both sides pass structural gates; rejected must fail the pitfall detector
- write to `shards/v3/preference/`, never embed on the SFT row

### 5.7 Shards

```
shards/v3/supervised/{work_type}/{record_type}/shard_XXXX.jsonl
shards/v3/preference/shard_XXXX.jsonl
shards/v3/fact_packs/{work_type}.jsonl
shards/v3/dead_letter/{reason}/shard_XXXX.jsonl
```

Same atomic temp→rename as `core.append_record`. Shard size 500.

---

## 6. Verification

`verify_v3.py` is the publish gate. Axes:

1. Schema / required fields by `record_type`
2. Fact-pack recompute from seed
3. Invented-number subset check (`nums.py` + whitelist)
4. `must_mention` / `forbidden_claims`
5. `FINAL ANSWER:` exam-only
6. exam-liturgy share across the exam slice ≤ 0.25
7. family share ≤ 0.03
8. exam share in [0.12, 0.18]
9. tool schemas ⊆ `cosimo.tools.registry` and conversation roles valid
10. implementation hidden tests
11. preference id prefix disjoint + chosen ≉ SFT target (shingle overlap)
12. gold_bar_v3 near-dup (minhash) against train
13. teacher model pinned in metadata; mixed-teacher run must be intentional

`--quick` skips 10–12. CI runs `--quick` on a smoke inventory (one variant per
family). Full gate runs before `push_to_hub`.

---

## 7. Airflow (optional wrapper)

DAG `cosimo_v3_corpus` in `ops/airflow/dags/`. Each task is a CLI:

```
python -m dataset.pipelines.v3.cli inventory --out shards/v3/plan.json
python -m dataset.pipelines.v3.cli packs --plan shards/v3/plan.json
python -m dataset.pipelines.v3.cli render --types analysis,memo --limit 200
python -m dataset.pipelines.v3.cli verify
python -m dataset.pipelines.v3.cli prefer
python -m dataset.pipelines.v3.cli publish --dry-run
```

Airflow mapped tasks: one task per `work_type` so a bad computer does not block
risk while valuation renders.

Pools:

- `teacher` — sized to RPM
- `cpu` — packs, verify, prefer

XCom carries paths only. Completions stay on disk.

Local without Airflow:

```
make -C dataset v3-smoke
# or
python -m dataset.pipelines.v3.cli smoke
```

CI (`.github/workflows/corpus-v3.yml`): smoke inventory + verify --quick.
No teacher key in GitHub if you do not want billed CI; mock the teacher with
fixtures under `dataset/tests/v3/fixtures/`.

---

## 8. Harness changes (minimal, staged)

The 00–09 script numbers stay. You change config and parsers, not the DAG of
training.

### 8.1 `configs/base.yaml`

```yaml
model:
  base_id: Qwen/Qwen3.8-27B
  revision: "<pin SHA>"
  max_seq_length: 8192          # 16384 later; start 8k on Spark
  load_in_4bit: true            # QLoRA default on 128 GB if seq long
  dtype: bfloat16

dataset:
  hub_id: btech-software/cosimo-quant-assistant-v3
  revision: "<pin SHA>"
  preference_config: preference
  mix: []                       # no v1

prompt:
  identity: |                   # shorten vs 2494 chars
    You are Cosimo, a quantitative finance assistant at Btech Software.
    Prefer ranges when the inputs do not identify a point. Do not invent
    prints, filings, or tickers. Use tools when a number must be retrieved
    or computed. Never use exam liturgy unless the user asked an exam item.
  exam_protocol: ...            # unchanged, exam rows only
```

~~Archive the Phi-4 block as `configs/base.phi4.yaml` so old runs replay.~~ Done in
PR5, then dropped: no v2 model was ever pushed, so nothing replayed. See decision
log #7.

### 8.2 Chat template

Replace `configs/chat_template.jinja` with the Qwen3.8 instruct template that
preserves `<think>` and tool-call tokens. `cosimo_ft/chat.py` must not strip
think blocks on `exam`/`analysis` if the student is meant to think; strip them
from the *supervised target* for `abstention` and short desk replies so you do
not train 2k tokens of rumination on “what’s missing?”.

Policy:

| record_type | keep teacher/student think in target? |
|---|---|
| exam, analysis, memo, critique, implementation | yes, or a short think |
| abstention, grounded (short) | no |
| agentic | no think inside tool-call turns; optional think before first call |

### 8.3 `01_prepare_data.py`

- Parse `fact_pack` JSON.
- Split holdout on `scenario_id` prefix (family), not stem wrappers.
- Drop mix-with-v1.
- Map `memo`/`critique`/`grounded` onto the existing assistant-eval buckets
  (new suite files under `jobs/fine-tune/suites/`).
- Preference: already disjoint ids; keep the “chosen not in SFT” check.

### 8.4 `09_assistant_eval.py`

Add:

- `invented_number_rate` against eval fact packs
- `register_match`
- `must_mention_hit_rate`

Keep `exam_shape_rate` and `mean_new_tokens`. Those are the collapse detectors.

### 8.5 Tools

`jobs/fine-tune/cosimo_ft/tools.py` becomes:

```python
from cosimo.tools.registry import SCHEMAS, render_tools
```

`02_prepare_tool_data.py` can stay for synthetic extra tool rows, but v3 agentic
rows already carry real schemas. Do not generate a second incompatible dialect.

---

## 9. Docker / hardware mapping

| Job | Where | Image |
|---|---|---|
| fact packs, verify, prefer | any CPU | `docker/corpus` |
| teacher render | CPU + network | `docker/corpus`, env `TEACHER_*` |
| SFT / DPO / eval / merge | DGX Spark | `docker/fine-tune` (existing NGC + Unsloth) |
| serve | Spark or a bigger box | `docker/serve` vLLM |

Do not put `openai` into the Spark image unless you must. Do not put `torch`
into the corpus image.

`.env.example` gains:

```
TEACHER_BASE_URL=
TEACHER_API_KEY=
TEACHER_REASONING=deepseek-v4-flash
TEACHER_PROSE=deepseek-v4-flash
TEACHER_TIMEOUT_S=120
COSIMO_V3_OUT=dataset/shards/v3
```

---

## 10. Implementation sequence (repo PRs)

Each PR is one AGENTS.md unit: disjoint writes, a command that must pass.

### PR0 — contracts (no teacher, no Hub)

- `cosimo/tools/{schemas,registry}.py` extracted from current `tools.py`
- `jobs/fine-tune/cosimo_ft/tools.py` re-exports
- test: schema snapshots equal
- `dataset/taxonomy/work_types.yaml` with 5 types × families
- `dataset/pipelines/v3/packs/base.py` + `valuation_fcff.py`
- `dataset/tests/v3/test_fcff_pack.py` — recompute idempotent
- `docs/V3.md` pointer + Part III stub in `ARCHITECTURE.md`

Gate: `pytest dataset/tests/v3 jobs/fine-tune/tests -k tools`

### PR1 — CLI skeleton + verify numbers

- `teacher/client.py` + `prompts.py` (no live call in CI)
- `verify/invented_numbers.py` + tests from the smoke pack
- `cli.py inventory|packs|smoke`
- `shards/v3/` gitignore
- fixture teacher that echoes pack numbers

Gate: `python -m dataset.pipelines.v3.cli smoke`

### PR2 — prose renderer + dead letter

- `render/prose.py` repair loop
- write shards
- `verify_v3.py` axes 1–5
- live teacher behind `COSIMO_V3_LIVE=1`

Gate: 50 analysis rows, invented-number rate 0 on the dummy + live if keyed

### PR3 — agentic loop

- `oracle/runtime.py` + `faults.py` using `cosimo.tools`
- `render/agentic.py`
- conversation schema test vs serving parser

Gate: 20 conversations, 4 with faults, 4 no-call; every final number grounded

### PR4 — exam / impl / prefer / publish

- exam renderer with liturgy cap
- implementation hidden tests
- `prefer.py`
- `publish/dataset_card_v3.md` + `--corpus v3`
- Airflow DAG that shells the CLI

Gate: `verify_v3.py` on a 500-row slice; publish dry-run

### PR5 — harness student switch

- `base.yaml` → Qwen3.8-27B + v3 hub (or local jsonl override for smoke)
- chat template
- `01_prepare_data.py` family holdout
- `09_assistant_eval.py` new metrics
- ~~`base.phi4.yaml` archive~~ (dropped -- decision log #7)

Gate: `04_train_sft.py --dry-run` and a 200-row LoRA smoke on Spark

### PR6 — first real publish + smoke LoRA

- 2k–5k high-quality rows (the “read 100” milestone)
- pin Hub SHA
- smoke LoRA, quote `exam_shape_rate`, `mean_new_tokens`, invented-number rate
- only then scale inventory toward 45k

Do not open PR6 until someone has read 100 prose rows. That is the actual
quality gate. Everything before it is plumbing.

---

## 11. Testing strategy

```
dataset/tests/v3/
  test_pack_recompute.py
  test_inventory_caps.py
  test_invented_numbers.py
  test_agentic_oracle.py
  test_prefer_disjoint.py
  fixtures/teacher_echo.json
jobs/fine-tune/tests/
  test_tools_reexport.py
  test_prepare_v3_schema.py
  test_chat_template_qwen.py
```

No live teacher in default pytest. Mark `@pytest.mark.live` for the keyed path.

---

## 12. Failure modes this architecture is designed to prevent

| Old failure | Where it is now blocked |
|---|---|
| 71 stems × 1000 paints | inventory family cap + holdout by family |
| Style collapse | exam share cap + liturgy cap + short identity + assistant eval |
| DPO loss 0.0 | disjoint pref ids + paraphrased chosen |
| Analysis f-strings as judgement | teacher + invented-number gate |
| Agentic prompt collapse | loop + oracle, not a filled transcript |
| Tool schema drift | `cosimo.tools` single registry |
| v1 mix reintroduces collapse | `mix: []` |
| CWD-dependent publish | v3 CLI takes `--out` absolute paths |
| Teacher swap mid-run | model name in every row; verify axis 13 |
| Spark image polluted with ETL | two Dockerfiles |

---

## 13. Makefile surface (repo root or `dataset/Makefile`)

```
v3-inventory   python -m dataset.pipelines.v3.cli inventory
v3-packs       python -m dataset.pipelines.v3.cli packs
v3-smoke       python -m dataset.pipelines.v3.cli smoke
v3-render      python -m dataset.pipelines.v3.cli render
v3-verify      python -m dataset.pipelines.v3.cli verify
v3-prefer      python -m dataset.pipelines.v3.cli prefer
v3-publish     python -m dataset.pipelines.v3.cli publish
```

Fine-tune stays `docker/fine-tune/run.sh` + `run_all.sh`. Do not fold training
into Airflow on day one. One orchestrator per subsystem.

---

## 14. Decision log (so the next agent does not reopen them)

1. v3 is a new backend under `dataset/pipelines/v3/`, not a rewrite of `generate.py`.
2. Hub id is a new repo, not a config split on v2.
3. Airflow wraps CLI; CLI does not import Airflow.
4. Shared code lives in `cosimo/`, not in `dataset`↔`jobs` imports.
5. Student is Qwen3.8-27B; teacher is DeepSeek-V4-Flash unless the memo bake-off says otherwise.
6. First publish is thousands of good rows, not 113k.
7. ~~Phi-4 configs remain as `base.phi4.yaml` for replay of published v2 runs.~~
   **Reversed.** No v2 model was ever pushed, so there was no published run to
   replay and the archive guarded nothing. Phi-4 support is dropped: a 3.8B
   student cannot reach the analytical standard the corpus is built for (the
   teacher spends 16k-49k completion tokens to write 150 words of it). The
   harness stays model-agnostic -- there were no Phi-specific code branches to
   remove, only two config files -- so a different student remains a config
   change. The limit is capacity, not architecture, which is why a mid-size
   Qwen3 is still a legitimate fallback if the 27B proves impractical.
