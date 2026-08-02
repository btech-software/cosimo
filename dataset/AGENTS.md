# AGENTS.md — Working in this repo

Operational guidance for AI agents (and humans) modifying or running the Cosimo
pipeline. For the component map see [`ARCHITECTURE.md`](ARCHITECTURE.md); for the
record schema see [`FORMAT.md`](FORMAT.md).

---

## 1. Repo layout at a glance

```
2026-08-01/
├── FORMAT.md                     # record schema (authoritative)
├── README.md                     # dataset overview (consumers + devs)
├── ARCHITECTURE.md               # component map + code refs
├── AGENTS.md                     # this file
├── config/seed.json              # generation seed config
├── goldbar/gold_bar.jsonl        # curated gold-bar exemplars
├── pipelines/
│   ├── core.py                   # IDs, RNG, formatting
│   ├── generate.py               # generator driver
│   ├── progress.py               # live progress report
│   └── templates/                # 71 question stems
│       ├── wrappers.py           # vignette / CR / MCQ wrappers
│       ├── cfa_l1.py (33) cfa_l2.py (12) cfa_l3.py (9)
│       ├── frm1.py (10) frm2.py (7)
├── progress/progress.md          # live report (has honest gaps)
├── shards/                       # output: <program>/<program>_shard_XXXX.jsonl (110 shards)
├── taxonomy/taxonomy.json        # curriculum taxonomy
├── verification/
│   ├── verify_all.py             # 4-axis regression gate (must stay green)
│   ├── run_verify.py             # independent live harness
│   ├── fix_distractors.py        # distractor dedup fix
│   ├── sanitize_distractors.py   # distractor magnitude sanitizer
│   └── nums.py                   # robust numeric tokenizer
└── eval/
    ├── ab_eval.py                # gold-bar A/B
    └── diversity.py              # structural novelty report
```

## 2. Hard rules / conventions

1. **Run everything from the repository root.** The scripts do absolute
   `sys.path.insert(...)` and glob relative `shards/`; running from elsewhere
   breaks verification.
2. **Never hand-edit shard records.** Records are derived artifacts — regenerate
   via the pipeline. If you must repair data, use the deterministic fix scripts
   (`fix_distractors.py`, `sanitize_distractors.py`), which preserve answer/trace
   integrity.
3. **Generation is deterministic.** Seed per `(program, template, variant)`:
   `seed = (1000 * hash((program, template)) % 10**9 + variant * 7919) % 2**31`.
   Do not introduce randomness; keep `seq` reproducible
   (`100000 + variant*1000 + abs(hash % 1000)`).
4. **Shards are append-only + atomic.** `append_record(..., finalize=False)`
   writes a temp file; finalize renames at program end. Don't write shards inline.
5. **Preference pairs are gated at ~35%.** A pair is emitted only when a template
   returns `flawed` (`{"answer", "reasoning_trace", "pitfall"}`) **and**
   `rng.r.random() < PAIR_RATIO` (from `config/seed.json`
   `preference_pair_ratio`, default 0.35). `preference.py` is legacy and not
   wired in — prefer inline, and see the checklist item below.
6. **Deterministic finalize.** `generate.py` rewrites (`_dedup_distractors`,
   `_dedup_wrong`) any stored distractor or flawed wrong answer numerically equal
   to the correct answer (preserving `$`/`%` formatting). It never touches
   question/answer/trace. `run_verify.py` reproduces this finalize when checking
   stored pairs.
7. **Keep `verify_all.py` green.** It is the regression gate; any generator change
   requires re-running it (exit 0 == PASS, exit 1 == FAIL).

## 3. Commands

```bash
# Generate / extend (deterministic, resumable)
python3 pipelines/generate.py

# Filters
PER_TEMPLATE=50 python3 pipelines/generate.py
PROGRAM=CFA_Level_II python3 pipelines/generate.py
TEMPLATE=eq_gordon python3 pipelines/generate.py

# Verification (must pass after any change)
python3 verification/verify_all.py

# Independent live harness + progress refresh
python3 verification/run_verify.py

# Progress report only
python3 pipelines/progress.py

# Quality fixes (already applied; deterministic)
python3 verification/fix_distractors.py
python3 verification/sanitize_distractors.py

# Evaluation
python3 eval/ab_eval.py
python3 eval/diversity.py
```

## 4. Adding a new question stem

1. Add a template function to the relevant `pipelines/templates/<program>.py`
   matching the contract (see `ARCHITECTURE.md` §3.2): `meta`, `question`,
   `answer`, `distractors`, `reasoning_trace`, and optional `flawed`.
2. Register it in that module's `TEMPLATES` dict (stem name → fn).
3. Run `python3 pipelines/generate.py` (optionally with `TEMPLATE=<stem>`).
4. **Run `python3 verification/verify_all.py`** — it must exit 0.
5. Update `progress/` and (if counts/coverage change) `README.md` composition table.

## 5. Gotchas (check these before editing)

- **Dead code removed**: `preference.py`, `cfa_l1_a.py`, and `cfa_l1_b.py` were deleted.
- **Preference-ratio resolved**: pairs are gated at ~35% via
  `config/seed.json` `preference_pair_ratio`; the deterministic finalize keeps
  every emitted pair concrete (wrong ≠ correct).
- **Absolute-path coupling**: verification/generate scripts assume repo-root CWD.
- **Novelty bound**: 71 distinct stems exist; within-stem records differ
  only by sampled numbers. Diversity gains require new templates.
- **Question types**: `Calculation`, `Vignette`, `Constructed Response`, and `MCQ`
  (wrappers in `pipelines/templates/wrappers.py`).
- **Gold-bar overlap**: `goldbar/gold_bar.jsonl` overlaps shards, so
  `eval/ab_eval.py` is calibration-vs-gold, not an independent oracle.

## 6. Open-items checklist

Track these; mark resolved when fixed:

- [x] Adopt or remove orphaned `pipelines/preference.py` — removed.
- [x] Clean up dead templates `cfa_l1_a.py` / `cfa_l1_b.py` — deleted.
- [x] Add an MCQ question-type wrapper — `wrap_mcq` added to
  `pipelines/templates/wrappers.py`.
- [x] Grow template count beyond 55 stems — now 71 stems (16 added).
