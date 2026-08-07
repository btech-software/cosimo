# Cosimo v2 — synthetic financial reasoning corpus

`{{REPO_ID}}` — **{{TOTAL}}** records across six record types, generated and
verified by code. Built as a **complement** to
[`btech-software/cosimo-cfa-frm-71k`](https://huggingface.co/datasets/btech-software/cosimo-cfa-frm-71k),
designed to be mixed with it (the composition targets below are shares of the
*mixed* set, in which the 71k corpus is capped at ~30%).

The objective is not a model that passes CFA/FRM exams. It is an assistant to a
Head of Quantitative Asset Management — one that reasons about valuation, risk
and market microstructure, and is honest about what it does not know. The first
corpus taught exam arithmetic and, measurably, little else: 62.9% in-domain
accuracy against 18.0% on held-out stem families, and a served checkpoint that
compressed its answers to four-step exam traces. Every design decision below is
downstream of a measured failure.

## Composition

| record_type | rows | share of this corpus |
|---|---:|---:|
{{TYPE_TABLE}}

| split (config `default`) | rows |
|---|---:|
{{PROGRAM_TABLE}}

**{{GENERATORS}}** distinct generators. Taxonomy coverage: **{{COVERAGE}}**
declared subtopics ({{COVERAGE_PCT}}%) appear in generated records.

## Record types

- **`exam`** — computed question, distractors, verified answer, reasoning trace.
  `FINAL ANSWER:` appears **only** here: it is a grading contract, not a house
  style.
- **`analysis`** — open-ended question with a long-form answer (no `FINAL
  ANSWER:`, no mandatory step enumeration). The long-form body is composed
  per-record from topic-keyed paragraph builders that each compute their own
  illustrative numbers from the record's seeded RNG.
- **`abstention`** — an underspecified / unanswerable / false-premise prompt
  plus the response that identifies what is missing (`metadata.defect` carries
  the defect class).
- **`agentic`** — a multi-turn conversation with tool schemas, Hermes-format
  tool calls, tool results and a final answer. Roles are
  `user`/`assistant`/`tool`.
- **`implementation`** — a modelling task plus idiomatic Python that parses and
  executes (verified at generation time), with test code.
- **`preference`** (config `preference`, {{PREF_ROWS}} rows) — standalone
  chosen/rejected pairs in the `cosimopref_` id namespace, **disjoint from every
  supervised id by construction**. Four failure modes:

| mode | rows |
|---|---:|
{{MODE_TABLE}}

## Verification

Eleven gates run over the full corpus before publishing
(`verification/verify_all.py` in the source repository); the publish script
refuses to push on any failure:

structure · numeric recomputation from stored seeds · format (`FINAL ANSWER:`
exam-only) · implementation code execution · agentic structure · preference
id-space disjointness · terminology (eponym/concept collocation) · template
artifacts · question duplication and boilerplate share · response-length
distribution · eval-suite overlap.

Every numeric answer is computed by code, never written; verification
re-executes each exam record from its stored `(template, seed)` and compares.

## Schema notes

`conversation`, `tool_schemas`, `verification` and `metadata` are JSON-encoded
strings: their key sets differ per record type, and a stable schema across all
splits was preferred over nested structs. Parse with `json.loads` where
non-empty.

Exam records in the source pipeline carry an embedded `preference_pair` whose
`chosen` side is the record's own reasoning trace. That field is **deliberately
not published**: training SFT on a pair's chosen side pre-saturates the DPO
margin (measured: loss exactly 0.0 from step 10, five GPU-hours for a 0.16%
adapter change). Use the `preference` config, whose ids no supervised row
shares.

## Known limitations

1. **Synthetic.** Generated from templates and verified by recomputation, not
   written by charterholders. Judgement-bearing prose is composed from
   parameterized paragraph builders; it is topically keyed and numerically
   grounded, but it inherits the perspectives of its generators.
2. **Coverage is partial.** {{COVERAGE}} declared subtopics are covered; the
   gap is tracked as data in the source repository and closing it is the
   highest-value ongoing work.
3. **Agentic conversations are short** (single-digit turns) and tool results
   are mocked. They teach the wire format and call selection, not judgement
   about when a tool is the right analytical move.
4. **Abstention answers share a structural skeleton** (identify the gaps, list
   them, ask) with varied framing. That is partly inherent to the type.
5. **Original content only.** Inspired by public Learning Outcome Statements;
   no proprietary exam material is reproduced.

## License

MIT. No real client, position or counterparty data; every number is
synthetically generated.
