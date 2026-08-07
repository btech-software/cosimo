Generate the second Cosimo training corpus: a **complementary** dataset that teaches the reasoning and judgement the first one could not, designed to be **mixed** with `btech-software/cosimo-cfa-frm-71k` rather than to replace it.

This supersedes `001_dataset_prompt.md`. That brief was executed, the corpus it produced was trained on, and the run was measured end-to-end. Read the findings below before designing anything: they are not predictions, they are results, and every requirement here exists because something specific went wrong.

**The objective is the one in `jobs/fine-tune/README.md`:** an assistant to a Head of Quantitative Asset Management — one that reasons about valuation, risk, market microstructure and research papers, and is honest about what it does not know. Exam accuracy is a milestone, not the target. This corpus is one increment in a planned series; it does not have to teach everything, but it must not repeat what the first one got wrong.

---

## What the first corpus got right — keep all of it

- **Every number is computed, never written.** Templates compute, then stringify; verification re-executes from the stored seed and compares. Keep this absolutely.
- **Deterministic, resumable, idempotent generation** keyed by `(program, template, variant)`. Generation runs for days; it must survive restarts.
- **Content-hashed ids, atomic shard writes, a 4-axis regression gate.** All good engineering. Reuse it.
- **Original content only**, inspired by public Learning Outcome Statements. Never reproduce proprietary exam material.

## What it got wrong — all measured, all must be fixed

1. **71 generator stems produced 71 000 rows.** Scale came from randomising numbers inside a fixed set of templates, so the model memorised phrasing skeletons. Measured: in-domain accuracy **62.9 %** against **18.0 %** on held-out stem families — a 45-point gap that is memorisation, not capability.

2. **The corpus covers 13.6 % of its own taxonomy.** Only **25 of 184** declared subtopics have any generator at all:

   | Program | Subtopics | Covered | Gap |
   | --- | ---: | ---: | ---: |
   | CFA Level I | 58 | 12 | 46 |
   | CFA Level II | 55 | 8 | 47 |
   | CFA Level III | 45 | 5 | 40 |
   | FRM Part 1 | 21 | 4 | 17 |
   | FRM Part 2 | 28 | 2 | 26 |

   The taxonomy was written and then largely ignored. **Closing this gap is the single highest-value thing this corpus does.**

3. **Every supervised target was a terse formulaic trace** — `ASSUMPTIONS:` / `Step 1.` / `FINAL ANSWER:`, 150–400 tokens. The base model is a long chain-of-thought reasoner; training on this compressed mean response length from **~750 tokens to 120** and taught the model that *being Cosimo means answering in four steps*. Asked afterwards to walk through hedging a convexity mismatch, the served checkpoint produced an exam trace and invented a term — "**Durbin-Watson duration**" — that does not exist.

4. **Preference pairs were unusable for preference learning.** The corpus `reasoning_trace` *is* the `chosen` side, and SFT trained on those same rows: **22 048 of 22 048 pairs overlapped**, a 100 % collision. By the time DPO started the reward margin was hundreds of nats, the sigmoid saturated, and the loss was exactly `0.0` from step 10. Five GPU-hours moved the adapter 0.16 % and changed no metric.

5. **The pairs differ by one arithmetic slip.** `chosen` and `rejected` share a skeleton and diverge on a wrong-formula branch. That teaches "do not flip this sign". It does not teach judgement.

6. **Nothing in the corpus teaches abstention.** Every target is a confident computation. The persona claims to be "brutally honest about what you don't know" while 100 % of the training signal says *always answer*.

7. **Tool data was one round-trip, always.** No chains, no parallel calls, no recovery from a failed call.

8. **The persona advertises capabilities with zero training data**: deriving Black-Scholes from Itô, game theory (Nash, Bayesian games, mechanism design, Shapley), reading a paper and producing idiomatic Python. None of it is in the corpus and none of it is measured.

---

## Goal

A new corpus, generated under `dataset/` in this repository, that mixes with the existing 71k to produce an SFT/preference set with roughly this composition. These are targets to design against, not a formula to satisfy exactly:

| Component | Share of the mixed SFT set | Why |
| --- | ---: | --- |
| Exam items — **existing 71k, capped** | ≤ 30 % | Verified arithmetic is real value. It stops being useful the moment it dominates. |
| Exam items — **new subtopics** | ~15 % | Closes the 159-subtopic taxonomy gap. New stems, not new numbers in old stems. |
| Open-ended analysis | ~25 % | The actual job. Long-form, prose, no single number. |
| Abstention and calibration | ~10 % | Underspecified, unanswerable, false-premise. Correct answer is to ask or refuse. |
| Agentic multi-step tools | ~12 % | Chains of 2–4 calls, parallel calls, failed-call recovery, and no-call-appropriate. |
| Paper → implementation | ~8 % | Extract a model from a described paper, produce clean Python, state what breaks in production. |

**Scale target is stems and subtopics, not rows.** A corpus of 30 000 rows from 400 distinct generators across 150 subtopics is worth far more than 200 000 rows from 71. If you must choose, choose breadth.

### Response-length distribution is a first-class property

Record the token-length distribution of the supervised targets and treat it as a design constraint, not an outcome. The mixed corpus must contain a **substantial fraction of long-form targets (800+ tokens)**, because the base model can already reason at length and the first corpus trained that out of it. A corpus whose p95 target length is under 400 tokens has already failed, whatever else it scores.

---

## Record types and schema

Extend `FORMAT.md` rather than inventing a parallel format. Every record keeps `id`, `program`, `topic`, `subtopic`, `difficulty`, `question_type`, `verified`, `verification`, `metadata`, and adds a **`record_type`** discriminator so the fine-tuning harness can mix and weight them:

- `exam` — as today: question, distractors, computed answer, reasoning trace.
- `analysis` — open-ended question, long-form answer. No `FINAL ANSWER:` line, no mandatory step enumeration.
- `abstention` — a defective prompt plus the response that identifies what is missing. `metadata.defect` is one of `underspecified` / `unanswerable` / `false_premise`.
- `agentic` — a full multi-turn conversation with tool schemas, calls, results and a final answer.
- `implementation` — a described model or paper, plus an idiomatic object-oriented Python implementation and an honest note on what fails with real data.

Format constraints that follow directly from the measured failures:

- **`FINAL ANSWER:` appears only on `exam` records.** It is a grading contract, not a house style. Putting it everywhere is what taught the model to answer every question as an exam item.
- **`ASSUMPTIONS:` and `Step N.` must not be universal.** Vary the shape of exam traces deliberately: some prose, some enumerated, some tabular, some working backwards from the answer. A model cannot learn that structure is a *choice* if it only ever sees one structure.
- **No record type may be produced by a single template.** Each needs many independent generators.

### Preference pairs — rebuilt

- Pairs must live in a **disjoint id space from the SFT rows**, or carry an explicit flag the preparation step can split on. The harness now enforces zero overlap (`data.preference_holdout_frac`, and a validation gate that fails on collision), but a corpus that ships colliding pairs is still a corpus half of which cannot be used.
- **`rejected` must be plausible and wrong for an interesting reason.** Target: right method with a wrong assumption; a confident answer to an underspecified question; a correct calculation that answers a different question than the one asked; a fluent answer with an invented term. Retire "same skeleton, one flipped sign" as the dominant mode.
- Cover every `record_type`, not just `exam`. The most valuable pairs are `abstention` ones: *asked for the missing input* versus *answered anyway*.

---

## Concrete quality bar (non-negotiable)

The gold bar is **no longer a set of exam questions**. That was the mistake baked into `001`: the bar defined the target, the target was exam items, and the model became an exam solver.

Build a gold bar of **~200 assistant transcripts** — what an excellent quant assistant actually says to a Head of Quantitative Asset Management. It must contain long-form analysis, honest uncertainty, tool use, and code, alongside exam items. Sources: public CFA/FRM Learning Outcome Statements, high-quality open financial education material, published papers and their reference implementations, and rigorously verified synthetic exemplars.

Every batch must win a **blind A/B against that bar** on:

1. **Correctness** — numerical and conceptual.
2. **Reasoning depth** — does it explain *why*, or only *what*.
3. **Calibration** — does it state its assumptions and flag what it cannot know.
4. **Format appropriateness** — does the shape of the answer fit the question asked.
5. **Terminology validity** — every technical term is real and used correctly.
6. **Absence of hallucination.**

The finished corpus must additionally win on **subtopic coverage**, **generator diversity**, and **response-length distribution**. A batch that wins on correctness and loses on format appropriateness has failed; that combination is exactly what shipped last time.

---

## Verification — three gates, all mandatory

1. **Code execution for every numeric.** Recompute from the stored seed, compare to the persisted answer, drop on mismatch. This is the one thing the first pipeline got unambiguously right.
2. **Terminology gate.** Check every technical term against a curated finance vocabulary. Invented collocations — a real eponym welded to the wrong concept, as in "Durbin-Watson duration" — must **block** the record. Note that the eponym and the concept are individually valid; it is the pairing that is fabricated, so a token-level check is insufficient.
3. **Frontier-model critic with an explicit rubric.** Judgement-bearing content has no computable ground truth. The critic scores against the six axes above and performs the blind A/B. It must inspect real files and real samples, never summaries.

Verification must be **re-runnable over the whole corpus** as a regression gate, like the existing `verification/verify_all.py`, and must exit non-zero on any failure.

---

## Anti-patterns — do not do these

- **Do not scale by re-randomising existing stems.** Row count is not the goal; the 45-point generalisation gap is what that buys.
- **Do not attach `FINAL ANSWER:` to non-exam records.**
- **Do not generate preference pairs whose `chosen` is also an SFT target.**
- **Do not let one response shape dominate.** Uniform formatting is the failure mode, not the quality signal.
- **Do not overlap the assistant-eval suites.** `jobs/fine-tune/suites/{open_ended,calibration,agentic}.jsonl` are held-out measurement instruments. Generating near-duplicates of those prompts contaminates the only evaluation that measures the objective. Check for overlap explicitly and record the check.
- **Do not target `microsoft/Phi-4-mini-flash-reasoning`.** `001` named it; it is a `Phi4FlashForCausalLM` SambaY hybrid, not trainable with Unsloth/PEFT. The training target is **`unsloth/Phi-4-mini-reasoning`**.

---

## Deliverables

Everything reproducible lives under `dataset/` relative to the repository root, following the existing layout and its contracts (`FORMAT.md`, `ARCHITECTURE.md`, `AGENTS.md`):

- `dataset/taxonomy/` — extended coverage map, with the covered/uncovered status of every subtopic tracked as data, not prose.
- `dataset/pipelines/templates/` — the new generators, registered per program, following the existing `fn(rng, seq) -> dict` contract.
- `dataset/goldbar/` — the assistant-transcript gold bar.
- `dataset/verification/` — all three gates, plus a full-scan regression runner.
- `dataset/shards/` — output JSONL, sharded, atomic, resumable (gitignored).
- `dataset/publish/` — Hub publishing script and dataset card. The card must state the composition table, the coverage numbers, and the known limitations honestly.
- A live progress page showing: rows by `record_type`, **subtopic coverage heat-map**, distinct generator count, response-length distribution, gold-bar A/B win rate per axis, current bottlenecks, and shard status.

---

## Instructions for the lead agent

- Divide the goal into the smallest pieces that can be independently built and judged: taxonomy gap analysis, per-record-type generation pipelines, each verification gate, gold-bar construction, preference-pair construction, coverage and diversity trackers, storage format, publishing.
- For every important piece, fan out a builder subagent and a **completely separate** critic subagent in fresh contexts.
- Each critic inspects real output — files, numbers, samples — performs the blind A/B against the gold bar, identifies the single largest remaining gap, and sends the work back. Loop until it wins or improvements become negligible.
- Build the **gold bar first**, then the **taxonomy gap analysis**, then generation. The bar defines the target; last time the target was wrong and everything downstream faithfully hit it.
- Prefer incremental, resumable generation. Write all artifacts to disk. Do not run training.
- Do not prescribe architecture, exact decomposition, or a fixed number of rounds. Decide the best technical approach yourself.

**Measure before you declare done.** The corpus is finished when it wins the bar on all six axes at scale, covers a large majority of the taxonomy, draws from hundreds of distinct generators, and carries a response-length distribution that does not collapse the base model's ability to think at length using chain-of-thought.

Start now.
