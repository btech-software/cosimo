# Cosimo v1 / v2 audit and v3 specification

**Repos / artifacts reviewed**
- GitHub: [btech-software/cosimo](https://github.com/btech-software/cosimo)
- Hub: [btech-software/cosimo-cfa-frm-71k](https://huggingface.co/datasets/btech-software/cosimo-cfa-frm-71k) (v1)
- Hub: [btech-software/cosimo-quant-reasoning-v2](https://huggingface.co/datasets/btech-software/cosimo-quant-reasoning-v2) (v2)
- Source of truth in-repo: `dataset/README.md`, `dataset/FORMAT.md`, `dataset/publish/dataset_card.md`, `ARCHITECTURE.md`, `jobs/fine-tune/README.md`, and the `dataset/pipelines/templates/v2_*.py` generators.

**Goal stated by the project (and this brief)**  
Produce a fine-tuning corpus for a *quant assistant* — an aide to a Head of Quantitative Asset Management — not an exam-passer. v1 failed that goal in measurement. v2 correctly diagnosed the failure and still only half-solves it.

This document is the design for **Cosimo Quant Reasoning v3**: hybrid code-verified facts + teacher-LLM prose, produced by an Airflow ETL that talks to any OpenAI-v1-compatible endpoint.

---

## 0. Verdict in one page

| | v1 `cosimo-cfa-frm-71k` | v2 `cosimo-quant-reasoning-v2` |
|---|---|---|
| What it actually is | 71k exam items from **71 stems**. Numbers change. The question does not. | 113,574 rows, five record types, 299 generators, 7,500 disjoint preference pairs. Still template-bound. |
| What it is good at | Numeric integrity. Reproducible seeds. Preference *pitfalls* that are concrete. | Admitting v1's failure in public. Mixing formats. Blocking `FINAL ANSWER:` leakage. Disjoint DPO IDs. |
| What it is bad at | Being an assistant corpus. Diversity. Preference hygiene. Dedup. | Being *judgement*. Coverage (72/207 subtopics). Agentic depth. Prose quality. Desk realism. |
| Measured harm | Style collapse: mean completion **~750 → 120 tokens**. In-domain **62.9%** vs held-out stems **18.0%**. DPO loss **0.0** when chosen ⊆ SFT. | Not yet measured on the mixed corpus (your own README says so). Structural gates only on non-exam rows. |
| Fit to "quant assistant" | Poor. It trains a five-step calculator that will Durbin-Watson a duration question. | Better shape, same soul: CFA/FRM taxonomy + Mad-libs paragraphs + mocked 3-turn tool chats. |

**v3 thesis.** Keep the one thing v1/v2 got right — *numbers are computed, never sampled from a language model* — and stop asking Python f-strings to write like a desk. Facts come from code. Voice, structure, trade-offs, and refusal quality come from a teacher model that is *not allowed to invent numbers*. Verification becomes two-phase: recompute + grounded-prose gates. Preference pairs come from contrastive generation against locked facts, plus harvested model failures, not four canned pitfall enums.

If you ship another 100k Mad-libs rows and call it v3, you will measure the same 18% on anything that is not a sibling of a training stem.

---

## 1. What you built that is actually good

Say this first, because the rest is harsh and the engineering is not amateur.

1. **You measured the failure.** Most dataset authors never do. `cosimo_test` 62.9% vs `cosimo_unseen_stems` 18.0%, plus the token-length collapse and `exam_shape_rate`, is the most valuable artifact in the repo. Design from that, not from row count.
2. **Numeric answers are code-backed.** `reference_code_exec` + stored seed + recompute is the correct primitive for finance SFT. Do not abandon it.
3. **Identity vs exam protocol split** in `jobs/fine-tune/configs/base.yaml` is the right idea. The ~600-token Cosimo block on every example is expensive, but binding persona off the exam contract is why v2 exists.
4. **Disjoint preference IDs in v2** (`cosimopref_`) after measuring DPO loss = 0.0 from step 10 is grown-up. v1 shared IDs; you fixed it.
5. **Append-only sharded generation with deterministic `(program, template, variant)` seeds** is a real ETL, not a notebook dump.
6. **The eleven-gate `verify_all.py` list** (structure, numeric, format, impl exec, agentic structure, pref disjointness, terminology, template artifacts, duplication, length, eval-suite overlap) is the right *skeleton* of a publish gate. It is just incomplete for prose and judgement.
7. **You already know coverage is 72/207 (34.8%)** and you already wrote that closing it is the highest-value work. Believe yourself.

None of that should be thrown away. v3 is a generation *and verification* upgrade, not a rewrite of the harness.

---

## 2. Brutal review of v1 — `cosimo-cfa-frm-71k`

### 2.1 What it is
71,000 supervised exam records + 24,711 nested preference pairs. CFA L1 33k / L2 12k / L3 9k / FRM1 10k / FRM2 7k. 71 templates, 142 shards of 500. Every number recomputed from seed.

### 2.2 The structural-novelty problem is fatal
The README is honest: *“Structural novelty is limited to 71 distinct stems; variation within stems comes only from sampled numbers.”*

That is not a dataset of 71,000 questions. It is a dataset of **71 questions with 1,000 coats of paint**. A 3.8B model will memorize the paint-removal algorithm (the stem) and look competent on `cosimo_test`. It will die on `cosimo_unseen_stems` (18%). That gap is not “the model is small.” It is the corpus teaching 71 tricks.

The Hub preview making this worse: the default viewer is a wall of `tvm_annuity_fv` — *“deposits $X at the END of each month … compounded monthly … future value after N months.”* Even if the other 70 stems exist, the published surface looks like one generator won the lottery. That is what a downstream user, and a tokenizer, will overweight.

### 2.3 Style collapse was not an accident. It was the loss surface.
Every target:
- `ASSUMPTIONS:`
- `Step 1.` `Step 2.` `Step 3.`
- `FINAL ANSWER:`

Train a small reasoning model on 63k of those and of course it becomes a four-step calculator. You measured it: mean new tokens 750 → 120, p95 179, format compliance 4% → 100%, including on GSM8K and MATH-500. The first run did not “fail to generalize finance.” It *succeeded* at imitating the only register it was shown.

The identity prompt saying “you are the main assistant for the Head of Quant at BlackRock / GS / JPM / Citadel” on top of that corpus is cosplay. The gradients never saw a CIO memo.

### 2.4 Preference pairs were unusable for the reason you later measured
v1 nested `preference_pair` on the same id as the SFT row, with `chosen` equal to the SFT target. DPO then sees chosen ≈ SFT completion and the margin saturates (loss 0.0, 0.16% adapter change after five GPU-hours). That is not a DPO corpus. It is a duplicate of SFT with a `rejected` sibling.

Rejected traces that “commit exactly one targeted pitfall” are a good *idea*. The implementation — generate the correct intermediates, then splice one wrong formula — produces rejected text that is often a near-copy of chosen with a different last number. That is a weak preference signal compared to “fluent, confident, wrong-framed analysis.”

### 2.5 Dedup and distractor hygiene
Your own harness notes **15,366 internal duplicate questions** in v1, left in place. Across v1∩v2, 1,840 exam questions were byte-identical; you dropped 2,683 rows at mix time rather than fixing the generator. That is operational debt becoming training noise.

The dataset card sample for v1 even shows distractors collapsing to the correct answer (`["2,626,704.14", "2,626,704.14", "2,626,704.14"]`). You later wrote `_dedup_distractors` that *adds 7.0 + i* when equality is detected. That is a bandage: a distractor of `correct + 8` is not a *finance* distractor (annuity-due, wrong compounding, ignoring intra-year). It is a number that will never be chosen by a model that has learned “pick the nearby float.” Pitfalls listed in metadata (`annuity due vs ordinary`) are not actually instantiated in those patched distractors.

### 2.6 Gold bar is not a test
`goldbar/gold_bar.jsonl` overlaps generated shards. A/B eval is calibration against your own templates. `suite_overlap.py` Jaccard > 0.6 is a leak fence for external suites, not an independent finance benchmark.

### 2.7 Curriculum ≠ job
v1 is a CFA/FRM item bank. A Head of Quant’s week is not 71k TVM/duration/BSM items. It is: messy data, a paper that overclaims, a PM who wants a number by 4pm, a risk limit that does not fit the model, a factor that flipped sign in the last regime, and a memo that must survive an IC. v1 contains none of that register.

---

## 3. Brutal review of v2 — `cosimo-quant-reasoning-v2`

v2 is the correct *political* document. The dataset card reads like a postmortem, which is rare and good. The corpus is still not a quant assistant.

### 3.1 Mix and shape — the part that works
113,574 supervised + 7,500 preference.

| type | rows | share |
|---|---:|---:|
| analysis | 40,591 | 35.7% |
| exam | 24,360 | 21.4% |
| agentic | 19,503 | 17.2% |
| abstention | 16,120 | 14.2% |
| implementation | 13,000 | 11.4% |

Default mix with v1 capped (~12% of the merged trainable pool) yields SFT train 125,939 at ~30% exam / 70% non-exam. That is the right *ratio* to fight style collapse. Ratio is not content.

### 3.2 Analysis records are Mad-libs with a calculator
`v2_analysis.py` (~2k lines) is the exhibit. `eq_fcff_dcf` draws `rev, g, wacc, o_margin, capex_p, nwc_p, tax, n`, computes FCFF/TV/EV and ±20% growth, then interpolates one paragraph:

> “Mature firm: revenue {rev}M, FCFF margin {m}. Growth {g} for {n} years then perpetuity. WACC {wacc}. … What EV results and how does a +/- perturbation of terminal growth change it?”

The “answer” is the same paragraph with the computed numbers dropped in, plus two canned sentences about terminal growth. That is not analysis. It is a worked example with a thesis glued on. Forty thousand of these teach:

- always narrate NOPAT → capex → NWC → FCFF0 → sum PV + TV
- always perturb terminal g by ±20%
- never discuss mid-year convention, fade, ROIC = WACC at terminal, maintenance vs growth capex inconsistency, or the fact that `FCFF0 = rev * o_margin * (1-t) - rev * capex_p + rev * nwc_p` **adds** NWC rather than subtracting an *increase* in NWC (the sign is a generator bug, and it will be faithfully learned)

I did not audit every function, but this pattern is the file. `fi_duration_convexity` computes a duration-like number off a confused coupon/YTM mix (`cpn_per = ytm/2` used as the coupon in the price sum). Template bugs become ground truth because verification for `analysis` is `method: structural`, not recomputation of the prose numbers against a spec.

**This is the central v2 failure:** you promoted the record type that should carry judgement, then generated the judgement with the same f-string machine that writes exam traces, and you do not re-check the numbers inside the paragraph against a schema of facts.

### 3.3 Agentic records teach the wire format and almost nothing else
Your card already says it: *single-digit turns, mocked tool results, teach format not judgement.* The source is blunter:

> `q` is a short fixed label on most generators; the conversation varies but the question did not, so rows collapsed to a handful of distinct prompts.

Then `core.scenario_clause` sprinkles a desk name on the question to fake entropy — the same trick used in abstention (`_DESKS` × `_OPENERS` × `_CLOSERS`). Surface variation is not task variation.

Concrete problems in `v2_agentic.py`:
- Tools are 2–3 toy functions (`get_fundamentals`, `compute_metrics`) with mocked JSON. No pagination, no vendor field chaos, no as-of date, no corporate action, no missing field.
- Conversations are 3–5 turns with the call sequence predetermined. The model never has to *choose not to call*, *call twice because the first result is stale*, or *reconcile two sources*.
- `agentic_eq_resid_income` hardcodes the user utterance (`BV0 $30M NI $5M re 12%`) while the tool args use the *sampled* `bv0, ni, re`. When those diverge, the conversation is internally inconsistent. Structural verification does not catch it.
- Role naming mixes `tool_result` with Hermes `tool`. The fine-tune stack must paper over this.
- 19.5k rows of this will make the model fluent at `<tool_call>` XML and reckless at tool selection.

### 3.4 Abstention is one skeleton in 20 jackets
`_ANSWER_LEADS` / `_ANSWER_CLOSERS` / `_DESKS` / `_OPENERS` exist specifically because *every abstention row shared one answer string*. That is an admission that the type collapsed. `_restyle` varies framing and keeps the body. Good patch, still one body.

A real desk refusal is not “list the missing inputs.” It is:
- “this is a false precision request and here is the range you can defend”
- “the premise that duration is additive across currencies is wrong”
- “I will not compute a Sharpe on two months of returns”
- “the filing does not contain that segment; anyone who answers is fabricating”

Three defect enums (`underspecified`, `unanswerable`, `false_premise`) do not cover stale data, conflicting sources, jurisdiction, material nonpublic, or “answerable only as a range.”

### 3.5 Implementation records are toy functions with a war story
`_impl_equity_dcf` is a 6-line Gordon/stage-1 DCF and a test `assert dcf_value(...) > 0`. The “honest production limitations” FORMAT.md asked for are not a substitute for: vectorization, holiday calendars, day-count, missing prices, corporate actions, or a test that can fail. You already shipped a revision where 15/26 generators `SyntaxError`’d because `dedent`/`strip` order was wrong. That should terrify you about “verified at generation time.”

These will not teach a model to implement a paper. They will teach it to emit a 10-line function and a sentence about assumptions.

### 3.6 Coverage and taxonomy
Taxonomy declares **207 subtopics** across 40 topics. v2 covers **72 (34.8%)**. The uncovered set is not random: L3 private wealth / GIPS / alternatives construction, FRM operational / liquidity / current issues, L2 FSA distortions, and anything that is institutional process rather than a formula.

Worse: the *primary axis is still exam program*. Splits are `cfa_level_i` … `frm_part_2`. A quant assistant corpus should split on **work type** (valuation, risk, construction, execution, research, model-risk, data), not on which exam booklet the stem was filed under.

### 3.7 Preference v2 is small and schematic
7,500 pairs, four modes:

| mode | rows |
|---|---:|
| false_confidence | 2,750 |
| answers_different_question | 1,750 |
| wrong_assumption | 1,750 |
| invented_term | 1,250 |

Useful, disjoint, and *narrow*. 7,425 train pairs is 5.9% of SFT. Preference will not move a 3.8B model off a 126k SFT prior if the rejected side is a template of “invent Durbin-Watson duration.” You need rejected sides that look like *good writing of a wrong model*.

### 3.8 Schema / Hub UX regressions
v1 used Arrow structs. v2 JSON-encodes `conversation`, `tool_schemas`, `verification`, `metadata` so a single schema fits all types. That is convenient for `datasets` and hostile to everyone else (filtering, SQL, the Hub viewer). The viewer then shows exam/TVM again, so the public impression of v2 is “more TVM.” Nested structs per config, or a typed `payload` plus a documented parse helper, would have been better.

### 3.9 The goal / data mismatch is still there
Card: *not intended to train models that pass CFA/FRM; intended to assist a Head of Quant.*  
Content: CFA/FRM taxonomy, exam stems, paragraph builders keyed by LOS, mocked fundamentals for AAPL/MSFT, no papers, no order-book, no risk-system extract, no IC memo, no conflicting vendor data.

You changed the *mixture weights*. You did not change the *generator ontology*.

### 3.10 What v2 did *not* measure
`jobs/fine-tune/README.md`: *“Nothing run on mixed corpus yet; validate that style collapse is resolved via `exam_shape_rate`.”*

So v2 is a hypothesis about style collapse, not a result. Do not announce v2 as the assistant corpus until `09_assistant_eval.py` on the mixed run says so.

---

## 4. What a quant assistant corpus actually has to teach

A Head of Quant’s assistant is not a charterholder chatbot. The job skills, in training-data terms:

1. **Grounded calculation** — compute the number that the model of the world implies, show the model, show the sensitivities.
2. **Model criticism** — say when the model is the wrong object (Gordon growth on a depleting asset; Gaussian VaR in a gap-risk book).
3. **Data suspicion** — missing, restated, vendor-split, as-of mismatch, look-ahead.
4. **Tool discipline** — retrieve, compute, check, refuse to free-hand a price.
5. **Communication register** — IC memo, desk Slack, risk committee, auditor. Different lengths. No `FINAL ANSWER:` in a memo.
6. **Calibration** — range, not point; “I don’t know”; “that’s not identified.”
7. **Implementation** — code that fails a real test, not `assert value > 0`.
8. **Research transfer** — read a result table / abstract and say what would change on *this* book.
9. **Process memory** — multi-turn: yesterday’s mandate, today’s constraint, do not reboot the universe each message.

v1 taught (1) in one register. v2 added a cardboard version of (4), (5), (6), (7). v3 has to make (2), (3), (8), (9) first-class and make (5) *generated by a writer, locked to facts from (1)*.

---

## 5. Cosimo v3 — design

Name: **`btech-software/cosimo-quant-assistant-v3`**  
Companion preference config: **`preference`** with `cosimov3pref_` IDs, disjoint by construction.

### 5.1 Non-negotiable invariants (keep from v1/v2)

- Every number that appears in a target was produced by reference code from a stored seed (or by a tool result that was itself produced by reference code).
- Generation is resumable and idempotent: seed = `hash(record_type, template_id, scenario_id, variant)`.
- Publish refuses on any failed gate.
- Preference IDs never collide with SFT IDs; chosen text is never byte-identical to an SFT target.
- `FINAL ANSWER:` exists only on `exam` records.
- No proprietary exam items. LOS are scaffolding only.
- One epoch at train time. Synthetic data does not get a second pass.

### 5.2 What changes

| axis | v2 | v3 |
|---|---|---|
| Ontology | Exam program → topic → stem | **Work type** → scenario → fact pack → teacher render |
| Prose | f-string paragraph builders | Teacher LLM, facts injected as JSON, numbers locked |
| Verification | numeric for exam; structural elsewhere | numeric + **fact-coverage** + **no-invented-numbers** + style/shape + optional LLM-judge |
| Agentic | 3–5 mocked turns | 6–16 turns, real tool schemas from `cosimo_ft/tools.py`, injected faults |
| Abstention | 3 defects, restyled skeleton | 8 defects, including range-only and conflicting-source |
| Implementation | 6-line functions, `assert x > 0` | spec + hidden tests + known-fail cases on dirty data |
| Preference | 4 template modes, 7.5k | contrastive teacher + harvested failures, ≥25k, mode taxonomy expanded |
| Coverage metric | 72/207 LOS cells | **work-type coverage** + LOS coverage as a secondary report |
| Human loop | none | gold-bar of 300–500 charterholder/desk reviewed items, *disjoint* from generators |
| Splits | by exam program | `train` / `val` / `iid_test` / `unseen_scenario_family` / `assistant_eval` |

### 5.3 Record types

Keep five. Add three. Retire nothing that is already paid for — downsample exam instead.

| `record_type` | train share target | what the target is |
|---|---:|---|
| `exam` | 12–18% | short item, verified number, *varied* trace shape. Exists so arithmetic does not regress. |
| `analysis` | 22–28% | desk note / IC paragraph. 400–1,400 tokens. Facts listed in `fact_pack`. No step liturgy. |
| `memo` **new** | 8–12% | 1–2 page IC/risk memo: recommendation, assumptions, sensitivities, what would change the call. |
| `critique` **new** | 8–12% | given a flawed memo or model output (provided in `stimulus`), write the review. |
| `grounded` **new** | 8–12% | question over a provided table / synthetic filing excerpt / returns window. Answer must cite cells. |
| `abstention` | 8–10% | refuse or bound. Eight defect classes. |
| `agentic` | 12–16% | multi-turn tool conversation, including faults and “do not call.” |
| `implementation` | 6–8% | spec + code + tests that can fail + limitations section. |
| `preference` (separate config) | n/a | 25k–40k pairs, disjoint IDs. |

Exam drops from 21% of v2 (and 30% of the mixed train set) to the low teens. That is intentional. You already know exam overweight collapses style.

### 5.4 The fact pack — the actual v3 invention

Every non-trivial record carries a machine-readable `fact_pack`:

```json
{
  "schema_version": "v3.0",
  "scenario_id": "eq.fcff_dcf.mature_consumer.v3",
  "seed": 1569636562,
  "entities": [{"name": "Northwind Consumer", "ticker": "NWIN", "currency": "USD"}],
  "inputs": {"rev_m": 1280.0, "ebit_margin": 0.18, "tax": 0.24, "g": 0.03, "wacc": 0.09, "n": 6},
  "computed": {
    "nopat_m": 174.96,
    "fcff0_m": 98.4,
    "ev_m": 1842.17,
    "ev_g_lo_m": 1610.02,
    "ev_g_hi_m": 2144.88
  },
  "formulas": ["FCFF = NOPAT - reinvestment", "TV = FCFF_n*(1+g)/(wacc-g)"],
  "allowed_numbers": [1280.0, 0.18, 0.24, 0.03, 0.09, 6, 174.96, 98.4, 1842.17, 1610.02, 2144.88],
  "forbidden_claims": ["stable ROIC > WACC in perpetuity without fade"],
  "must_mention": ["terminal growth vs WACC gap", "reinvestment consistency"],
  "register": "ic_memo",
  "as_of": "2026-06-30"
}
```

The teacher sees `fact_pack` and a render contract. It does **not** see a blank “write a DCF analysis.” If the completion contains a number that is not in `allowed_numbers` (with a documented rounding policy) the row is dropped or regenerated. If a `must_mention` item is absent, regenerate. If a `forbidden_claims` item appears, reject.

This is how you get voice without losing the only property that made v1/v2 worth building.

### 5.5 Work-type ontology (primary axis)

Replace “which booklet” as the primary key.

```
valuation.equity.dcf
valuation.equity.multiples
valuation.equity.residual_income
valuation.fi.oas_spread
valuation.derivatives.replication
risk.market.var_es
risk.market.factor
risk.credit.el_ul
risk.liquidity.liquidation_horizon
risk.model.validation
portfolio.construction.constraints
portfolio.attribution.brinson_carino
portfolio.rebalance.tax_aware
execution.microstructure.impact
execution.tca.arrival
research.paper_transfer
research.data_quality
process.ic_memo
process.model_risk_challenge
process.calibration
```

Each work type has:
- 8–20 **scenario families** (the thing you hold out — successor of `holdout_families`)
- a fact-pack computer (pure Python, deterministic)
- a render brief per register (`desk_chat`, `ic_memo`, `risk_committee`, `auditor`, `code_review`)
- a list of legal pitfalls for preference rejected sides

LOS / program remain *metadata* so you can still mix a thin exam slice and still report CFA-shaped coverage. They are no longer the generator’s spine.

### 5.6 Teacher-LLM contract

Endpoint: any OpenAI-v1 chat completions (`/v1/chat/completions`), so Airflow can point at OpenAI, vLLM, SGLang, Azure, a local teacher, without code changes.

**System** (stable, versioned): Cosimo teacher. You write as the named register. You may only use numbers in `allowed_numbers`. You may round as specified. You may not invent tickers, filings, or constants. If the brief cannot be met with the fact pack, you say so and list the gap — that output is an abstention candidate, not an analysis.

**User** message is structured:
1. `register` + `record_type` + length band
2. `fact_pack` JSON
3. `stimulus` if critique/grounded
4. hard constraints (`must_mention`, `forbidden_claims`, “no FINAL ANSWER”, “no Step N. liturgy”)
5. optional `voice_hint` (short, sampled): skeptical, constructive, time-pressed, teaching-the-analyst

**Sampling for diversity, not for facts.** Temperature 0.7–0.9 on prose records. Temperature 0 on anything that is allowed to touch numbers that are not already in the pack (i.e. never). If you need two phrasings, call twice; do not let the model recompute.

**Teacher quality bar.** Use a model that can write. Phi-4-mini is the *student*. If the teacher is also 4B, v3 prose will look like v2 Mad-libs with extra adjectives. Budget for a 70B-class or a strong closed teacher on the prose stages only; keep computers local and free.

### 5.7 Preference design that can actually move DPO

Generate pairs *after* the SFT target exists.

For each SFT row with probability `p_pref` (say 0.25 on analysis/memo/critique/grounded, 0.4 on abstention):

1. Lock the same `fact_pack`.
2. Ask the teacher for a **rejected** completion under an explicit pitfall contract:
   - `wrong_assumption` — use a named wrong model (ordinary vs due; additive duration across FX; Gaussian VaR on options)
   - `false_precision` — emit a point estimate the pack only supports as a range
   - `invented_number` — (negative control; these should be *easy* rejects, cap at 10% of pairs)
   - `ignored_constraint` — mandate / limit / tax lot stated in stimulus, ignored
   - `overconfident_abstention_fail` — answers an underspecified pack
   - `wrong_register` — exam liturgy on a memo brief
   - `tool_skip` — free-hands a number a tool should have produced
   - `look_ahead` — uses a print that is after `as_of`
3. Verify rejected: it must (a) be fluent, (b) violate the pitfall, (c) *not* be a trivial copy, (d) preferably still use only allowed numbers or clearly wrong numbers that the computer can label.
4. Store under `cosimov3pref_<hash>` with `source_sft_id` recorded but **not equal**, and `chosen` re-rendered by the teacher a second time (paraphrase, same facts) so chosen ≠ SFT target bytewise.

Harvested pairs (better, smaller): run the current student on `assistant_eval` and on a slice of train-like scenarios, keep failures that a judge + computer agree are wrong, pair with the v3 target. This is how you get rejected sides that look like *your* model, not like your template.

### 5.8 Agentic v3

- Tool schemas are imported from `jobs/fine-tune/cosimo_ft/tools.py`. If a tool is not in the serving app, it does not appear in the corpus. End of discussion.
- A **tool oracle** implements each tool against the scenario’s fact pack (and optional dirty layers: missing fields, delays, renamed columns).
- A **planner template** only chooses *which faults to inject* (empty result, stale as-of, wrong ticker, rate limit, schema drift). It does not write the assistant utterances.
- Teacher fills assistant turns given: user goal, tools, oracle results so far, injected fault. After each proposed tool call, Airflow executes the oracle and feeds the real JSON back. This is a real multi-step loop, 6–16 turns.
- Include 15–20% **no-call** conversations: the fact pack is already in the user message; calling a tool is waste. Preference rejected = unnecessary call.
- Verify: every number in the final assistant turn is in the union of fact pack and tool results. `answer_uses_results` becomes a numeric subset check, not a string heuristic.

### 5.9 Implementation v3

Each record:
- `spec` — what the function must compute, edge cases, dirty-data policy
- `public_tests` — shown to the student
- `hidden_tests` — used at generation time and in later RL if you ever go there
- `dirty_fixture` — NaNs, duplicated dates, a holiday, a corporate action
- `reference_code` — passes public + hidden
- `limitations` — teacher-written, fact-locked

Drop the row if hidden tests do not pass. `assert value > 0` is not a test.

### 5.10 Exam slice (keep, shrink, diversify)

You still want a thin verified item bank so GSM-style arithmetic in finance units does not fall on the floor.

Rules:
- Cap exam at ≤18% of SFT.
- No stem family with >400 train rows. v1’s 1,000×71 is how you got 18% OOD.
- Hold out entire **scenario families**, not wrappers. You already learned that `v_`/`cr_`/`m_` leak.
- Trace shape is sampled: prose, back-of-envelope, short table, then the number. `ASSUMPTIONS:` + `Step N.` is at most 25% of exam traces.
- Distractors are generated from *named pitfalls*, then checked ≠ correct. If the pitfall computer cannot make a distinct number, drop the distractor, do not add 7.0.

Do not remix uncapped v1 into v3. If you need exam depth, regenerate from the same computers under the new distractor and trace rules. v1’s 15k dupes and shared-ID pairs stay in the museum.

### 5.11 Gold bar and eval

Build a **300–500 row gold bar** that is *not* produced by the generators:
- 100 exam-like items written or rewritten by a person
- 100 desk notes / memos
- 50 abstentions
- 50 agentic transcripts (even if tools are still synthetic)
- 50 critiques

Use it only for eval and for teacher-quality spot checks. Jaccard / embedding near-dup against the train generator output is a publish gate.

Keep `cosimo_unseen_stems` as `unseen_scenario_family`. Report that number in public. `iid_test` is a vanity metric and you already know it.

Assistant eval stays: `exam_shape_rate`, `mean_new_tokens`, `abstention_rate`, `unknown_terms`, `hallucinated_tool_rate`, plus new:
- `invented_number_rate` (parse completions against the fact pack)
- `must_mention_hit_rate`
- `register_match` (memo vs exam liturgy)
- `citation_cell_rate` on `grounded`

---

## 6. Airflow ETL

This is the pipeline you asked for. Treat each box as a task; fact computers are CPU; teacher calls are the pool.

```
seed_inventory          # scenario families × variants × record_type plan
        │
        ▼
build_fact_pack         # pure Python, seeded, writes fact_pack.jsonl
        │
        ├──────────────► render_exam          # no LLM, or LLM only for trace phrasing with locked numbers
        ├──────────────► render_prose         # analysis / memo / grounded / critique / abstention via teacher
        ├──────────────► render_agentic       # loop: teacher turn ↔ tool oracle
        └──────────────► render_implementation
                                │
                                ▼
                     verify_record            # numeric + invented-number + shape + tests
                                │
                     fail ─► retry_teacher (max 3) ─► dead_letter
                                │
                                ▼
                     make_preference          # contrastive teacher + optional harvest
                                │
                                ▼
                     dedup_and_split          # minhash questions, hold out families, disjoint pref ids
                                │
                                ▼
                     publish_hub              # refuse if any gate red
```

### 6.1 Suggested Airflow shape

- DAG `cosimo_v3_build`, schedule `@once` + manual trigger, max active tasks bounded by teacher RPM.
- Pool `llm_teacher` sized to the endpoint.
- XCom / object store (S3, GCS, or the repo’s shard disk) for fact packs and completions. Do not put completions in XCom.
- Task mapping over `scenario_family` so a bad family does not block others.
- Dead-letter shard is a first-class output. You will want it for teacher-prompt debugging.

### 6.2 OpenAI-v1 call

```python
client = OpenAI(base_url=os.environ["TEACHER_BASE_URL"], api_key=os.environ["TEACHER_API_KEY"])
resp = client.chat.completions.create(
    model=os.environ["TEACHER_MODEL"],
    temperature=0.8,
    max_tokens=1800,
    messages=[
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": render_brief(record_type, fact_pack, stimulus)},
    ],
)
```

Pin `TEACHER_MODEL` in the shard metadata. A silent teacher swap mid-run is how you get two house styles and call it diversity.

### 6.3 Verify task (the thing that makes LLM generation safe)

For every prose record:

1. Parse all numbers in `answer` / assistant turns (same tokenizer you already have in `dataset/verification/nums.py`).
2. Every parsed number must match an `allowed_numbers` entry within `abs_tol` / `rel_tol` *or* be in a tiny whitelist (100, 2 for “two-stage”, years in `as_of`, 252).
3. At least `k` of `must_mention` strings or their synonym list hit.
4. Zero hits on `forbidden_claims`.
5. Shape gate: no `FINAL ANSWER:` unless `record_type == exam`; `exam_shape_rate` features off-limits for memo/analysis.
6. Length band.
7. Terminology gate (reuse `terms.py`).
8. Optional teacher-as-judge *only as a secondary score*, never as numeric ground truth.

Fail ⇒ retry with temperature down and a “you invented N; remove it” repair prompt. Three strikes ⇒ dead letter.

### 6.4 Cost / volume sketch (order of magnitude)

Assume ~80k SFT prose-bearing rows need a teacher call, ~20k exam/impl mostly local, ~30k preference rejected calls, ~10% retries.

- ~130k teacher completions
- ~800 tokens out + ~1,200 in ≈ 260M tokens
- At a cheap 70B-class teacher this is a budget line, not a rounding error. Cap v3 at **40k–60k high-quality rows** rather than chasing v2’s 113k. You already saw that 71k of one stem family is worse than 8k of many. **Quality × scenario-family count** is the KPI, not rows.

Recommended first publish: **45k SFT + 20k preference**, coverage report attached, gold-bar eval attached. Expand after `unseen_scenario_family` and `invented_number_rate` look honest.

---

## 7. What to delete or freeze

- Freeze v1 on Hub. Do not keep mixing it at 12–30% once v3 exam exists. The mix was a crutch for exam depth and a vector for style collapse.
- Do not regenerate v2 analysis through the same `v2_analysis.py` builders and then “ask an LLM to rewrite.” That launders Mad-libs; the structure stays.
- Stop publishing JSON-encoded structs as the only form. v3 configs: `supervised` (typed columns + JSON `fact_pack` / `conversation` only where needed) and `preference`.
- Kill `assert x > 0` tests.
- Kill distractor `+ 7.0` finalize. If a pitfall cannot produce a distinct wrong number, there is no distractor.
- Kill scenario_clause-as-diversity. Desk names are flavour. They are not families.

---

## 8. Honest risks of v3

1. **Teacher contamination / license.** A closed teacher may have trained on CFA prep. Keep the *numbers* synthetic and the stems original; do not paste real item banks into prompts. Record teacher model + license in the card.
2. **Judge-model incest.** Do not let the same model grade its own prose as the only gate. Computers first.
3. **Cost-driven shrinkage of diversity.** The failure mode will be “we generated 40k rows from the 12 scenario families that were easy to computerize.” Put a coverage cap in `seed_inventory`: no family > 3% of train.
4. **Persona tax.** The 2,494-character identity block is ~600 tokens per example. On 45k rows that is ~27M tokens of the same sermon. Shorten `identity` for training; keep the long one for serving if you must. Measure whether the short identity still holds register.
5. **3.8B student ceiling.** A perfect v3 corpus will not make Phi-4-mini sound like a Citadel HoQ. It will make it a *competent small desk tool* that does not invent numbers and does not exam-shape a memo. Set the claim to that, or change the student.

---

## 9. Acceptance criteria for calling it v3

Publish is allowed when all of the following are true:

1. Every exam / implementation / fact_pack number recomputes from seed.
2. Invented-number rate on a 2k-row audit slice < 1%.
3. No scenario family > 3% of SFT train.
4. Exam share ∈ [0.12, 0.18].
5. Preference IDs disjoint; chosen ≉ SFT target (n-gram overlap below a set threshold).
6. Gold bar disjoint from train by near-dup.
7. A mixed-corpus fine-tune (even a smoke LoRA) shows:
   - `exam_shape_rate` on memo/analysis prompts **down** vs v1-trained checkpoint
   - `mean_new_tokens` **not** collapsed to ~120
   - `unseen_scenario_family` accuracy **materially above** v1’s 18% *or* an explicit write-up why the student is the bound
   - `invented_number_rate` down vs base
8. Dataset card lists coverage by **work type** and by LOS, teacher model, and the 18% story so the next person does not repeat it.

If you cannot run (7), you do not yet have a v3. You have a generator.

---

## 10. Recommended build order

1. Fact-pack computers for 15 scenario families across valuation / risk / portfolio / execution (two weeks of real work, not a weekend of templates).
2. Teacher render + invented-number gate on `analysis` and `abstention` only. Produce 2k rows. Read 100. You will hate the first teacher prompt. Fix it.
3. Agentic loop against real `tools.py` on 3 tools only.
4. Preference contrastive on the 2k.
5. Smoke LoRA. Look at `exam_shape_rate` and invented numbers.
6. Only then scale to 45k and add `memo` / `critique` / `grounded`.

Do not start by writing 299 new f-string generators. That is how you get v2.1.

---

## 11. Line to use internally

v1 proved you can verify numbers and still train the wrong animal.  
v2 proved you can change the mixture and still generate the same animal in five costumes.  
v3 is the first corpus whose *unit of generation* is a fact pack plus a writer, and whose *unit of holdout* is a scenario family rather than a wrapper prefix.
