# Cosimo v3.2 — architecture amendment against current `main`

This amends V3.1. It does not add a v4 backend. The Hub target is still
`btech-software/cosimo-quant-assistant-v3`. The student is still
`Qwen/Qwen3.8-27B`. Think-off on the wire, student-row schema, register
gate, operator modes, and `gold_bar_v3` as the publish brake stay.

V3.1 made rows *trainable*. V3.2 makes them *worth training*.

Evidence this amendment exists: `dataset/examples/v3/live_three_registers.jsonl`
(DeepSeek-V4-Flash-0731, think off, 9/9 board-green). Factory voice is
fixed. Quality is not:

- TCA analysis says the schedule both is and is not the risk
- attribution \(k\) is \(R_b \cdot k_{\text{Carino}}\), sector pieces
  sum to ~1.6 bp against 372.6 bp active, and the teacher invents a
  “sign flip” theory to save the pack
- VaR memo calls a −0.10% mean a “positive drift” / “$10M expected gain”
- analysis / grounded / memo share a first sentence and a five-beat essay
- `TEACHER_SYSTEM` requires mechanism, binding constraint, hidden
  assumption, what-would-move, *and* a call on every row

Those are prompt-and-pack bugs, not teacher bugs.

---

## Decision log (additions)

8. A think-only teacher is not the design. When the endpoint can disable
   thinking, strip-only lanes send an explicit disable and stamp
   `think_present` from returned reasoning text. DeepSeek-V4-Flash-0731
   on the three-register sample is the reference.
9. `must_mention` is a *conditional test*, not a slogan the answer must
   enact when the inequality in the pack says otherwise.
10. Kind is the job. Register is the shape. They are different briefs.
    Word-budget-only differentiation is how analysis and grounded collapse.
11. A pack that cannot add is a `PackError`, not a prose problem. The
    teacher may not invent theory to reconcile it.
12. `memo` × `desk_chat` is not a legal pair for work types whose families
    are desk-only (TCA). Inventory must not emit it.

---

## A. Pack contract — additivity and honest \(k\)

### A.1 Carino (blocking)

`dataset/pipelines/v3/packs/portfolio_attribution.py` today uses

```
k = ln((1+Rp)/(1+Rb)) / ((Rp - Rb) / Rb)   # = Rb * k_carino
```

Replace with the textbook period factor:

```
k = ln((1+Rp)/(1+Rb)) / (Rp - Rb)          # Rp == Rb already PackError
```

Effects stay

```
alloc_i = (wp_i - wb_i) * (rb_i - Rb) * k
sel_i   =  wp_i         * (rp_i - rb_i) * k
int_i   = (wp_i - wb_i) * (rp_i - rb_i) * k
```

Assert, or `PackError`:

```
abs(sum_i(alloc+sel+int) * 1e4 - active_bps) <= 0.15
```

Canonical must include:

```
active_bps
allocation_total_bps
selection_total_bps
interaction_total_bps
reconciling_residual_bps    # ~0 after the assert
carino_k
```

Question may alias active as a whole-bp figure (373). Answer uses
`active_bps` (372.6).

`must_mention` for this work type becomes `["allocation versus selection"]`.
Drop `"skill signal"` and `"Carino factor"` as required phrases. A valid
call is “no single effect is large enough to act on.”

Do not re-render attribution prose until this assert is green in
`dataset/tests/v3/test_attribution_pack.py` on the three-register seeds.

### A.2 Conditional mentions (all work types)

`FactPack` grows:

```python
conditional_mentions: list[{when: str, mention: str}]
conditional_forbids:  list[{when: str, claim: str}]
```

Evaluated against `computed` / `inputs` at pack-build time into the
existing `must_mention` / `forbidden_claims` lists. Examples:

```
TCA: when participation >= participation_cap
       mention: schedule risk at the cap
     when participation <  participation_cap
       forbid:  "the schedule itself becomes the risk"

VaR: when daily_mean < 0
       forbid:  "positive drift", "expected gain"
```

The teacher never sees a slogan the inequality has already ruled out.

### A.3 Display is part of the brief

`render_brief` injects a `NUMBER POLICY` block built from
`pack["display"]` and `canonical`:

- integers: display string or bare integer, never `430567.0`
- money: `$359,667` when display says so
- participation already in % if display is `%`; do not emit `0.048555`
  when `4.86%` exists
- no more decimals than the display/canonical entry

`overprecise_numbers` / `integer_format_offenders` fail a row that emits
the raw canonical float when a display form exists.

---

## B. Prompts — three jobs, one short system

### B.1 System (all `BRIEF_KINDS`)

Replace `TEACHER_SYSTEM` with ≤ 8 lines:

```
You write student-facing desk answers for Cosimo.
Use only canonical figures from the pack, in their display form when one
exists. Do not invent entities or prints.
If the pack does not support the question, say what is missing and stop.
Do not mention the pack, the gate, allowed numbers, or this contract.
```

Delete from system: mixed “junior desk” / “head of quant” persona; the
five-beat list (mechanism, binding constraint, hidden assumption,
what-would-move, call); “cover every must_mention.”

Those five beats are the liturgy V3.1 removed `FINAL ANSWER:` to escape.
They produced `live_three_registers.jsonl`.

### B.2 User brief = kind template + register shape + work-type addendum

`render_brief(pack, kind)` composes three strings. Kind and register are
not interchangeable.

#### Kind: `analysis`

```
KIND: analysis
REGISTER: {register}

Answer the question in that register.
Caps: desk_chat ≤ 12 sentences and ≤ 160 words;
      ic_memo ≤ 220 words; risk_committee ≤ 220 words.

Order:
1. First sentence answers the question (number + unit).
2. One mechanism sentence that uses a pack figure.
3. One constraint or assumption that would change that number.
4. A call consistent with (1)–(3). If they conflict, say so and
   prefer the number.

Forbidden: restating every input; naming the pack or the gate;
"binding constraint" unless register is risk_committee and you mean
a limit; "Call:" as a label when register is desk_chat.
```

#### Kind: `grounded`

```
KIND: grounded
REGISTER: {register}

≤ 8 sentences, ≤ 140 words.
First sentence: headline number + unit.
Then at most three figures, each tied to one phrase from the question.
One caveat. No call unless the question asked for one.
No section headings.
Do not open with the sentence analysis would use on this pack.
```

#### Kind: `memo`

```
KIND: memo
REGISTER: {register}

ic_memo headings: Finding / Evidence / Call
risk_committee headings: Exposure / Assumption / Limits / Breach case
desk_chat: illegal pair — inventory must not emit this
           (see §C).

Finding is one sentence. Evidence is the decomposition.
Call is one sentence and must not contradict Finding.
No pasted formulas. If alloc+sel+int ≠ active, Finding is
"the pack does not reconcile" and Call is abstain.
```

#### Kind: `abstention` / `critique`

Abstention: 2–5 sentences, what is missing, no substitute number.  
Critique: one defect in a supplied draft; do not rewrite the memo.

`WORD_BUDGETS` tighten to match the caps above. The old bands
(analysis 120–400, grounded 110–340) *licensed* the 288-word desk note.

### B.3 Register shape (already a gate; now also a brief)

Keep `verification/register.py` as the repair axis. The brief must not
ask for a shape the gate forbids (`Call:` on desk_chat, trade
recommendation on risk_committee).

`_REGISTER_HINTS` become the shapes in §B.2, not one-liners.

### B.4 Work-type addenda (appended to the user turn)

Selected by `pack["work_type"]`. Not in the system prompt.

**`execution.tca.arrival`**

```
Participation is shares/ADV. Compare it to the cap in the pack.
If participation < cap: impact is the cost; do not call schedule the risk.
Half-spread is not the whole cost. Arrival is the benchmark in this pack;
do not introduce a decision price that is not in canonical.
```

**`portfolio.attribution.brinson_carino`**

```
Report allocation_total, selection_total, interaction_total, and active.
If they do not add to active, abstain on "which effect to act on."
Do not explain a sign flip with a story about k unless k is in canonical
and the signed effect matches the raw (wp-wb)*(rb-Rb) sign.
The effect worth acting on is the largest absolute total that is also a
decision (weights vs names). If both are under 5 bp, say so.
```

**`risk.market.var_es`**

```
A negative daily mean is a negative drift (adds to expected loss).
Never call it a gain. VaR is a quantile; ES is the tail mean.
Do not recommend a trade in risk_committee.
```

**`valuation.equity.dcf` / `valuation.equity.multiples`**

```
EV is not a share price. If price/share count is missing, the call is
"no ownership call" — not "own near this EV."
```

---

## C. Inventory — illegal pairs

`work_types.yaml` / `inventory.py`:

- Do not emit `record_type=memo` when the family’s register is `desk_chat`.
  TCA families are desk-only; they get analysis / grounded / exam /
  agentic / implementation / abstention, not memo.
- `memo` requires a family whose register is `ic_memo` or
  `risk_committee`.
- Attribution `must_mention` and variant counts stay at the WIP 20
  until the additivity test is green and the three-register attribution
  coordinates have been re-rendered.

A test in `test_inventory.py` lists illegal `(work_type, family,
record_type)` triples and fails if `expand_jobs` produces one.

---

## D. Contradiction gates (repair, not only publish)

`gate_violations` grows three pack-aware checks. Hits become repair
lines, then dead letters.

| Tag | When |
|---|---|
| `schedule_risk_below_cap` | answer matches schedule-as-risk language and `participation < cap` |
| `drift_sign` | `daily_mean < 0` and answer matches `positive drift` / `expected gain` |
| `unreconciled_call` | `abs(alloc+sel+int-active) > 0.2 bp` and answer picks a skill signal / effect to act on |

`slice_audit` fails a slice that contains any of these tags on sft rows.

Repair copy stays “rewrite the draft fixing every listed violation” and
adds: “do not mention the violation tag in the answer.”

---

## E. Think policy (clarification, not a revert)

`routing.py` lanes stay as V3.1:

- analysis / memo / grounded / abstention → prose, `think=False`
- exam / critique / implementation / agentic planner → reasoning, `think=True`

`teacher/client.py` `build_body`:

```
body["thinking"] = {"type": "enabled" if think else "disabled"}
```

Omit-the-field is not disable. The three-register sample is the proof
that DeepSeek honors disable; stamp `think_present` from
`reasoning_content`, not from the request flag.

Strip-only budgets return to **800** visible tokens and **120s**. The
3–4k floor was a think-forced hedge and is not the design.

`v3_think_ablation.md` must record a real think-on vs think-off run on
a teacher that can disable. 0=0 on a think-forced teacher is not a
verdict. Promote memo think-on only if invented-number rate drops ≥ 2
points.

`slice_audit` may require `think_present is false` on analysis /
grounded / abstention **after** `probe_teacher.py` has asserted that
disable produces empty `reasoning_content` on this endpoint. Until that
probe is green, do not use `think_present` as a readiness bit (the
Qwen-forced-think failure mode).

---

## F. Target shape (same coordinates, after the jump)

Re-render the nine ids in `live_three_registers.jsonl` against the new
briefs and the fixed attribution computer. Acceptance, per row:

- first sentence of `analysis` ≠ first sentence of `grounded` on the
  same `scenario_id`
- desk_chat analysis ≤ 160 words
- no `schedule itself becomes the risk` when participation < cap
- attribution: printed totals add to `active_bps`, or the row is an
  abstention
- VaR: no “positive drift” when `daily_mean < 0`
- no `Call:` on desk_chat
- `think_present` false on these kinds
- completion_tokens < 900

Those nine become the first draft of `dataset/goldbar/gold_bar_v3.jsonl`
after a human read. Rows that fail stay in
`examples/v3/_rejected_three_registers.jsonl` as a regression fixture.

Do not raise `LIMIT` or run `full` because the sixteen-axis board is
green. The board did not see the errors this amendment names.

---

## G. What does not change

- Two subsystems, Hub seam, `cosimo.tools`
- Student Qwen3.8-27B, pinned SHA, Spark QLoRA
- Student row schema (`row.py`), teacher logs optional
- Register gate as a generation-time repair axis
- `dataset_build.sh` modes, `CONFIRM_FULL`, missing gold bar refuses publish
- Frozen v1/v2 generators

---

## H. Implementation order on current `main`

PR-H1  Carino \(k\) + additivity test + attribution `must_mention` shrink  
PR-H2  `conditional_mentions` / `conditional_forbids` folded at pack time  
PR-H3  `TEACHER_SYSTEM` shrink + kind × register × work-type briefs  
PR-H4  illegal inventory pairs; drop TCA `memo`  
PR-H5  contradiction tags in `gate_violations` + slice_audit  
PR-H6  explicit `thinking.disabled` + probe  
PR-H7  re-render the nine coordinates; gold-bar the keepers  

No live-slice larger than those nine until H7. The quality jump is a
diff on known seeds, not a new inventory.
