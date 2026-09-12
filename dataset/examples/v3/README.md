# examples/v3

Two kinds of file -- generated and captured -- and the difference matters.

## `<record_type>.jsonl` — generated, one row each

Eight files, one per record type, each holding exactly one student row, all
produced by `make_examples.py` through the *real* renderers driven by scripted
teachers. They regenerate byte-identically, and `test_examples_and_leak_gate.py`
asserts both that they do and that each one clears the verify board.

They exist to answer "what shape is a v3 row?" — and they replaced
`pipelines/v3/example_generation.jsonl`, which answered that question with nine
`analysis` rows whose `messages` opened on the teacher's own system turn. A
reader learning the corpus from that file learned the labelling protocol
(amendment §A, §H).

Regenerate with:

    uv run --group corpus python dataset/examples/v3/make_examples.py

## `live_analysis.jsonl` — captured, not regenerable

Six `analysis` rows from a real teacher (`qwen3.8-flash-next`, 2026-09-11),
across four work types and all three registers the plan emits.

**Three of the six would no longer ship.** All passed every axis when they were
captured; the register gate was then tightened and the three `desk_chat` rows
now draw `register desk_chat labels its call`. They end
`... Call: execute under current participation` mid-paragraph, and the heading
rule is anchored at a line start -- rightly, since "the finding: we are long"
mid-sentence is prose -- so the memo's speech act walked straight past a field
whose whole job was to catch it.

They are kept rather than pruned. A capture that shows a gate being wrong is
worth more than a capture curated to look clean, and re-rendering would answer
a different question (see below). The three that still pass are the `ic_memo`
row and both `risk_committee` rows.

This file is deliberately **not** regenerable and must never be wired into
`make_examples.py`. Its whole value is that it is what the endpoint actually
wrote, at temperature 0.7, on the day the amendment's gates first ran live.
A regenerated version would be a different sample and would answer a different
question.

`verification.teacher.usage` carries only `completion_tokens`: the rows were
rebuilt from the captured answers plus the recomputed packs, so the row content
is the teacher's verbatim, and that one field is partial. Recorded here rather
than silently padded.

### Why it is worth reading

It is the same nine coordinates that produced the pre-amendment sample now in
`_teacher_logs/legacy_analysis_9rows.jsonl`, with **byte-identical pack
figures** — only the entity names differ (the pool widening touched draws that
come after the numbers). So the two files are a controlled before/after on the
same scenarios, same model:

| | legacy (pre) | live (post) |
| --- | ---: | ---: |
| shipped | 9/9 | 6/9 at capture time |
| median completion tokens | 7,100 | 3,799 |
| median words | 149 | 143 |
| billed : kept | 35.7x | 21.5x |
| raw floats like `430567.0` | 58 | **0** |

The last row is amendment §D in production: `430567.0` becomes `430,567`,
which is the worked example the amendment gives.

The three that did not ship at capture time were all dead-lettered by one
defect in the register gate — an `ic_memo` that closed `Call: <decision>` was
scored as making no call, because the check read a phrase list while
`_MEMO_HEADINGS` was already matching the same heading to permit memo
scaffolding. Fixed; the legacy rows that tripped it now pass.

## `live_v3_2.jsonl` — the amendment's acceptance rows

Eight rows from `deepseek-v4-flash-0731` (2026-09-12, think off) over the v3.2
packs, briefs and gates: `analysis` / `grounded` / `memo` across
`execution.tca.arrival`, `portfolio.attribution.brinson_carino` and
`risk.market.var_es`. Eight and not nine because `memo` x `desk_chat` is no
longer a legal pair (§C) -- the ninth row of the older capture existed only
because the inventory asked for a document the register cannot hold.

Read it beside `live_three_registers.jsonl` below. Same coordinates, same
teacher, same three registers; everything else is the amendment.

| | v3.1 capture | v3.2 capture |
| --- | ---: | ---: |
| rows | 9 | 8 |
| words, min / median / max | 225 / 288 / 516 | **80 / 117 / 158** |
| median completion tokens | 476 | **~200** |
| rows carrying a §D contradiction | 7 of 9 | **0 of 8** |
| analysis/grounded pairs sharing a first sentence | 3 of 3 | **0 of 3** |
| rows shipped on the first attempt | — | **6 of 8** |

What the contradictions were, and are not any more: a 4.86% clip written up as
a pacing problem on all three TCA rows; an effect named worth acting on out of
sector pieces that missed the active return by 371 bp, on all three attribution
rows (the old Carino factor made that unavoidable -- the teacher had to invent
a mechanism to save the pack); and a -0.10% daily mean reported as $10M of
expected gain on the VaR memo. None of them invented a number, which is why
sixteen axes passed every one.

`test_live_v3_2_capture.py` holds this file to §F's acceptance list -- the word
band its kind and register allow, no contradiction against the recomputed pack,
no shared first sentence, completion under 900 tokens, think off -- reading the
current tables rather than pinned literals, so the assertions move when the
contract moves. It is deliberately *not* held to the full sixteen-axis board,
for the reason the next section gives.

**What it does not cover.** Three of five work types and three of eight record
types. `critique`, `abstention`, the two valuation lanes and every
reasoning-lane kind (`exam`, `agentic`, `implementation`) still have no live
sample -- and until 2026-09-12 the reasoning lanes ran at a 2,048-token cap
against a measured 3,081-3,736-token chain of thought, so their first-pass
quality has never been observed at all. "The corpus reads well" is not a claim
this directory supports yet; "a prose row under the v3.2 briefs is short,
fact-locked and does not contradict its pack" is.

## `live_three_registers.jsonl` — captured, not regenerable

Nine rows from a real teacher (`deepseek-v4-flash-0731`, 2026-09-11): three
work types x `analysis` / `memo` / `grounded`, three rows in each of the three
registers the plan emits. At capture, **9/9 shipped and all nine cleared the
sixteen-axis board**, with no dead letters.

**None of the nine would ship under the v3.2 gates, and that is why the
amendment exists.** Measured against the current contract: all nine are over
their kind's word band (288, 237 and 334 words where the desk now takes 160,
140 and 300), and seven carry one of the three §D contradiction tags --
`schedule_risk_below_cap` on the three TCA rows (a 4.86% clip written up as a
pacing problem), `unreconciled_call` on the three attribution rows (an effect
named worth acting on out of pieces that missed the active return by 371 bp,
which the old Carino factor made inevitable), and `drift_sign` on the VaR memo
(a -0.10% daily mean reported as $10M of expected gain). Not one of them
invented a number, which is exactly why sixteen axes passed them.

The file is kept, not pruned, for the same reason `live_analysis.jsonl` is:
a captured sample that stops clearing a tightened gate is the evidence the
tightening was needed. It is not regenerable -- re-rendering these coordinates
needs a billed teacher -- so read the prose below as a record of what the
V3.1 briefs produced, never as a statement that the current board accepts it.

Two things make it worth keeping beside `live_analysis.jsonl`.

### It is the first sample with thinking genuinely off

`verification.teacher.think_present` is `false` on every row. Every earlier
capture had it `true` -- not as a choice but because `build_body` sent only
DeepSeek's cloud `thinking: {type: ...}` field, which a vLLM/SparkInfer serve
ignores, so the server's own `thinking: true` default stood and the project
read a 10x token bill as a property of the model.

| | `live_analysis` (think on) | `live_three_registers` (think off) |
| --- | ---: | ---: |
| rows shipped | 6/9 at capture | **9/9** |
| median completion tokens | 3,799 | **476** |
| median words | 143 | **288** |
| billed : kept | 21.5x | **1.6x** |

Twice the prose for an eighth of the tokens. That is amendment §B's claim, and
this file is the first evidence for it rather than against it.

### It is what a gate's own briefing is worth

The first live render of these nine lanes shipped 1. The next three rounds
shipped 4, then 7, then 9 -- and not one row of the difference came from
asking the teacher for better writing. Every fix was a case of the pipeline
demanding something it had never stated or could not be satisfied at all:

* four register rules the gate enforced and the brief never mentioned,
  including a sentence ceiling the model was refused for crossing but never
  shown (now `verification.register.register_shape`, read by both sides);
* a repair turn that appended "Do not shorten what was compliant" to a
  *length* violation -- cut and do not cut, three attempts running;
* the format axis advising "write 373" for a `373.0` that `rounding_drift`
  then refused as the question's spelling of a 372.6;
* that same axis demanding a bare `0` where the pack's canonical value **is**
  `0.0` -- the row returned an identical 256 words three times rather than
  misreport its pack, and it was right;
* an invented-number tolerance that was purely relative, so `3.31` of a
  `3.309` passed while `-0.42` of a `-0.416` failed. One act -- a desk writing
  a percentage to two places -- sorted by magnitude.

So the honest reading of the earlier "think-off costs gate compliance" result
is that it measured five bugs, not a trade-off.

### What it does not show

Axes 15 and 16 report **below support**, not pass. Near-duplicate and register
separation need eight rows per cell and this file has one per cell, so
*"the three registers are distinct"* remains unmeasured -- it needs roughly
twenty-four rows. Three rows per register is enough to read, not to certify.

Like `live_analysis.jsonl` this file is deliberately **not** regenerable and
must never be wired into `make_examples.py`.

## `_teacher_logs/` — debug, gitignored except the legacy capture

Where the factory's own transcripts go when
`COSIMO_V3_KEEP_TEACHER_MESSAGES=1` is set. Never a shard, never trainable:
these carry the teacher's system turn, which is the one thing a student row
may not (§A).

`legacy_analysis_9rows.jsonl` is kept as the *before* side of the comparison
above, and as the fixture `test_examples_and_leak_gate.py` uses to prove the
old row shape now fails the prepare gate.


## What `live_analysis.jsonl` does not show

It is six rows of one record type. `critique`, `abstention` and `agentic`
still have **no live sample at all** -- their files here are the scripted
fixture. (`memo` and `grounded` gained one in
`live_three_registers.jsonl` above.) So "the corpus reads well" is not a claim
this directory supports yet; "an `analysis` row is fact-locked and has the
right shape" is.

Two of the six are `large_cap_intraday` and two are `equity_longonly`: the
same job twice, different sizes. That is an artefact of reusing the legacy
sample's coordinates for a controlled comparison, not a property of the
family -- the entity pools now carry 244 distinct names across 262 packs
(worst within-family reuse 1, down from 20). A real slice will not twin like
this; a six-row A/B on fixed coordinates has to.

Finally: **do not train on `analysis.jsonl`.** That row is the scripted
teacher, and it reads like one -- "Reading the pack directly, avg fill price
at 295.7747, cost bps vs arrival at 38.07, each as computed." It proves the
schema and would poison the voice if it reached SFT. The generated files are
shape tests; `live_analysis.jsonl` is the quality sample.
