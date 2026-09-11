# examples/v3

Two kinds of file, and the difference matters.

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

## `_teacher_logs/` — debug, gitignored except the legacy capture

Where the factory's own transcripts go when
`COSIMO_V3_KEEP_TEACHER_MESSAGES=1` is set. Never a shard, never trainable:
these carry the teacher's system turn, which is the one thing a student row
may not (§A).

`legacy_analysis_9rows.jsonl` is kept as the *before* side of the comparison
above, and as the fixture `test_examples_and_leak_gate.py` uses to prove the
old row shape now fails the prepare gate.


## What this sample does not show

It is six rows of one record type. `memo`, `critique`, `grounded`,
`abstention` and `agentic` have **no live sample at all** -- their files here
are the scripted fixture. So "the corpus reads well" is not a claim this
directory supports yet; "an `analysis` row is fact-locked and has the right
shape" is.

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
