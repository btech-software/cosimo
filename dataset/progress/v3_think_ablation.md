# v3 think ablation (amendment §B)

The amendment's test, run: *20 analysis rows think-off vs think-on;
promote think-on only if the invented-number rate drops by >= 2 points.*

- record type: `analysis`
- teacher: `deepseek-v4-flash-0731`
- rows asked per arm: 20
- budgets: 800 think-off / 6144 think-on

| metric | think-off | think-on |
| --- | ---: | ---: |
| rows answered | 20 | 20 |
| invented-number rate | 0.000 | 0.000 |
| gate-clean rate | 0.000 | 0.050 |
| empty-draft rate | 0.000 | 0.200 |
| median words | 84 | 97 |
| mean completion tokens | 138 | 3070 |
| wall seconds | 105.8 | 1777.3 |

**Δ invented-number rate: +0.0 points** (bar: ≥ 2 points to promote)

## Verdict: KEEP think-off for `analysis`

Think stays off for the prose lanes. `routing.py` already reflects
this; `COSIMO_V3_MEMO_THINK=1` remains the one operator override,
and it is for re-running this measurement on the memo lane, not for
turning reasoning back on because a run looked thin.

---

## Provenance (2026-09-12)

**Why the earlier run carried no verdict, and what changed.** The first
attempt at this bake-off was taken against `qwen3.8-flash-next`, which
reasons whatever the request says, so both arms returned empty drafts and
the file correctly refused to call that a tie. This run is against
`deepseek-v4-flash-0731`, which honours `thinking: disabled` --
`probe_teacher.py` measured 123-143 completion tokens with the flag off
against 3,081-3,736 with it on, on identical briefs.

That second number is why the first honest attempt *still* failed: at the
old `MAX_TOKENS_THINK_ON = 2048` the think-on arm came back 60% empty at a
1,960-token mean, which measures the cap and not the flag. The cap is now
6,144 (worst case observed 4,453, plus the renderer's escalation ladder for
the tail). It binds `exam`, `critique`, `implementation` and `agentic` as
well as this measurement -- every reasoning row was truncating on its first
call and recovering through an escalation nobody had to pay for.

**The table above predates two brief corrections**, made because the run
exposed them; the verdict does not move (think-on offered no invented-number
improvement and cost 17x the wall clock), but the first-pass numbers are
better than it records. Re-measured on the same twenty packs, think-off only:

| first-pass rate | at the run above | after the two fixes |
| --- | ---: | ---: |
| rows gate-clean without repair | 0.00 | **0.45** |
| rows missing a `must_mention` point | 20/20 | **2/20** |
| malformed-prose findings | 10 | **3** |
| median words | 84 | 102 |

The two fixes, both of them contradictions the corpus was carrying:

1. §B.1 removed "cover every `must_mention`" from the system turn and only
   the `grounded` brief said it again, while the gate went on refusing all
   five kinds for it. `POINTS_POLICY` now states it once, in the user turn.
2. The `execution.tca.arrival` addendum said "do not introduce a decision
   price this pack does not contain" while the pack's own anchor is "arrival
   versus decision benchmark" -- the brief forbade what the gate required,
   and that single pairing accounted for 19 of the 19 coverage failures. The
   addendum now asks for the absence to be *named*, which is what §F's target
   prose does ("Arrival is the benchmark; a different decision time is not in
   the pack").
