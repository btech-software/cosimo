# v3 think ablation (amendment §B)

The amendment's test, run: *20 analysis rows think-off vs think-on;
promote think-on only if the invented-number rate drops by >= 2 points.*

- record type: `analysis`
- teacher: `qwen3.8-flash-next`
- rows asked per arm: 20
- budgets: 800 think-off / 2048 think-on

| metric | think-off | think-on |
| --- | ---: | ---: |
| rows answered | 20 | 20 |
| invented-number rate | 0.000 | 0.000 |
| gate-clean rate | 0.000 | 0.000 |
| empty-draft rate | 1.000 | 1.000 |
| median words | 0 | 0 |
| mean completion tokens | 800 | 2048 |
| wall seconds | 492.2 | 1295.3 |

## NOT YET MEASURED

This run produced no usable comparison, so it carries no verdict and
does not satisfy §F's precondition. `dataset_build.sh full` stays
blocked, which is the point: a bake-off that measured nothing must
not read like one that measured a tie.

- **think-off arm**: 100% of drafts came back empty at a 800-token budget -- the teacher spent the budget before it wrote a word, so this arm measured the cap, not the flag
- **think-on arm**: 100% of drafts came back empty at a 2048-token budget -- the teacher spent the budget before it wrote a word, so this arm measured the cap, not the flag

### What this usually means

A teacher that reasons *unconditionally*. §B's budgets assume the
think flag decides whether a chain of thought is produced -- 800
tokens being twice the widest prose band once nothing reasons first.
A model that reasons whatever the flag says spends that budget before
it reaches an answer and returns `content: null` with
`finish_reason: length`, identically in both arms.

Check `think_present` in the per-row detail (`--json`). If it is true
on the think-off arm, the flag is not reaching the model and the
budgets need the teacher's reasoning overhead added to them:

    COSIMO_V3_THINK_OVERHEAD=8000 COSIMO_V3_LIVE=1 \
      uv run --group corpus python dataset/tools/think_ablation.py

Raising it is a statement about the *endpoint*, not about the corpus:
the amendment's 800/2048 are what an answer costs, and the overhead is
what this particular teacher spends before writing one.


**Δ invented-number rate: +0.0 points** (bar: ≥ 2 points to promote)  — meaningless while an arm is degenerate, see above
