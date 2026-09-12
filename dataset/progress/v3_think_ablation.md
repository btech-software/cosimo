# v3 think ablation (amendment §B)

The amendment's test, run: *20 analysis rows think-off vs think-on;
promote think-on only if the invented-number rate drops by >= 2 points.*

- record type: `analysis`
- teacher: `deepseek-v4-flash-0731`
- rows asked per arm: 20
- budgets: 800 think-off / 2048 think-on

| metric | think-off | think-on |
| --- | ---: | ---: |
| rows answered | 20 | 20 |
| invented-number rate | 0.000 | 0.000 |
| gate-clean rate | 0.000 | 0.000 |
| empty-draft rate | 0.000 | 0.600 |
| median words | 82 | 0 |
| mean completion tokens | 131 | 1960 |
| wall seconds | 113.6 | 1215.4 |

## NOT YET MEASURED

This run produced no usable comparison, so it carries no verdict and
does not satisfy §F's precondition. `dataset_build.sh full` stays
blocked, which is the point: a bake-off that measured nothing must
not read like one that measured a tie.

- **think-on arm**: 60% of drafts came back empty at a 2048-token budget -- the teacher spent the budget before it wrote a word, so this arm measured the cap, not the flag

### What this usually means

Most likely the think flag never reached the model, so both arms
thought and the comparison had nothing to compare. A
vLLM/SparkInfer serve reads the toggle from the chat template via
`chat_template_kwargs` and ignores DeepSeek's cloud
`thinking: {type: ...}`; omitting the kwarg lets the server's own
`thinking: true` default stand. That is exactly how this tool first
returned a tidy 0.000-vs-0.000 on forty calls that every one of
them returned empty.

Check `think_present` in the per-row detail (`--json`). If it is
true on the think-off arm, fix the request dialect before touching
budgets. Only if the reasoning is genuinely unstoppable does this
help:

    COSIMO_V3_THINK_OVERHEAD=8000 COSIMO_V3_LIVE=1 \
      uv run --group corpus python dataset/tools/think_ablation.py

Raising it is a statement about the *endpoint*, not about the corpus:
the amendment's 800/2048 are what an answer costs, and the overhead is
what this particular teacher spends before writing one.


**Δ invented-number rate: +0.0 points** (bar: ≥ 2 points to promote)  — meaningless while an arm is degenerate, see above
