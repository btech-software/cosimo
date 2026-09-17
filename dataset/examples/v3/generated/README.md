# Generated — the 2026-09-16 live render, for independent evaluation

140 student rows copied out of the rendered corpus (`dataset/shards/v3/sft/`)
so they can be read and scored outside the pipeline. The shard tree is
gitignored; this copy is not. The rows are unchanged — same bytes, sorted by id.

**Not training data.** `smoke_corpus.py` reads the shard tree and never this
directory, and nothing else walks `examples/`. The training corpus still holds
these rows; this is a copy, not a move.

| file | rows | families |
| --- | --- | --- |
| `analysis.jsonl` | 40 | rates_book 16, mid_cap_swing 7, us_small_cap 7, specialty_retail 6, mature_consumer 4 |
| `critique.jsonl` | 38 | rates_book 10, mid_cap_swing 8, us_small_cap 8, specialty_retail 6, mature_consumer 6 |
| `abstention.jsonl` | 26 | rates_book 23, mid_cap_swing 3 |
| `memo.jsonl` | 25 | rates_book 9, us_small_cap 8, specialty_retail 4, mature_consumer 4 |
| `grounded.jsonl` | 8 | rates_book 4, us_small_cap 2, mid_cap_swing 2 |
| `agentic.jsonl` | 3 | large_cap_intraday (ranks 1–3) |

## Provenance

- **Teacher:** `deepseek-v4-flash-0731` on both lanes (`verify_v3` axis 14 pins a
  single model). Prose kinds `analysis`, `memo`, `grounded`, `abstention` ran with
  thinking off; `critique` and `agentic` ran on the reasoning lane, thinking on.
- **Rendered with** `make v3-render WORK=… FAMILY=… TYPES=<one kind> LIMIT=… LIVE=1`,
  one bounded slice per kind so no kind consumed another's budget.
- **Selection rule:** every prose row on the five families that had no prose
  before this render (`risk.market.var_es.rates_book`,
  `execution.tca.arrival.mid_cap_swing`,
  `portfolio.attribution.brinson_carino.us_small_cap`,
  `valuation.equity.dcf.mature_consumer`,
  `valuation.equity.multiples.specialty_retail`), plus every agentic row.
- **Checked on copy:** no reserved eval coordinate, no teacher brief or
  fingerprint, `verified: true`, and a clean `verify_v3` board (quick axes).

## Read with this in mind

- **Survivors only.** Every row here passed the render gates (invented numbers,
  `must_mention`, register shape, the agentic message band). Rows the gates
  refused went to `dataset/shards/v3/dead_letter/` and are not included, so
  pass rates computed on this directory are biased upward. Scoring the
  generator, not just its output, needs the dead letters too.
- **Two committed one-row examples are among these rows.** The top-level
  `analysis`, `abstention`, `grounded`, `memo`, `critique` and `agentic`
  examples were drawn from this same render.
- **Known style issue:** the rank-2 agentic row's answer opens with a markdown
  heading (`## Desk Answer`) in the `desk_chat` register. The register gate
  refuses memo labels (`Finding:`, `Call:`), not generic headings, so it passed.
