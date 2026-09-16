# Examples — one real row per record type

Eight files, one per record type, one row each. They answer "what shape is a v3
row?" and nothing else: they are documentation, not training data and not a
corpus sample.

| file | provenance |
| --- | --- |
| `analysis`, `grounded`, `memo`, `critique`, `abstention` | live teacher, prose lane (think off) |
| `agentic`, `implementation` | live teacher, reasoning lane (think on) |
| `exam` | deterministic — the exam renderer builds it from the pack, so there is no teacher to be live |

Drawn from eight pack coordinates across five work types: eight examples of one
computer would show the schema and hide the range.

Regenerate with:

    uv run --group corpus python dataset/examples/v3/make_examples.py

It lifts a real row per record type out of a rendered corpus (`COSIMO_V3_OUT`,
else `dataset/shards/v3`) and falls back to a scripted teacher only for a type
no live render has produced, printing which is which.
`test_examples_and_leak_gate.py` asserts every example clears the verify board
and carries no teacher transcript.

**Why "real" is load-bearing.** These files used to be rendered through the real
renderers with a *scripted* teacher supplying the prose. When a smoke corpus
swept the directory up, the adapter learned that prose and answered a VaR
question with the fixture harness's own opening line. `smoke_corpus.py` excludes
this directory by name regardless — an example is documentation, and training
reads the shard tree.

## `_teacher_logs/`

Where the factory's transcripts go when `COSIMO_V3_KEEP_TEACHER_MESSAGES=1` is
set — never a shard, never trainable, because they carry the teacher's system
turn. Gitignored.
