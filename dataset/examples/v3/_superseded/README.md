# Superseded captures — evidence, not examples

Nothing in this directory is an example of what the corpus produces today, and
nothing here is training material. Two files, kept for one reason each.

`live_analysis.jsonl` — six `analysis` rows, `qwen3.8-flash-next`, think forced
on. The first live sample the project ever took. Three of the six would not
ship under the gates that existed a week later, which is why it was kept then
and why it is kept now: it is the standing proof that a captured sample can
stop clearing a tightened gate, and that this is a finding about the gate
rather than a failure in the file.

`live_three_registers.jsonl` — nine rows, `deepseek-v4-flash-0731`, think off,
board-green at capture. The **before** picture of the v3.2 amendment, and the
evidence for every claim in the comparison table in `../README.md`: 288-word
median, four of nine contradicting their own packs, `analysis` and `grounded`
sharing a first sentence, a `memo` on a `desk_chat` family. Delete it and the
amendment's central claim becomes an assertion nobody can check.

Both are excluded from `dataset/tools/smoke_corpus.py` by name, and the
underscore prefix keeps them out of any glob that walks the corpus.
