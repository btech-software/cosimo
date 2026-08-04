> **Superseded by [`003_dataset_rev1_prompt.md`](003_dataset_rev1_prompt.md).** Kept as the
> provenance record for `btech-software/cosimo-cfa-frm-71k`: this is the brief that produced the
> corpus, and it explains why that corpus looks the way it does. It was executed, trained on and
> measured; `003` documents what the measurements showed and what the next corpus must do
> differently. Do not run this one again.

Generate a large-scale, high-quality synthetic dataset for fine-tuning Cosimo — a financial reasoning specialist based on microsoft/Phi-4-mini-flash-reasoning.

Goal: Produce 50k–200k original, curriculum-aligned examples covering CFA Levels I–III and core FRM topics. Each example must contain:
- a novel question (MCQ, vignette, calculation, or constructed-response style)
- a complete, verified step-by-step reasoning trace (Chain-of-Thought) with formulas and explicit assumptions
- correct final answer
- metadata (topic, subtopic, difficulty, question type, pitfalls addressed)

The dataset must be balanced across the full public CFA/FRM taxonomy, include preference pairs (strong vs flawed reasoning), and be ready for direct SFT + DPO/ORPO use. All content must be original (inspired by public Learning Outcome Statements only — never copy proprietary exam material).

Concrete quality bar (non-negotiable):
A curated gold-standard set of expert-level CFA-style questions with perfect, numerically verified reasoning traces. Sources for the bar: public CFA Institute sample questions + Learning Outcome Statements, high-quality open financial education resources, and rigorously verified synthetic exemplars. Every generated example (and every batch) must win a blind A/B comparison against this gold bar on correctness, reasoning depth/clarity, numerical accuracy, educational value, and absence of hallucinations. The finished dataset must also win on overall curriculum coverage, diversity, and statistical quality metrics.

Instructions for the lead agent:
- Divide the entire goal into the smallest pieces that can be independently built and judged (taxonomy, generation pipelines per topic/difficulty, verification harnesses with code execution, preference-pair creation, diversity/coverage trackers, storage format, etc.).
- For every important piece, fan out a builder subagent and a completely separate critic subagent in fresh contexts.
- Each critic must inspect the real output (files, numbers, samples), perform a blind A/B comparison against the gold bar whenever possible, identify the single largest remaining gap, and send the work back. Keep looping indefinitely on each piece until it wins or improvements become negligible.
- Maintain a simple live progress page (progress.md or progress.html) that continuously shows: total examples generated, coverage heat-map, quality scores, sample winning/losing pairs, current bottlenecks, and dataset shards.
- Use subagents extensively. Write all artifacts to disk (JSONL shards, verification scripts, taxonomy files, gold-bar references). Prefer incremental, resumable generation so the loop can run for days.
- Do not prescribe architecture, exact decomposition, or fixed number of rounds. Decide the best technical approach yourself.

Start now. Build the gold bar first if it does not already exist, then enter the full Gauntlet Loop. Do not stop until the dataset meets or exceeds the bar at scale.
