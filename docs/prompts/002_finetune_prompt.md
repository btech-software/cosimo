Create a complete, production-ready fine-tuning harness and execution pipeline for Cosimo. Do NOT run any actual fine-tuning, training loops, or long GPU jobs.

Goal: Produce a clean, self-contained directory in the repository that I can clone onto an NVIDIA DGX Spark and execute myself. The directory must contain everything needed to:
- Set up the environment (Unsloth + dependencies)
- Load and format the dataset https://huggingface.co/datasets/btech-software/cosimo-cfa-frm-71k
- Run Supervised Fine-Tuning (SFT) with Unsloth on microsoft/Phi-4-mini-flash-reasoning (or the closest Unsloth equivalent such as unsloth/Phi-4-mini-reasoning)
- Run preference optimization (DPO or ORPO) using the preference pairs in the dataset
- Benchmark the original base model and any future fine-tuned checkpoints head-to-head
- Iterate safely with clear configs, logging, and reproducibility

The final deliverable is a well-documented directory "jobs/fine-tune" in Git repository (or folder structure) with:
- Environment setup (requirements, Docker or conda/uv instructions optimized for DGX Spark)
- Data preparation scripts (download, split, format into Unsloth/TRL-ready chat or completion format, create train/val/test, handle preference pairs)
- Training scripts for SFT and for preference optimization (configurable via YAML or CLI)
- Evaluation harness that scores both the base model and any checkpoint on held-out Cosimo data + general math/reasoning benchmarks
- Baseline measurement script that records the original model’s performance (so later fine-tunes can be compared)
- Clear README with exact commands I must run on the DGX Spark
- Live progress page showing status of the harness itself

Concrete quality bar (non-negotiable):
The harness must be complete, correct, and ready to run without further coding. A critic must be able to inspect the repository and confirm:
1. All scripts are present, executable, and free of obvious bugs
2. Data loading and formatting correctly handle the Cosimo schema (question + reasoning_trace + answer + preference_pair)
3. Training scripts follow current Unsloth + frontier best practices (response-only training, proper LoRA targets, recommended hyperparameters as defaults, logging, checkpointing)
4. Evaluation harness produces comparable metrics for base vs fine-tuned models
5. README contains a single, clear “how to run on DGX Spark” sequence
6. No training is executed by the agent — only scaffolding is created

Principles to follow (from Unsloth and top labs):
- High-quality verified data first
- Multi-stage design: SFT → preference optimization (DPO/ORPO)
- Train on completions/responses only
- Target all major linear modules, sensible LoRA rank/alpha defaults, response-only loss
- Strong evaluation and baseline comparison built in from day one
- Full reproducibility (seeds, configs, exact commands)

Instructions for the lead agent:
- Divide the work into the smallest independently buildable and judgable pieces (repo structure, environment files, data prep, SFT script, preference script, evaluation harness, baseline runner, README, configs, progress tracking).
- For every important piece, spawn a builder subagent and a completely separate critic subagent in fresh contexts.
- Each critic must inspect the real files on disk, compare them against the quality bar and against current Unsloth + post-training best practices, identify the single largest remaining gap, and send the work back. Keep looping until the entire harness wins the bar.
- Maintain a simple live progress page (progress.md) that shows which components are complete, open issues, and overall readiness.
- Write everything to disk as clean, documented code and configuration. Do not execute training or long-running GPU jobs. Do not download large model weights beyond what is strictly necessary for script validation.
- Do not prescribe the exact file layout or number of rounds. Decide the best technical approach yourself.

Start now. Build the evaluation harness and baseline measurement capability first, then complete the rest of the scaffolding until the repository is fully ready for me to clone and run on the DGX Spark.
