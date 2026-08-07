#!/usr/bin/env python3
"""Stage 1: supervised fine-tuning of a LoRA adapter on the Cosimo SFT corpus.

Loss is computed on the assistant response only. The prompt is masked out after
the trainer is built, by unsloth's ``train_on_responses_only``, and the result is
verified against the first training example before a single optimizer step runs
(see ``check_masking`` below). That check is the point of this script: a chat
template change would otherwise silently train the model on its own prompts.

Run the pre-flight in TWO steps. --dry-run stops at trainer construction, so it
proves the data, masking, LoRA targets and trainer wiring are sane but never
enters compute_loss and never touches the optimizer. Follow it with a five-step
real run into a throwaway directory, which is what actually exercises the
forward/backward path:

Example:
    ./scripts/04_train_sft.py --dry-run
    ./scripts/04_train_sft.py --run-name sft_smoke --set sft.max_steps=5 --force
    ./scripts/04_train_sft.py --run-name sft

If the smoke run fails inside compute_loss with logits that are missing or a
sentinel, set UNSLOTH_RETURN_LOGITS=1: Unsloth's compiled forward can skip
materialising logits (unsloth_zoo/compiler.py:2254), while TRL 0.24's
SFTTrainer.compute_loss reads outputs.logits unconditionally
(sft_trainer.py:1104). The variable costs ~3.3 GB of bf16 logits at batch 4 x
2048 x vocab 200064, which is affordable in 128 GB unified memory.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

# Unsloth must be imported before transformers/trl so its patches land first.
import unsloth  # noqa: E402,F401  isort:skip
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import set_seed  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from cosimo_ft import chat  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402
from cosimo_ft import modeling  # noqa: E402
from cosimo_ft import tools as tools_mod  # noqa: E402
from cosimo_ft.runlog import RunDir  # noqa: E402

logger = logging.getLogger("train_sft")

STAGE = "sft"

# How many rows of the raw dataset are searched when locating the source record
# of the first tokenized training example (see resolve_masking_report).
ALIGNMENT_WINDOW = 64

# Rows scanned for the truncation report. The persona block put every example
# ~600 tokens closer to max_length, so this is now worth measuring.
TRUNCATION_SAMPLE = 512

# Fraction of sampled rows allowed to lose their final-answer tag before the
# pre-flight complains.
TRUNCATION_WARN_RATE = 0.01


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-name", default=None, help="run directory under runs/")
    parser.add_argument("--train-file", default=None, help="override sft.train_file")
    parser.add_argument("--val-file", default=None, help="override sft.val_file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build data, model and trainer, print the pre-flight report, then exit "
        "without training",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume from the latest checkpoint"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing run directory"
    )
    config_mod.add_config_args(parser)
    return parser


def report_targets() -> list[str]:
    """TensorBoard always; Weights & Biases only when a key is present."""
    targets = ["tensorboard"]
    if os.environ.get("WANDB_API_KEY"):
        targets.append("wandb")
    return targets


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch through transformers' single entry point."""
    set_seed(seed)


def install_chat_template(tokenizer: Any, cfg: dict) -> None:
    """Install the harness chat template, refusing to train without it.

    The stock Phi-4-mini-reasoning template hardcodes "Your name is Phi, an AI
    math expert developed by Microsoft." ahead of every system message. Training a
    Cosimo identity underneath that sentence, or exporting a checkpoint that
    reintroduces it at serving time, is the specific failure this prevents. The
    tokenizer saved next to the adapter therefore carries this template.
    """
    if not chat.apply_chat_template_override(tokenizer, cfg):
        raise SystemExit(
            "chat.template_path is not set, so the tokenizer would keep the vendor "
            "chat template, which prepends the Microsoft identity preamble to every "
            "system message and contradicts the identity being trained.\n"
            "Set chat.template_path (configs/base.yaml ships "
            "configs/chat_template.jinja) before training."
        )


def load_jsonl_split(path: str | list[str], label: str) -> Any:
    """Load the prepared JSONL file(s) for one split as a datasets.Dataset.

    A list is concatenated, which is how the synthetic tool-calling rows from
    scripts/02_prepare_tool_data.py are mixed into the exam corpus. Every listed
    file must exist; an empty one (01b writes those when tools.enabled is false)
    is skipped with a log line rather than silently, because "the tool rows are
    missing" and "the tool rows are empty" are different problems.
    """
    paths = [path] if isinstance(path, str) else list(path)
    if not paths:
        raise SystemExit(f"{label} is empty; nothing to load")

    resolved_paths = []
    for entry in paths:
        resolved = config_mod.harness_path(entry)
        if not resolved.is_file():
            raise SystemExit(
                f"missing {label} file: {resolved}\n"
                "Run scripts/01_prepare_data.py (and scripts/02_prepare_tool_data.py "
                "for the tool-calling rows) first, or point at another file with "
                f"--{label.replace('_', '-')} / --set sft.{label}=<path>."
            )
        if resolved.stat().st_size == 0:
            logger.warning("%s is empty; skipping it", resolved)
            continue
        resolved_paths.append(str(resolved))

    if not resolved_paths:
        raise SystemExit(f"every {label} file is empty: {paths}")
    return load_dataset("json", data_files=resolved_paths, split="train")


def split_files_present(path: str | list[str]) -> bool:
    """True when every file listed for an optional split exists.

    `sft.val_file` is a list (exam rows plus tool rows), so the scalar
    `harness_path(path).is_file()` check cannot be applied to it directly.
    """
    paths = [path] if isinstance(path, str) else list(path)
    return bool(paths) and all(
        config_mod.harness_path(entry).is_file() for entry in paths
    )


def as_int_list(values: Any) -> list[int]:
    """Normalize a labels/input_ids column entry to a plain list of ints."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(v) for v in values]


def normalize(text: str) -> str:
    """Collapse whitespace so decoded text can be compared to source text."""
    return " ".join(str(text).split())


def contains_subsequence(haystack: list[int], needle: list[int]) -> bool:
    """True when ``needle`` appears as a contiguous run inside ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    span = len(needle)
    for i in range(len(haystack) - span + 1):
        if haystack[i] == first and haystack[i : i + span] == needle:
            return True
    return False


def marker_ids(tokenizer: Any, marker: str) -> list[int]:
    """Token ids for a chat marker, as train_on_responses_only matches them."""
    return list(tokenizer.encode(marker, add_special_tokens=False))


def resolve_masking_report(
    trainer: Any,
    tokenizer: Any,
    questions: list[str],
    instruction_part: str,
    response_part: str,
    tag: str,
    index: int = 0,
) -> dict:
    """Decode the supervised and unsupervised spans of one training example.

    ``train_on_responses_only`` may drop fully masked rows, so the source record
    of ``train_dataset[0]`` is located by finding which of the first
    ``ALIGNMENT_WINDOW`` questions appears in the fully decoded sequence. The
    question is used for the human-readable report only; correctness is asserted
    structurally against the marker token ids.

    ``index`` selects the row: 0 is the first (exam) row, and the last row is used
    to verify the synthetic tool-calling conversations, whose masking is a
    different shape -- several assistant turns with a tool result between them.
    """
    example = trainer.train_dataset[index]
    if "labels" not in example:
        raise RuntimeError(
            "train_on_responses_only did not add a 'labels' column to the training "
            "dataset. The trainer was probably built with packing=True, or the "
            "dataset was already tokenized with labels. Response-only masking "
            "cannot be verified, so training is refused."
        )
    input_ids = as_int_list(example["input_ids"])
    labels = as_int_list(example["labels"])
    if len(labels) != len(input_ids):
        raise RuntimeError(
            f"labels/input_ids length mismatch ({len(labels)} vs {len(input_ids)}) "
            "on the first training example; the masking pass is not aligned with "
            "the tokenized sequence."
        )

    supervised_ids = [t for t, lab in zip(input_ids, labels) if lab != -100]
    masked_ids = [t for t, lab in zip(input_ids, labels) if lab == -100]
    supervised = tokenizer.decode(supervised_ids)
    masked = tokenizer.decode(masked_ids)

    # Structural evidence: the supervised span must start exactly where the
    # response marker ends, and must contain no instruction marker.
    response_ids = marker_ids(tokenizer, response_part)
    instruction_ids = marker_ids(tokenizer, instruction_part)
    tail = masked_ids[-len(response_ids) :] if response_ids else []
    starts_after_response = bool(response_ids) and tail == response_ids
    instruction_in_supervised = bool(instruction_ids) and contains_subsequence(
        supervised_ids, instruction_ids
    )

    # Align against the whole sequence, not the masked part: a completely
    # unmasked example has no masked part, and that is exactly the failure the
    # next check has to be able to name.
    full_norm = normalize(tokenizer.decode(input_ids))
    question = None
    for candidate in questions[:ALIGNMENT_WINDOW]:
        if candidate and normalize(candidate) in full_norm:
            question = candidate
            break

    return {
        "n_tokens": len(input_ids),
        "n_supervised_tokens": len(supervised_ids),
        "n_masked_tokens": len(masked_ids),
        "supervised_text": supervised,
        "masked_text": masked,
        "question": question,
        "starts_after_response_marker": starts_after_response,
        "instruction_marker_in_supervised": instruction_in_supervised,
        "response_part": response_part,
        "instruction_part": instruction_part,
        # A tool-calling row: either a synthetic one from 02_prepare_tool_data.py
        # or an `agentic` corpus record. Identified by the schema markers in the
        # prompt, which every such row carries -- unlike <tool_call>, which the
        # tools.no_call_rate rows deliberately lack.
        "is_tool_row": tools_mod.TOOL_SCHEMA_OPEN in tokenizer.decode(input_ids),
        # Whether this row is under the FINAL ANSWER grading contract. The exam
        # protocol lives in the system block and 01_prepare_data.py attaches it
        # to exam rows only, so the tag appearing in the *masked* span is an
        # exact test -- and it needs no column the trainer's text-only view of
        # the dataset has already dropped.
        "expects_final_answer": tag in masked,
    }


def check_masking(report: dict, tag: str) -> None:
    """Assert response-only masking is correct. Raises with an actionable message."""
    supervised = report["supervised_text"]
    question = report["question"]

    if report["n_supervised_tokens"] == 0 or not supervised.strip():
        raise RuntimeError(
            "response-only masking left NO supervised tokens on the first training "
            "example: every label is -100 and the run would learn nothing.\n"
            "Check that chat.instruction_part and chat.response_part in the config "
            "match the markers the tokenizer's chat template actually emits, and "
            "that model.max_seq_length is long enough that the response marker "
            "survives truncation."
        )

    # Only exam rows are under the final-answer contract. Analysis, abstention,
    # agentic and implementation records -- and the synthetic tool rows -- render
    # with exam=False and carry no tag by design: it is an exam-grading artefact
    # and letting it leak into an open-ended answer is the style collapse this
    # corpus was rebuilt to avoid. Their supervised span is checked structurally
    # below, like every other row.
    if report["expects_final_answer"] and tag not in supervised:
        raise RuntimeError(
            f"the supervised span does not contain the final-answer tag {tag!r}.\n"
            "Either the completion was truncated before its last line (raise "
            "model.max_seq_length), or chat.response_part points at the wrong "
            "marker so the mask starts in the middle of the answer.\n"
            f"supervised span was: {supervised!r}"
        )

    if not report["expects_final_answer"] and tag in supervised:
        raise RuntimeError(
            f"a non-exam row is being trained to emit {tag!r}. Its system block "
            "carries no exam protocol, so the tag in its target can only have "
            "come from the supervised text itself.\n"
            "That contract belongs to exam records alone; training it onto "
            "analysis, abstention, agentic or implementation rows is how the "
            "model learns that every answer is a five-step exam trace.\n"
            "Re-run scripts/01_prepare_data.py, whose validation gate checks "
            f"this for every row.\nsupervised span was: {supervised!r}"
        )

    # The prompt-exclusion assertion is structural, not textual. Matching question
    # words against the supervised span both false-positives (reasoning traces
    # legitimately restate the setup: "at the end of each month") and misses short
    # leaks. The marker token ids are exact: the supervised span must begin
    # immediately after the last response marker, and must contain no instruction
    # marker, which catches a single leaked token.
    if (
        not report["instruction_marker_in_supervised"]
        and report["starts_after_response_marker"]
    ):
        return

    if report["instruction_marker_in_supervised"]:
        raise RuntimeError(
            f"the instruction marker {report['instruction_part']!r} appears INSIDE "
            "the supervised span: the prompt is not masked and the model would be "
            "trained to generate its own questions.\n"
            "chat.instruction_part / chat.response_part do not match the chat "
            "template. For unsloth/Phi-4-mini-reasoning they must be '<|user|>' "
            "and '<|assistant|>'.\n"
            f"question was: {question!r}"
        )

    raise RuntimeError(
        "the supervised span does not begin immediately after the response marker "
        f"{report['response_part']!r}, so the mask boundary is in the wrong place "
        "and prompt tokens are being trained on (or answer tokens dropped).\n"
        "Check chat.response_part against the markers configs/chat_template.jinja "
        "emits; for unsloth/Phi-4-mini-reasoning it must be '<|assistant|>'.\n"
        f"masked span ends: {report['masked_text'][-120:]!r}\n"
        f"supervised span starts: {supervised[:120]!r}"
    )


def truncation_report(trainer: Any, tokenizer: Any, max_length: int, tag: str) -> dict:
    """Measure how often max_length is eating the end of the supervised span.

    The identity block added ~600 tokens to every example, so rows that used to
    sit comfortably inside max_length can now be cut. A row truncated past its
    ``FINAL ANSWER:`` line still trains -- it just teaches the model to stop
    before answering -- so unsloth's fully-masked-row filter does not catch it.
    """
    dataset = trainer.train_dataset
    n = min(TRUNCATION_SAMPLE, len(dataset))
    at_cap = 0
    missing_tag = 0
    non_exam_rows = 0
    for i in range(n):
        row = dataset[i]
        input_ids = as_int_list(row["input_ids"])
        labels = as_int_list(row["labels"])
        if len(input_ids) >= max_length:
            at_cap += 1
        supervised = [t for t, lab in zip(input_ids, labels) if lab != -100]
        masked = [t for t, lab in zip(input_ids, labels) if lab == -100]
        # Only exam rows carry the final-answer tag, and their system block is
        # the only one carrying the exam protocol -- so the tag in the masked
        # (prompt) span identifies them. Scoring the other four record types
        # here would report a fabricated truncation rate of ~78%.
        if tag not in tokenizer.decode(masked):
            non_exam_rows += 1
            continue
        if tag not in tokenizer.decode(supervised):
            missing_tag += 1
    scored = n - non_exam_rows
    return {
        "sampled": n,
        "at_length_cap": at_cap,
        "non_exam_rows": non_exam_rows,
        "missing_final_answer": missing_tag,
        "missing_rate": (missing_tag / scored) if scored else 0.0,
    }


def print_truncation_report(report: dict, max_length: int) -> None:
    print(
        f"truncation scan: {report['at_length_cap']}/{report['sampled']} rows at the "
        f"{max_length}-token cap, {report['missing_final_answer']} without a "
        "final-answer tag in the supervised span "
        f"({report['non_exam_rows']} non-exam rows excluded, they carry no tag "
        "by design)"
    )
    if report["missing_rate"] > TRUNCATION_WARN_RATE:
        logger.warning(
            "%.1f%% of sampled rows lose their final-answer line to truncation. "
            "Those rows teach the model to stop before answering. Raise "
            "model.max_seq_length (check the token-length percentiles in "
            "data/processed/split_manifest.json) or shorten the prompt.",
            100.0 * report["missing_rate"],
        )


def print_masking_report(report: dict) -> None:
    print("-" * 72)
    print("response-only masking report (first training example)")
    print(
        f"tokens: {report['n_tokens']} "
        f"(supervised {report['n_supervised_tokens']}, "
        f"masked {report['n_masked_tokens']})"
    )
    print("-" * 72)
    print(
        "UNSUPERVISED span (labels == -100), must contain the identity block and "
        "the question:"
    )
    print(report["masked_text"])
    print("-" * 72)
    print("SUPERVISED span (labels != -100), must be the answer only:")
    print(report["supervised_text"])
    print("-" * 72)


def build_sft_config(cfg: dict, output_dir: Path, logging_dir: Path, has_eval: bool):
    """Build the TRL 0.24.0 SFTConfig from the resolved ``sft`` block."""

    def s(key: str, default: Any = None) -> Any:
        return config_mod.get(cfg, f"sft.{key}", default)

    # Without an eval file there is nothing to evaluate against; asking the
    # Trainer for periodic eval would then crash mid-run.
    eval_strategy = s("eval_strategy", "steps") if has_eval else "no"

    return SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        report_to=report_targets(),
        seed=int(s("seed", config_mod.get(cfg, "seed", 3407))),
        # TRL 0.24.0: the field is `max_length`, not `max_seq_length`.
        max_length=int(config_mod.get(cfg, "model.max_seq_length", 2048)),
        dataset_text_field=s("dataset_text_field", "text"),
        dataset_num_proc=s("dataset_num_proc", 4),
        # Packing concatenates rows into fixed-length blocks, which destroys the
        # 1:1 mapping between a sequence and a prompt/response pair that
        # train_on_responses_only relies on. It must stay False.
        packing=False,
        # The dataset is reduced to its text column below, so TRL treats it as
        # language modeling; masking is entirely unsloth's job.
        completion_only_loss=False,
        per_device_train_batch_size=int(s("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(s("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(s("gradient_accumulation_steps", 8)),
        num_train_epochs=float(s("num_train_epochs", 1)),
        max_steps=int(s("max_steps", -1)),
        learning_rate=float(s("learning_rate", 2.0e-4)),
        lr_scheduler_type=s("lr_scheduler_type", "cosine"),
        warmup_ratio=float(s("warmup_ratio", 0.03)),
        weight_decay=float(s("weight_decay", 0.01)),
        optim=s("optim", "adamw_torch_fused"),
        max_grad_norm=float(s("max_grad_norm", 1.0)),
        bf16=bool(s("bf16", True)),
        fp16=bool(s("fp16", False)),
        # Unsloth installed its own checkpointing on the model already.
        gradient_checkpointing=bool(s("gradient_checkpointing", False)),
        logging_steps=int(s("logging_steps", 10)),
        eval_strategy=eval_strategy,
        eval_steps=int(s("eval_steps", 250)),
        save_strategy=s("save_strategy", "steps"),
        save_steps=int(s("save_steps", 250)),
        save_total_limit=int(s("save_total_limit", 3)),
        dataloader_num_workers=int(s("dataloader_num_workers", 4)),
        group_by_length=bool(s("group_by_length", False)),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    cfg = config_mod.load_config(stage=STAGE, extra=args.config, overrides=args.set)

    run_name = args.run_name or config_mod.get(cfg, "sft.run_name", "sft")
    runs_dir = config_mod.get(cfg, "paths.runs_dir", "runs")
    run = RunDir(runs_dir, run_name)
    if run.exists() and not (args.force or args.resume or args.dry_run):
        raise SystemExit(
            f"{run.root} already exists. Pass --resume to continue it, --force to "
            "overwrite it, or --run-name to train into a different directory."
        )
    if run.exists() and args.force and not (args.resume or args.dry_run):
        # RunDir.create() is mkdir(exist_ok=True), so without this --force would
        # only bypass the guard: stale checkpoints/checkpoint-* would survive and a
        # later --resume would silently continue them under different
        # hyperparameters. manifest.json is written only after train() returns, so
        # a crashed run is indistinguishable from a fresh one on disk.
        logger.warning("--force: removing the existing run directory %s", run.root)
        shutil.rmtree(run.root)

    seed = int(config_mod.get(cfg, "sft.seed", config_mod.get(cfg, "seed", 3407)))
    seed_everything(seed)

    text_field = config_mod.get(cfg, "sft.dataset_text_field", "text")
    tag = config_mod.get(cfg, "prompt.final_answer_tag", "FINAL ANSWER:")

    # --- data -------------------------------------------------------------
    train_path = args.train_file or config_mod.get(cfg, "sft.train_file")
    val_path = args.val_file or config_mod.get(cfg, "sft.val_file")
    raw_train = load_jsonl_split(train_path, "train_file")
    raw_val = None
    if val_path and split_files_present(val_path):
        raw_val = load_jsonl_split(val_path, "val_file")
    else:
        logger.warning("no validation file at %s; periodic eval disabled", val_path)

    # `question` is not trained on; it is what the masking check asserts against.
    missing = [c for c in (text_field, "question") if c not in raw_train.column_names]
    if missing:
        raise SystemExit(
            f"{train_path} is missing the column(s) {missing} (found "
            f"{raw_train.column_names}). Regenerate it with 01_prepare_data.py."
        )
    first_example = raw_train[0]
    questions = raw_train.select(range(min(ALIGNMENT_WINDOW, len(raw_train))))[
        "question"
    ]

    # TRL 0.24.0 routes any dataset carrying both `prompt` and `completion`
    # through its prompt-completion path, which ignores dataset_text_field and
    # builds its own completion mask. Our JSONL carries those columns for the
    # eval tooling, so the training view is reduced to the single text column:
    # then TRL tokenizes `text` as plain language modeling and unsloth owns the
    # masking.
    train_dataset = raw_train.select_columns([text_field])
    eval_dataset = raw_val.select_columns([text_field]) if raw_val is not None else None

    # --- model ------------------------------------------------------------
    model, tokenizer = modeling.load_for_training(cfg)
    install_chat_template(tokenizer, cfg)
    requested_targets = config_mod.get(cfg, "lora.target_modules", "auto")
    targets = modeling.resolve_target_modules(model, requested_targets)
    model = modeling.attach_lora(model, cfg)

    # --- trainer ----------------------------------------------------------
    if args.dry_run:
        # A dry run must not create or disturb runs/<name>.
        scratch = Path(tempfile.mkdtemp(prefix="cosimo-sft-dryrun-"))
        output_dir, logging_dir = scratch / "checkpoints", scratch / "tb"
    else:
        run.create("tb", "checkpoints", "adapter")
        output_dir, logging_dir = run.checkpoints_dir, run.tb_dir

    sft_args = build_sft_config(cfg, output_dir, logging_dir, eval_dataset is not None)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # TRL 0.24.0: the tokenizer argument is `processing_class`.
        processing_class=tokenizer,
    )

    # Response-only training. Applied AFTER construction because it rewrites the
    # trainer's already-tokenized datasets in place.
    instruction_part = config_mod.get(cfg, "chat.instruction_part", "<|user|>")
    response_part = config_mod.get(cfg, "chat.response_part", "<|assistant|>")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
        # Explicit, and small on purpose. Left to itself unsloth_zoo picks
        # num_proc ~= cpu_count+4 and forks datasets.map workers *after* the model
        # is on the GPU (a live CUDA context in a forked child is unsupported), and
        # its auto-sizing path imports psutil, which is not in the pinned
        # dependency group and unsloth_zoo is installed --no-deps. The masking map
        # is cheap, so nothing is lost.
        num_proc=int(config_mod.get(cfg, "sft.masking_num_proc", 2)),
    )

    report = resolve_masking_report(
        trainer, tokenizer, list(questions), instruction_part, response_part, tag
    )
    print_masking_report(report)
    check_masking(report, tag)
    logger.info("response-only masking verified on the first training example")

    # The tool rows are appended after the corpus (configs/sft.yaml lists them
    # second), so the last row is one of them. Their masking is a different
    # shape -- several assistant turns with tool results between them -- and it is
    # only correct because the chat template renders that tool result as a
    # <|user|> turn, which is what train_on_responses_only masks on. That is
    # load-bearing and unproven at this point in the run, so it is checked rather
    # than assumed. The corpus's own `agentic` records have the same shape and
    # are checked by the same assertion when one is sampled.
    tail_report = resolve_masking_report(
        trainer,
        tokenizer,
        list(questions),
        instruction_part,
        response_part,
        tag,
        index=len(trainer.train_dataset) - 1,
    )
    if tail_report["is_tool_row"]:
        check_masking(tail_report, tag)
        logger.info("response-only masking verified on a tool-calling example")
    else:
        logger.warning(
            "the last training row is not a tool-calling example, so multi-turn "
            "tool masking was not verified. Did scripts/02_prepare_tool_data.py "
            "run, and is tools.enabled true?"
        )

    max_length = int(config_mod.get(cfg, "model.max_seq_length", 2048))
    trunc = truncation_report(trainer, tokenizer, max_length, tag)
    print_truncation_report(trunc, max_length)

    print(f"resolved LoRA target modules: {targets}")
    model.print_trainable_parameters()
    print(f"train rows: {len(trainer.train_dataset)}")
    if eval_dataset is not None:
        print(f"eval rows: {len(trainer.eval_dataset)}")

    if args.dry_run:
        print("-" * 72)
        print("first rendered training example:")
        print(first_example[text_field])
        print("-" * 72)
        print(f"dry run: nothing was trained. Scratch trainer output: {scratch}")
        return

    # --- train ------------------------------------------------------------
    run.save_config(cfg)
    run.save_env()
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)

    trainer.model.save_pretrained(str(run.adapter_dir))
    tokenizer.save_pretrained(str(run.adapter_dir))

    run.write_manifest(
        {
            "stage": STAGE,
            "config_hash": config_mod.config_hash(cfg),
            "base_id": config_mod.get(cfg, "model.base_id"),
            "adapter_dir": str(run.adapter_dir),
            "lora_target_modules": targets,
            "chat_template_path": config_mod.get(cfg, "chat.template_path"),
            "truncation_scan": trunc,
            "train_rows": len(trainer.train_dataset),
            "eval_rows": len(trainer.eval_dataset) if eval_dataset is not None else 0,
            "train_files": {"train": train_path, "val": val_path},
            "metrics": dict(result.metrics),
        }
    )

    print(f"adapter: {run.adapter_dir}")
    print(f"tensorboard: {run.tb_dir}")
    print(f"manifest: {run.path('manifest.json')}")


if __name__ == "__main__":
    main()
