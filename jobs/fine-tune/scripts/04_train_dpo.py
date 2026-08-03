#!/usr/bin/env python3
"""Stage 2: preference optimisation (DPO) on top of the SFT adapter.

The base model is loaded and the adapter produced by 03_train_sft.py is attached
as *trainable*. No reference model is constructed: with a PEFT model and
``ref_model=None``, TRL runs the reference forward pass through the same weights
with the adapter disabled, which costs no extra memory.

Note what that reference actually is. Disabling the adapter yields the **base**
model, i.e. the pre-SFT weights -- not the SFT policy. So ``beta`` is a KL pull
back toward the untuned base, working *against* the formatting SFT installed, not
anchoring the model to it. If DPO degrades FINAL ANSWER compliance, RAISING beta
makes that worse, not better; lower beta or fewer steps is the fix. Pointing TRL
at the SFT policy as reference would need an explicit ``ref_model`` (a second set
of weights) or a second adapter named via ``ref_adapter_name``.

--dry-run stops at trainer construction: it never enters compute_loss and
never takes an optimizer step. Follow it with a short real run into a
throwaway directory before committing to the full one (see 03_train_sft.py
for the UNSLOTH_RETURN_LOGITS=1 mitigation if the loss path fails).

Example:
    ./scripts/04_train_dpo.py --sft-adapter runs/sft/adapter --dry-run
    ./scripts/04_train_dpo.py --sft-adapter runs/sft/adapter --run-name dpo_smoke --set dpo.max_steps=5 --force
    ./scripts/04_train_dpo.py --sft-adapter runs/sft/adapter
"""

from __future__ import annotations

import argparse
import json
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

# PatchDPOTrainer has to run before trl is imported. In unsloth 2026.8.1 it is a
# no-op (the DPO patches were folded into the main import), but older and newer
# releases do real work here, so the call stays — guarded, because it is not part
# of unsloth's stable surface.
try:  # noqa: E402
    from unsloth import PatchDPOTrainer

    PatchDPOTrainer()
except ImportError:  # pragma: no cover - depends on the installed unsloth build
    pass

from datasets import load_dataset  # noqa: E402
from transformers import set_seed  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

from cosimo_ft import chat  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402
from cosimo_ft import modeling  # noqa: E402
from cosimo_ft.runlog import RunDir  # noqa: E402

logger = logging.getLogger("train_dpo")

STAGE = "dpo"

# The three columns TRL's preference pipeline consumes, in "standard" text form.
PREFERENCE_COLUMNS = ["prompt", "chosen", "rejected"]


# Rows scanned for the truncation report on the prepared preference pairs.
TRUNCATION_SAMPLE = 512

# Fraction of sampled rows allowed to hit a cap before the pre-flight complains.
TRUNCATION_WARN_RATE = 0.01


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile; no numpy dependency at module scope."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def truncation_scan(
    trainer: Any, max_prompt_length: int, max_completion_length: int
) -> dict:
    """Measure how often the prompt/completion caps are actually biting.

    Both caps are lossy in a direction that matters here:

    * the prompt is truncated from the LEFT, unconditionally
      (``dpo_trainer.py:738`` / ``orpo_trainer.py:505``), and the left of our
      prompt is the identity block -- so an over-long prompt trains against a
      persona-less prefix that does not match how the model is served;
    * the completion is truncated from the RIGHT
      (``dpo_trainer.py:740-742``), and the right of our completion is the
      ``FINAL ANSWER:`` line -- so an over-long pair loses the very value that
      distinguishes chosen from rejected, which can make the pair identical and
      the preference gradient degenerate.

    Neither is detectable from the loss curve, so it is measured here.
    """
    dataset = trainer.train_dataset
    n = min(TRUNCATION_SAMPLE, len(dataset))
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    prompt_capped = 0
    completion_capped = 0
    degenerate = 0
    for i in range(n):
        row = dataset[i]
        p = len(row.get("prompt_input_ids", []) or [])
        c = len(row.get("chosen_input_ids", []) or [])
        r = len(row.get("rejected_input_ids", []) or [])
        prompt_lengths.append(p)
        completion_lengths.append(max(c, r))
        if max_prompt_length and p >= max_prompt_length:
            prompt_capped += 1
        if max_completion_length and max(c, r) >= max_completion_length:
            completion_capped += 1
        if c and list(row.get("chosen_input_ids", [])) == list(
            row.get("rejected_input_ids", [])
        ):
            degenerate += 1
    return {
        "sampled": n,
        "prompt_p99": percentile(prompt_lengths, 0.99),
        "prompt_max": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_at_cap": prompt_capped,
        "completion_p99": percentile(completion_lengths, 0.99),
        "completion_max": max(completion_lengths) if completion_lengths else 0,
        "completion_at_cap": completion_capped,
        "identical_pairs": degenerate,
    }


def print_truncation_scan(
    scan: dict, max_prompt_length: int, max_completion_length: int
) -> None:
    print(
        f"truncation scan ({scan['sampled']} rows): "
        f"prompt p99={scan['prompt_p99']} max={scan['prompt_max']} "
        f"cap={max_prompt_length} hit={scan['prompt_at_cap']} | "
        f"completion p99={scan['completion_p99']} max={scan['completion_max']} "
        f"cap={max_completion_length} hit={scan['completion_at_cap']}"
    )
    n = scan["sampled"] or 1
    if scan["prompt_at_cap"] / n > TRUNCATION_WARN_RATE:
        logger.warning(
            "%d/%d prompts hit max_prompt_length=%d. The prompt is truncated from "
            "the LEFT, so those rows lost the start of the identity block and train "
            "against a prompt that does not match serving. Raise max_prompt_length.",
            scan["prompt_at_cap"],
            n,
            max_prompt_length,
        )
    if scan["completion_at_cap"] / n > TRUNCATION_WARN_RATE:
        logger.warning(
            "%d/%d completions hit max_completion_length=%d. Completions are "
            "truncated from the RIGHT, so those rows lost their FINAL ANSWER line "
            "-- the one place chosen and rejected differ. Raise "
            "max_completion_length (and max_length with it).",
            scan["completion_at_cap"],
            n,
            max_completion_length,
        )
    if scan["identical_pairs"]:
        logger.warning(
            "%d/%d sampled pairs have identical chosen/rejected token sequences "
            "after truncation: those contribute no preference gradient.",
            scan["identical_pairs"],
            n,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sft-adapter",
        default=None,
        help="path to the stage-1 adapter, e.g. runs/sft/adapter (required)",
    )
    parser.add_argument("--run-name", default=None, help="run directory under runs/")
    parser.add_argument("--train-file", default=None, help="override dpo.train_file")
    parser.add_argument("--val-file", default=None, help="override dpo.val_file")
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


def install_chat_template(tokenizer: Any, cfg: dict) -> None:
    """Install the harness chat template, refusing to train without it.

    The stock Phi-4-mini-reasoning template hardcodes "Your name is Phi, an AI
    math expert developed by Microsoft." ahead of every system message, which
    contradicts the identity stage 1 trained. The tokenizer saved next to the DPO
    adapter carries this template so serving matches training.
    """
    if not chat.apply_chat_template_override(tokenizer, cfg):
        raise SystemExit(
            "chat.template_path is not set, so the tokenizer would keep the vendor "
            "chat template, which prepends the Microsoft identity preamble to every "
            "system message and contradicts the identity being trained.\n"
            "Set chat.template_path (configs/base.yaml ships "
            "configs/chat_template.jinja) before training."
        )


def load_preference_split(path: str, label: str) -> Any:
    """Load a prepared preference JSONL and reduce it to prompt/chosen/rejected."""
    resolved = config_mod.harness_path(path)
    if not resolved.is_file():
        raise SystemExit(
            f"missing {label} file: {resolved}\n"
            "Run scripts/01_prepare_data.py first, or point at another file with "
            f"--{label.replace('_', '-')} / --set dpo.{label}=<path>."
        )
    dataset = load_dataset("json", data_files=str(resolved), split="train")
    missing = [c for c in PREFERENCE_COLUMNS if c not in dataset.column_names]
    if missing:
        raise SystemExit(
            f"{resolved} is missing the column(s) {missing} (found "
            f"{dataset.column_names}). Regenerate it with 01_prepare_data.py."
        )
    # prompt/chosen/rejected are already rendered strings ("standard" format), so
    # TRL applies no chat template on top of them. Dropping the descriptive
    # columns also matters because TRL's preference collator forwards whatever
    # columns survive.
    return dataset.select_columns(PREFERENCE_COLUMNS)


def resolve_adapter(args: argparse.Namespace, cfg: dict) -> Path:
    """Locate the stage-1 adapter, failing with an actionable message."""
    raw = args.sft_adapter or config_mod.get(cfg, "dpo.sft_adapter")
    if not raw:
        raise SystemExit(
            "DPO continues the adapter trained by scripts/03_train_sft.py, so "
            "--sft-adapter is required, e.g. --sft-adapter runs/sft/adapter"
        )
    path = config_mod.harness_path(raw)
    config_file = path / "adapter_config.json"
    if not config_file.is_file():
        raise SystemExit(
            f"{path} does not look like a PEFT adapter (no adapter_config.json).\n"
            "Point --sft-adapter at the 'adapter' directory of a finished SFT run."
        )

    # The adapter must have been trained on the base model we are about to load,
    # or stage 2 continues a delta that does not belong to these weights.
    base_id = config_mod.get(cfg, "model.base_id")
    try:
        trained_on = json.loads(config_file.read_text(encoding="utf-8")).get(
            "base_model_name_or_path"
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {config_file}: {exc}") from exc
    if trained_on and str(trained_on) != str(base_id):
        raise SystemExit(
            f"{path} was trained on base model {trained_on!r} but model.base_id is "
            f"{base_id!r}. Continuing a LoRA on weights it was not trained on "
            "produces a model that trains but reasons badly.\n"
            "Fix model.base_id, or point --sft-adapter at the matching run."
        )
    return path


def build_dpo_config(cfg: dict, output_dir: Path, logging_dir: Path, has_eval: bool):
    """Build the TRL 0.24.0 DPOConfig from the resolved ``dpo`` block."""

    def s(key: str, default: Any = None) -> Any:
        return config_mod.get(cfg, f"dpo.{key}", default)

    eval_strategy = s("eval_strategy", "steps") if has_eval else "no"

    # TRL 0.24.0 types loss_type as list[str]; a bare string would be iterated
    # character by character.
    loss_type = s("loss_type", ["sigmoid"])
    if isinstance(loss_type, str):
        loss_type = [loss_type]

    return DPOConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        report_to=report_targets(),
        seed=int(s("seed", config_mod.get(cfg, "seed", 3407))),
        beta=float(s("beta", 0.1)),
        loss_type=list(loss_type),
        max_length=int(s("max_length", 2048)),
        max_prompt_length=int(s("max_prompt_length", 1408)),
        max_completion_length=int(s("max_completion_length", 640)),
        # keep_end slices prompt[-max_prompt_length:], dropping the identity
        # prefix; max_prompt_length is sized so this should not trigger.
        truncation_mode=s("truncation_mode", "keep_end"),
        disable_dropout=bool(s("disable_dropout", True)),
        dataset_num_proc=s("dataset_num_proc", 4),
        per_device_train_batch_size=int(s("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(s("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(s("gradient_accumulation_steps", 8)),
        num_train_epochs=float(s("num_train_epochs", 1)),
        max_steps=int(s("max_steps", -1)),
        learning_rate=float(s("learning_rate", 5.0e-6)),
        lr_scheduler_type=s("lr_scheduler_type", "cosine"),
        warmup_ratio=float(s("warmup_ratio", 0.1)),
        # Deliberately 0.0 here (SFT/ORPO use 0.01): beta already regularises how
        # far the policy may move from the reference. See configs/dpo.yaml.
        weight_decay=float(s("weight_decay", 0.0)),
        optim=s("optim", "adamw_torch_fused"),
        max_grad_norm=float(s("max_grad_norm", 1.0)),
        bf16=bool(s("bf16", True)),
        fp16=bool(s("fp16", False)),
        # Unsloth installed its own checkpointing when the adapter was created.
        gradient_checkpointing=bool(s("gradient_checkpointing", False)),
        logging_steps=int(s("logging_steps", 10)),
        eval_strategy=eval_strategy,
        eval_steps=int(s("eval_steps", 250)),
        save_strategy=s("save_strategy", "steps"),
        save_steps=int(s("save_steps", 250)),
        save_total_limit=int(s("save_total_limit", 3)),
        dataloader_num_workers=int(s("dataloader_num_workers", 4)),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    cfg = config_mod.load_config(stage=STAGE, extra=args.config, overrides=args.set)

    run_name = args.run_name or config_mod.get(cfg, "dpo.run_name", "dpo")
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

    adapter_path = resolve_adapter(args, cfg)
    seed = int(config_mod.get(cfg, "dpo.seed", config_mod.get(cfg, "seed", 3407)))
    set_seed(seed)

    # --- data -------------------------------------------------------------
    train_path = args.train_file or config_mod.get(cfg, "dpo.train_file")
    val_path = args.val_file or config_mod.get(cfg, "dpo.val_file")
    train_dataset = load_preference_split(train_path, "train_file")
    eval_dataset = None
    if val_path and config_mod.harness_path(val_path).is_file():
        eval_dataset = load_preference_split(val_path, "val_file")
    else:
        logger.warning("no validation file at %s; periodic eval disabled", val_path)

    # --- model ------------------------------------------------------------
    model, tokenizer = modeling.load_for_training(cfg)
    install_chat_template(tokenizer, cfg)

    # Attach the SFT adapter as the trainable policy. is_trainable=True is what
    # makes this a continuation of stage 1 rather than a frozen initialisation.
    # Because the result is a PeftModel and ref_model stays None, DPOTrainer uses
    # the adapter-disabled model as the implicit reference: it wraps the reference
    # forward passes in disable_adapter() (dpo_trainer.py:915-926), so there is no
    # second set of weights in memory. The reference is therefore the PRE-SFT BASE
    # policy, not the SFT policy -- see the module docstring for why that changes
    # how beta should be read.
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    logger.info("attached SFT adapter %s as trainable", adapter_path)

    # --- trainer ----------------------------------------------------------
    if args.dry_run:
        # A dry run must not create or disturb runs/<name>.
        scratch = Path(tempfile.mkdtemp(prefix="cosimo-dpo-dryrun-"))
        output_dir, logging_dir = scratch / "checkpoints", scratch / "tb"
    else:
        run.create("tb", "checkpoints", "adapter")
        output_dir, logging_dir = run.checkpoints_dir, run.tb_dir

    dpo_args = build_dpo_config(cfg, output_dir, logging_dir, eval_dataset is not None)
    trainer = DPOTrainer(
        model=model,
        # See the comment above: PEFT + ref_model=None => adapter-disabled base.
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # TRL 0.24.0: the tokenizer argument is `processing_class`.
        processing_class=tokenizer,
    )

    print(
        f"reference model: implicit (adapter disabled), ref_model={trainer.ref_model}"
    )
    model.print_trainable_parameters()
    scan = truncation_scan(
        trainer,
        int(config_mod.get(cfg, "dpo.max_prompt_length", 1408)),
        int(config_mod.get(cfg, "dpo.max_completion_length", 640)),
    )
    print_truncation_scan(
        scan,
        int(config_mod.get(cfg, "dpo.max_prompt_length", 1408)),
        int(config_mod.get(cfg, "dpo.max_completion_length", 640)),
    )
    print(f"train pairs: {len(trainer.train_dataset)}")
    if eval_dataset is not None:
        print(f"eval pairs: {len(trainer.eval_dataset)}")

    if args.dry_run:
        first = train_dataset[0]
        print("-" * 72)
        print("first preference pair:")
        print(f"prompt:\n{first['prompt']}")
        print(f"chosen:\n{first['chosen']}")
        print(f"rejected:\n{first['rejected']}")
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
            "chat_template_path": config_mod.get(cfg, "chat.template_path"),
            "sft_adapter": str(adapter_path),
            "adapter_dir": str(run.adapter_dir),
            "truncation_scan": scan,
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
