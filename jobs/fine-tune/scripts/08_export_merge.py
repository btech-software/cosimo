#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the base weights and export bf16 weights.

The merged model is a standalone checkpoint: it needs no PEFT at load time and
can be passed to scripts/06_evaluate.py as --merged, served, or converted
further. Merging is always done in bf16; merging into 4-bit weights would
quantize the adapter's contribution away.

Example:
    ./scripts/08_export_merge.py --run-name dpo
    ./scripts/08_export_merge.py --adapter runs/sft/adapter --out runs/sft/merged
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from cosimo_ft import chat  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402
from cosimo_ft import modeling  # noqa: E402
from cosimo_ft.runlog import RunDir, utc_now, write_json  # noqa: E402

logger = logging.getLogger("export_merge")

# The sentence the vendor template injects ahead of every system message. A
# merged checkpoint containing it would reintroduce the Microsoft identity at
# serving time, undoing the persona the run was trained to carry.
VENDOR_MARKER = "Microsoft"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="run whose adapter is merged; defaults --adapter to runs/<name>/adapter "
        "and --out to runs/<name>/merged",
    )
    parser.add_argument("--adapter", default=None, help="path to the PEFT adapter")
    parser.add_argument("--out", default=None, help="output directory for bf16 weights")
    parser.add_argument("--base-id", default=None, help="override model.base_id")
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="also export GGUF (optional and slow: it builds llama.cpp and requantizes "
        "the whole model; failures are reported, not fatal)",
    )
    parser.add_argument(
        "--gguf-quant",
        default="q4_k_m",
        help="GGUF quantization method passed to unsloth (default: q4_k_m)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output directory"
    )
    config_mod.add_config_args(parser)
    return parser


def resolve_paths(args: argparse.Namespace, cfg: dict) -> tuple[Path, Path]:
    """Work out the adapter and output directories from --run-name / --adapter."""
    runs_dir = config_mod.get(cfg, "paths.runs_dir", "runs")
    adapter = args.adapter
    out = args.out
    if args.run_name:
        run = RunDir(runs_dir, args.run_name)
        adapter = adapter or str(run.adapter_dir)
        out = out or str(run.merged_dir)
    if not adapter:
        raise SystemExit("pass --run-name or --adapter to say what should be merged")
    if not out:
        raise SystemExit("pass --run-name or --out to say where the merge should go")

    adapter_path = config_mod.harness_path(adapter)
    if not (adapter_path / "adapter_config.json").is_file():
        raise SystemExit(
            f"{adapter_path} does not look like a PEFT adapter (no "
            "adapter_config.json). Point --adapter at the 'adapter' directory of a "
            "finished training run."
        )
    return adapter_path, config_mod.harness_path(out)


def check_adapter_base(adapter_path: Path, base_id: str) -> None:
    """Fail when the adapter was trained against a different base checkpoint.

    Merging a LoRA into weights it was not trained on produces a model that
    loads cleanly and reasons badly, which is far harder to diagnose later.
    """
    config_file = adapter_path / "adapter_config.json"
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {config_file}: {exc}") from exc
    trained_on = data.get("base_model_name_or_path")
    if trained_on and str(trained_on) != str(base_id):
        raise SystemExit(
            f"{adapter_path} was trained on base model {trained_on!r} but this run "
            f"resolves model.base_id to {base_id!r}. Merging a LoRA into weights it "
            "was not trained on yields a model that loads but reasons badly.\n"
            "Pass --base-id to match, or point at the right adapter."
        )


def saved_chat_template(directory: Path) -> str | None:
    """Read back the chat template a saved tokenizer carries, if any.

    transformers writes it either to ``chat_template.jinja`` (current default for
    a single template) or inline in ``tokenizer_config.json``; check both.
    """
    jinja = directory / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    config_file = directory / "tokenizer_config.json"
    if config_file.is_file():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        template = data.get("chat_template")
        if isinstance(template, str):
            return template
    return None


def verify_saved_template(
    directory: Path, expected: str, label: str, source: str
) -> None:
    """Fail the export unless ``directory`` carries the harness chat template.

    A checkpoint served with the vendor template would prepend the Microsoft
    identity preamble to every system message, silently discarding the persona
    the run was trained to carry. Catching that here is the whole point of the
    export step verifying its own output.
    """
    found = saved_chat_template(directory)
    if not found:
        raise SystemExit(
            f"{label} at {directory} has no chat template. It would fall back to "
            "the vendor default at serving time. Re-run the export; if this "
            "persists, the tokenizer save path changed."
        )
    if VENDOR_MARKER in found:
        raise SystemExit(
            f"{label} at {directory} carries a chat template mentioning "
            f"{VENDOR_MARKER!r}: the vendor identity preamble survived the export "
            "and would override the trained persona at serving time."
        )
    if found.strip() != expected.strip():
        raise SystemExit(
            f"{label} at {directory} carries a chat template that differs from "
            f"{source}. Serving would not match training.\n"
            f"expected: {expected.strip()[:200]!r}\n"
            f"found:    {found.strip()[:200]!r}"
        )
    logger.info("%s carries the harness chat template", label)


def saved_state_dict_keys(directory: Path) -> list[str]:
    """List the tensor names in a saved checkpoint without loading the weights.

    Reads the shard index when the checkpoint is sharded, otherwise parses the
    safetensors header directly (8-byte little-endian length, then JSON).
    """
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / index_name
        if index.is_file():
            data = json.loads(index.read_text(encoding="utf-8"))
            return sorted(data.get("weight_map", {}))

    single = directory / "model.safetensors"
    if single.is_file():
        with single.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(length).decode("utf-8"))
        return sorted(k for k in header if k != "__metadata__")

    raise SystemExit(
        f"{directory} contains no recognisable weight file "
        "(model.safetensors[.index.json] / pytorch_model.bin.index.json). "
        "The merge did not write weights."
    )


def verify_merged_weights(directory: Path) -> int:
    """Fail unless the export is a genuinely merged, loadable checkpoint.

    Two failure modes this catches, both of which otherwise exit 0:
    * adapter keys (``lora_``/``base_layer``) in the state dict, meaning the LoRA
      was serialised rather than merged and every projection would be
      re-initialised at load time (transformers only warns about unexpected keys);
    * a config that transformers cannot resolve to an architecture.
    """
    keys = saved_state_dict_keys(directory)
    if not keys:
        raise SystemExit(f"{directory} saved an empty state dict.")

    bad = [k for k in keys if "lora_" in k or "base_layer" in k]
    if bad:
        raise SystemExit(
            f"{directory} still contains {len(bad)} adapter tensors, so the LoRA "
            "was NOT merged into the base weights. Loading this checkpoint would "
            "silently re-initialise every adapted projection.\n"
            f"examples: {bad[:5]}"
        )

    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(str(directory))
    except Exception as exc:
        raise SystemExit(
            f"{directory} has a config transformers cannot load ({exc}). The "
            "checkpoint would not be loadable by 06_evaluate.py --merged."
        ) from exc
    if not getattr(config, "architectures", None):
        raise SystemExit(
            f"{directory}/config.json declares no architectures; the checkpoint "
            "is not loadable."
        )
    logger.info(
        "merged checkpoint verified: %d tensors, architecture %s",
        len(keys),
        config.architectures,
    )
    return len(keys)


def gguf_from_merged(
    merged_dir: Path, gguf_dir: Path, quant: str, max_seq_length: int
) -> None:
    """Convert the merged checkpoint to GGUF via Unsloth's saver.

    Loads the merged directory as a plain model first, so the conversion sees
    already-merged weights and never touches the PeftModel wrapper.

    ``max_seq_length`` tracks model.max_seq_length rather than being hardcoded:
    exporting at a shorter length than the model was trained at would ship a
    GGUF that cannot hold the agentic conversations it was tuned for.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=False,
    )
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    cfg = config_mod.load_config(extra=args.config, overrides=args.set)

    adapter_path, out_path = resolve_paths(args, cfg)
    if out_path.exists() and any(out_path.iterdir()) and not args.force:
        raise SystemExit(f"{out_path} already exists and is not empty; pass --force")

    base_id = args.base_id or config_mod.get(cfg, "model.base_id")
    check_adapter_base(adapter_path, base_id)
    dtype = config_mod.get(cfg, "model.dtype", "bfloat16")
    if config_mod.get(cfg, "model.load_in_4bit", False):
        logger.warning(
            "model.load_in_4bit is true in the config but merging always loads the "
            "base weights in %s; a 4-bit merge would discard adapter precision",
            dtype,
        )

    model, tokenizer = modeling.load_for_inference(
        base_id,
        adapter_path=str(adapter_path),
        max_seq_length=int(config_mod.get(cfg, "model.max_seq_length", 2048)),
        load_in_4bit=False,
        dtype=dtype,
    )

    # The tokenizer comes from the base model, so it arrives with the vendor
    # template. Override it before saving: the merged checkpoint is what gets
    # served, and it must prompt exactly the way the adapter was trained.
    template_source = config_mod.get(cfg, "chat.template_path")
    if not chat.apply_chat_template_override(tokenizer, cfg):
        raise SystemExit(
            "chat.template_path is not set, so the exported checkpoint would carry "
            "the vendor chat template and reintroduce the Microsoft identity "
            "preamble at serving time.\n"
            "Set chat.template_path (configs/base.yaml ships "
            "configs/chat_template.jinja) before exporting."
        )
    expected_template = tokenizer.chat_template

    # The adapter directory was written by the training script, which applies the
    # same override; a mismatch means it predates that and should be re-exported.
    if saved_chat_template(adapter_path) is None:
        logger.warning(
            "%s carries no chat template. It was probably written before the "
            "template override existed; the merged export below is still correct, "
            "but anything loading the adapter directly will use the vendor default.",
            adapter_path,
        )
    else:
        verify_saved_template(
            adapter_path, expected_template, "adapter directory", template_source
        )

    out_path.mkdir(parents=True, exist_ok=True)

    # Merge via PEFT explicitly. Do NOT dispatch on hasattr(model,
    # "save_pretrained_merged"): Unsloth binds that method to the *base model
    # instance* (unsloth/save.py:5754), so on a PeftModel wrapper the attribute
    # resolves through PEFT's __getattr__ delegation and `self` inside the call
    # is the base model, not the wrapper. unsloth_generic_save then evaluates
    # isinstance(model, PeftModel) as False (unsloth/save.py:4183) and takes its
    # "full fine-tuned model" branch, writing the base model's state_dict --
    # which, because PEFT swapped the projections in place, carries
    # `...qkv_proj.base_layer.weight` / `...lora_A.default.weight` keys. The
    # adapter is never merged and reloading silently re-initialises every
    # projection. merge_and_unload() folds the deltas into the base weights and
    # strips the LoRA modules, so the saved keys are the plain architecture's.
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        raise SystemExit(
            f"expected a PeftModel after attaching {adapter_path}, got "
            f"{type(model).__name__}. Refusing to export: without the adapter "
            "attached there is nothing to merge."
        )
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))

    # Read the artifact back off disk rather than trusting the save calls. The
    # weights are the point of this script, so the self-check covers them too.
    verify_saved_template(
        out_path, expected_template, "merged checkpoint", template_source
    )
    n_keys = verify_merged_weights(out_path)
    print(f"merged {dtype} weights: {out_path} ({n_keys} tensors, no adapter keys)")

    gguf_path = None
    if args.gguf:
        gguf_path = out_path.parent / f"{out_path.name}_gguf"
        try:
            # Convert from the *merged* checkpoint, not from `model`: the same
            # instance-binding trap applies to save_pretrained_gguf, which would
            # otherwise serialise the unmerged base model.
            gguf_from_merged(
                out_path,
                gguf_path,
                args.gguf_quant,
                int(config_mod.get(cfg, "model.max_seq_length", 2048)),
            )
            print(f"gguf: {gguf_path}")
        except Exception as exc:  # llama.cpp build/conversion is failure-prone
            gguf_path = None
            logger.warning(
                "GGUF export failed (%s). The bf16 merge above is unaffected; "
                "convert it later with llama.cpp's convert_hf_to_gguf.py.",
                exc,
            )

    manifest = write_json(
        out_path / "export_manifest.json",
        {
            "created_at": utc_now(),
            "base_id": base_id,
            "adapter": str(adapter_path),
            "merged_dir": str(out_path),
            "dtype": dtype,
            "chat_template_path": template_source,
            "chat_template_verified": True,
            "gguf_dir": str(gguf_path) if gguf_path else None,
            "gguf_quant": args.gguf_quant if gguf_path else None,
            "config_hash": config_mod.config_hash(cfg),
        },
    )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
