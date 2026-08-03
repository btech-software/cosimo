"""Model and tokenizer loading.

All heavy imports live inside functions so the pure-logic modules stay
importable on a CPU-only machine. ``unsloth`` must be imported before
``transformers``/``trl`` — every entry point here does that first.

Note on ``unsloth/Phi-4-mini-reasoning``: its projections are fused, so the
per-layer linear modules are ``qkv_proj``, ``o_proj``, ``gate_up_proj``,
``down_proj``. The usual seven-module LoRA target list matches nothing on this
model, which is why ``resolve_target_modules`` reads the live module tree.
"""

from __future__ import annotations

import logging
from typing import Any

from . import config as config_mod

logger = logging.getLogger(__name__)

# Every projection name we are willing to adapt, fused or unfused.
KNOWN_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
)


def torch_dtype(name: str):
    """Map a dtype name from the config onto a torch dtype."""
    import torch

    dtype = getattr(torch, str(name), None)
    if dtype is None:
        raise ValueError(f"unknown dtype {name!r}")
    return dtype


def resolve_target_modules(model: Any, requested: str | list[str]) -> list[str]:
    """Return the LoRA target module basenames present on ``model``.

    ``requested == "auto"`` walks the live module tree; an explicit list is
    validated against it so a typo fails before training starts.
    """
    import torch.nn as nn

    present = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or module.__class__.__name__.endswith(
            "Linear"
        ):
            basename = name.rsplit(".", 1)[-1]
            if basename in KNOWN_PROJECTIONS:
                present.add(basename)

    if isinstance(requested, str) and requested.strip().lower() == "auto":
        resolved = sorted(present)
    else:
        requested_list = [requested] if isinstance(requested, str) else list(requested)
        resolved = sorted({name for name in requested_list if name in present})
        missing = sorted(set(requested_list) - present)
        if missing:
            logger.warning(
                "requested LoRA targets not found on the model and dropped: %s", missing
            )
    if not resolved:
        raise ValueError(
            "no LoRA target modules resolved. The model exposes these "
            "projection names: "
            f"{sorted(present) or 'none'} (requested: {requested!r}). For "
            "unsloth/Phi-4-mini-reasoning the correct targets are "
            "['down_proj', 'gate_up_proj', 'o_proj', 'qkv_proj']."
        )
    logger.info("resolved LoRA target modules: %s", resolved)
    return resolved


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is not None:
        return
    for candidate in ("<|endofprompt|>", "<|finetune_right_pad_id|>"):
        if candidate in (tokenizer.get_vocab() or {}):
            tokenizer.pad_token = candidate
            logger.info("tokenizer had no pad token; set pad_token=%s", candidate)
            return
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(
        "tokenizer had no pad token; set pad_token=eos_token=%s", tokenizer.eos_token
    )


def load_for_training(cfg: dict) -> tuple[Any, Any]:
    """Load the base model for training (right padding)."""
    import unsloth  # noqa: F401  # must precede transformers/trl
    from unsloth import FastLanguageModel

    base_id = config_mod.get(cfg, "model.base_id")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_id,
        max_seq_length=config_mod.get(cfg, "model.max_seq_length", 2048),
        dtype=torch_dtype(config_mod.get(cfg, "model.dtype", "bfloat16")),
        load_in_4bit=bool(config_mod.get(cfg, "model.load_in_4bit", False)),
        revision=config_mod.get(cfg, "model.revision"),
    )
    _ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    logger.info("loaded %s for training", base_id)
    return model, tokenizer


def load_for_inference(
    base_id: str,
    adapter_path: str | None = None,
    merged_path: str | None = None,
    *,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    dtype: str = "bfloat16",
    revision: str | None = None,
) -> tuple[Any, Any]:
    """Load a model for generation (left padding).

    ``merged_path`` is loaded directly as the model id; ``adapter_path`` is
    attached to ``base_id`` with PEFT. Passing both is a configuration error.
    ``revision`` pins the hub weights and is ignored for a local ``merged_path``.
    """
    import unsloth  # noqa: F401  # must precede transformers/trl
    from unsloth import FastLanguageModel

    if adapter_path and merged_path:
        raise ValueError("pass either adapter_path or merged_path, not both")

    model_id = merged_path or base_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=torch_dtype(dtype),
        load_in_4bit=bool(load_in_4bit),
        revision=None if merged_path else revision,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("attached adapter %s", adapter_path)

    try:
        FastLanguageModel.for_inference(model)
    except AttributeError:
        model.eval()
    _ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"
    logger.info("loaded %s for inference", model_id)
    return model, tokenizer


def attach_lora(model: Any, cfg: dict) -> Any:
    """Wrap the model in a LoRA adapter using the ``lora`` config block."""
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    targets = resolve_target_modules(
        model, config_mod.get(cfg, "lora.target_modules", "auto")
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config_mod.get(cfg, "lora.r", 32),
        lora_alpha=config_mod.get(cfg, "lora.lora_alpha", 32),
        lora_dropout=config_mod.get(cfg, "lora.lora_dropout", 0.0),
        bias=config_mod.get(cfg, "lora.bias", "none"),
        target_modules=targets,
        use_rslora=bool(config_mod.get(cfg, "lora.use_rslora", False)),
        use_gradient_checkpointing=config_mod.get(
            cfg, "lora.use_gradient_checkpointing", "unsloth"
        ),
        random_state=config_mod.get(cfg, "seed", 3407),
    )
    return model


def model_fingerprint(
    cfg: dict,
    *,
    base_id: str,
    adapter_path: str | None = None,
    merged_path: str | None = None,
) -> dict:
    """What identifies the evaluated artifact, as embedded in ``metrics.json``."""
    return {
        "base_id": base_id,
        "adapter_path": adapter_path,
        "merged_path": merged_path,
        "load_in_4bit": bool(config_mod.get(cfg, "model.load_in_4bit", False)),
        "dtype": config_mod.get(cfg, "model.dtype", "bfloat16"),
        "revision": config_mod.get(cfg, "model.revision"),
    }
