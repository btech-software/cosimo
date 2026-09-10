"""Model and tokenizer loading.

All heavy imports live inside functions so the pure-logic modules stay
importable on a CPU-only machine. ``unsloth`` must be imported before
``transformers``/``trl`` — every entry point here does that first.

Why ``resolve_target_modules`` reads the live module tree rather than trusting a
hardcoded list: projection naming is model-specific. Qwen3.8-27B exposes the
usual seven (``q_proj``/``k_proj``/``v_proj``/``o_proj`` plus
``gate_proj``/``up_proj``/``down_proj``) on its 16 attention layers and
``in_proj_*``/``out_proj`` on its 48 linear-attention layers; models with fused
projections expose ``qkv_proj``/``gate_up_proj`` instead, where that list matches
nothing at all. A wrong list resolves to zero targets and trains no adapter.
"""

from __future__ import annotations

import logging
from typing import Any

from . import chat
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
            f"{sorted(present) or 'none'} (requested: {requested!r}). Set "
            "lora.target_modules to names from that list, or leave it 'auto' to "
            "resolve them from the live module tree."
        )
    logger.info("resolved LoRA target modules: %s", resolved)
    return resolved


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is not None:
        return
    # Through the text tokenizer: a VLM processor has no get_vocab (chat
    # .text_tokenizer). Only this probe needs unwrapping -- pad_token and
    # eos_token read and write on the object unsloth handed back, which is what
    # the trainer and the collator are given.
    vocab = chat.text_tokenizer(tokenizer).get_vocab() or {}
    for candidate in ("<|endofprompt|>", "<|finetune_right_pad_id|>"):
        if candidate in vocab:
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
        use_exact_model_name=bool(
            config_mod.get(cfg, "model.use_exact_model_name", False)
        ),
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
    use_exact_model_name: bool = False,
) -> tuple[Any, Any]:
    """Load a model for generation (left padding).

    ``merged_path`` is loaded directly as the model id; ``adapter_path`` is
    attached to ``base_id`` with PEFT. Passing both is a configuration error.
    ``revision`` pins the hub weights and is ignored for a local ``merged_path``.

    ``use_exact_model_name`` must match what training used. Unsloth silently
    rewrites a model id to its own pre-quantized mirror when ``load_in_4bit`` is
    set, so training and inference can end up on *different base weights* -- and
    an adapter fitted to one does not belong on the other.
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
        use_exact_model_name=bool(use_exact_model_name),
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
