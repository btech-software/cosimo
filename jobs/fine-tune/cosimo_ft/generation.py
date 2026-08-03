"""Batched, deterministic text generation.

Greedy by default, left padding, prompts bucketed by tokenized length for
throughput and restored to the caller's order before returning.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch so a rerun reproduces the same generations."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_stop_ids(model: Any, stop_token_ids: list[int] | None) -> list[int]:
    if stop_token_ids is not None:
        return [int(t) for t in stop_token_ids]
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(t) for t in eos]


def _progress_iter(items, enabled: bool, total: int):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, total=total, desc="generate")


def generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 3407,
    stop_token_ids: list[int] | None = None,
    progress: bool = True,
) -> list[dict]:
    """Generate a continuation for each prompt.

    Returns one ``{"text", "new_tokens"}`` per prompt, in the input order.
    ``new_tokens`` excludes padding and the stop token itself, so
    ``new_tokens >= max_new_tokens`` means the generation was truncated.
    """
    import torch

    if not prompts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    seed_everything(seed)
    stop_ids = set(_normalize_stop_ids(model, stop_token_ids))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)

    previous_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"

    # Bucket by tokenized length so each batch pads as little as possible, then
    # write results back into the caller's positions.
    lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i))
    results: list[dict | None] = [None] * len(prompts)

    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    try:
        for batch in _progress_iter(batches, progress, len(batches)):
            encoded = tokenizer(
                [prompts[i] for i in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": pad_id,
            }
            if stop_ids:
                kwargs["eos_token_id"] = sorted(stop_ids)
            if temperature > 0.0:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            with torch.inference_mode():
                outputs = model.generate(**encoded, **kwargs)

            prompt_length = encoded["input_ids"].shape[1]
            for position, index in enumerate(batch):
                generated = outputs[position][prompt_length:].tolist()
                new_tokens = len(generated)
                for offset, token in enumerate(generated):
                    if token in stop_ids or (pad_id is not None and token == pad_id):
                        new_tokens = offset
                        break
                text = tokenizer.decode(
                    generated[:new_tokens], skip_special_tokens=True
                )
                results[index] = {"text": text, "new_tokens": new_tokens}
    finally:
        if previous_padding_side is not None:
            tokenizer.padding_side = previous_padding_side

    missing = [i for i, row in enumerate(results) if row is None]
    if missing:  # defensive: the reorder must cover every prompt
        raise RuntimeError(
            f"generation produced no output for prompt indices {missing[:10]}"
        )
    return [row for row in results if row is not None]
