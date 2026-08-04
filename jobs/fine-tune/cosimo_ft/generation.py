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


def plan_batches(
    lengths: list[int],
    order: list[int],
    *,
    batch_size: int,
    max_new_tokens: int,
    max_batch_tokens: int | None,
) -> list[list[int]]:
    """Group ``order`` into batches bounded by count **and** by token footprint.

    A fixed batch count is the wrong unit. Every sequence in a batch reserves
    ``prompt + max_new_tokens`` slots of KV cache, so the batch's real cost is
    ``len(batch) * (longest_prompt + max_new_tokens)`` -- which means raising
    ``max_new_tokens`` silently multiplies memory at a constant ``batch_size``.
    On the DGX Spark that is not a recoverable CUDA OOM: unified memory is
    shared with the host and the driver's allocations are unswappable, so the
    kernel OOM-killer takes the desktop session down with the job. Measured:
    16 x (575 + 768) = 21 488 slots completed; 16 x (575 + 4096) = 74 736 slots
    exhausted 121 GB and was killed.

    ``max_batch_tokens`` caps that product, so a larger generation budget costs
    wall clock (smaller batches) instead of stability. ``None`` disables the cap
    and restores pure count-based batching.

    ``order`` is expected sorted by ascending length, so the last index appended
    carries the batch's longest prompt.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in order:
        prompt_tokens = lengths[index]
        candidate_longest = max(longest, prompt_tokens)
        footprint = (len(current) + 1) * (candidate_longest + max_new_tokens)
        too_many = len(current) >= batch_size
        too_big = (
            max_batch_tokens is not None
            and current
            and footprint > max_batch_tokens
        )
        if current and (too_many or too_big):
            batches.append(current)
            current, longest = [], 0
            candidate_longest = prompt_tokens
        current.append(index)
        longest = candidate_longest
    if current:
        batches.append(current)

    # A single sequence over budget is emitted alone rather than dropped: the
    # caller asked for it, and refusing here would silently skip an eval item.
    if max_batch_tokens is not None:
        for batch in batches:
            cost = len(batch) * (max(lengths[i] for i in batch) + max_new_tokens)
            if cost > max_batch_tokens:
                logger.warning(
                    "a single prompt needs %d token slots, over the %d budget; "
                    "generating it alone. Lower max_new_tokens if this OOMs.",
                    cost,
                    max_batch_tokens,
                )
    return batches


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
    max_batch_tokens: int | None = None,
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

    batches = plan_batches(
        lengths,
        order,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_batch_tokens=max_batch_tokens,
    )
    widest = max(
        (len(b) * (max(lengths[i] for i in b) + max_new_tokens) for b in batches),
        default=0,
    )
    logger.info(
        "%d prompts in %d batches (max %d/batch, peak %d token slots, budget %s)",
        len(prompts),
        len(batches),
        max(len(b) for b in batches) if batches else 0,
        widest,
        max_batch_tokens if max_batch_tokens is not None else "unbounded",
    )
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
