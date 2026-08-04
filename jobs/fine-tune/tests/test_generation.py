"""Batch planning: the bound that keeps a large generation budget from taking
the machine down.

`batch_size` alone does not bound memory. Every sequence reserves
`prompt + max_new_tokens` of KV cache, so raising max_new_tokens at a fixed
count multiplies the reservation. Measured on the DGX Spark: 16 x (575 + 768) =
21 488 slots completed; 16 x (575 + 4096) = 74 736 slots exhausted 121 GB of
unified memory and was OOM-killed along with the desktop session.
"""

from __future__ import annotations

import pytest

from cosimo_ft import generation

# The real corpus, from split_manifest.json: prompt p50 522, p95 552, max 575.
PROMPT = 575


def footprints(batches, lengths, max_new_tokens):
    return [
        len(b) * (max(lengths[i] for i in b) + max_new_tokens) for b in batches
    ]


def sorted_order(lengths):
    return sorted(range(len(lengths)), key=lambda i: (lengths[i], i))


def test_every_prompt_appears_exactly_once():
    lengths = [PROMPT] * 37
    batches = generation.plan_batches(
        lengths,
        sorted_order(lengths),
        batch_size=16,
        max_new_tokens=2048,
        max_batch_tokens=24576,
    )
    assert sorted(i for b in batches for i in b) == list(range(37))


def test_budget_shrinks_the_batch_as_the_token_budget_grows():
    """The whole point: a bigger generation budget costs wall clock, not stability."""
    lengths = [PROMPT] * 64
    order = sorted_order(lengths)
    sizes = {}
    for max_new_tokens in (768, 2048, 4096):
        batches = generation.plan_batches(
            lengths,
            order,
            batch_size=16,
            max_new_tokens=max_new_tokens,
            max_batch_tokens=24576,
        )
        sizes[max_new_tokens] = max(len(b) for b in batches)
    assert sizes[768] > sizes[2048] > sizes[4096]


def test_the_configuration_that_died_is_now_bounded():
    """16 x (575 + 4096) = 74 736 slots is what OOM-killed the machine."""
    lengths = [PROMPT] * 16
    batches = generation.plan_batches(
        lengths,
        sorted_order(lengths),
        batch_size=16,
        max_new_tokens=4096,
        max_batch_tokens=24576,
    )
    assert max(footprints(batches, lengths, 4096)) <= 24576
    assert len(batches) > 1, "it must split rather than emit one 74 736-slot batch"


def test_the_configuration_that_completed_is_not_penalised():
    """16 x (575 + 768) = 21 488 ran fine; the cap must not shrink it."""
    lengths = [PROMPT] * 16
    batches = generation.plan_batches(
        lengths,
        sorted_order(lengths),
        batch_size=16,
        max_new_tokens=768,
        max_batch_tokens=24576,
    )
    assert len(batches) == 1
    assert len(batches[0]) == 16


def test_batch_size_still_caps_the_count():
    """With a generous budget, the count limit is what binds."""
    lengths = [10] * 40
    batches = generation.plan_batches(
        lengths,
        sorted_order(lengths),
        batch_size=8,
        max_new_tokens=16,
        max_batch_tokens=10**9,
    )
    assert all(len(b) <= 8 for b in batches)
    assert max(len(b) for b in batches) == 8


def test_none_budget_restores_count_based_batching():
    lengths = [PROMPT] * 32
    batches = generation.plan_batches(
        lengths,
        sorted_order(lengths),
        batch_size=16,
        max_new_tokens=4096,
        max_batch_tokens=None,
    )
    assert [len(b) for b in batches] == [16, 16]


def test_an_oversized_single_prompt_is_emitted_alone_not_dropped():
    """Refusing it would silently skip an eval item; the caller asked for it."""
    lengths = [50, 100_000, 50]
    batches = generation.plan_batches(
        lengths,
        sorted_order(lengths),
        batch_size=16,
        max_new_tokens=2048,
        max_batch_tokens=24576,
    )
    assert sorted(i for b in batches for i in b) == [0, 1, 2]
    assert [1] in batches


def test_longest_prompt_in_the_batch_drives_the_cost():
    """Cost is count x LONGEST prompt, not count x mean; a mixed batch must not
    be sized off its short members."""
    lengths = [100, 100, 100, 4000]
    batches = generation.plan_batches(
        lengths,
        sorted_order(lengths),
        batch_size=16,
        max_new_tokens=2048,
        max_batch_tokens=12288,
    )
    assert all(f <= 12288 for f in footprints(batches, lengths, 2048))


def test_zero_batch_size_is_rejected():
    with pytest.raises(ValueError):
        generation.plan_batches(
            [10],
            [0],
            batch_size=0,
            max_new_tokens=16,
            max_batch_tokens=1024,
        )


def test_empty_input_produces_no_batches():
    assert (
        generation.plan_batches(
            [], [], batch_size=16, max_new_tokens=2048, max_batch_tokens=24576
        )
        == []
    )
