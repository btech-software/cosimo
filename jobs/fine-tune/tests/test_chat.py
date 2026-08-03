"""Chat rendering, the two-block system prompt, and the template override.

Three properties are load-bearing for the whole harness and are pinned here:

* ``text == prompt + completion`` under the *shipped* chat template, which is
  what makes response-only masking valid;
* no rendered prompt carries the vendor "Microsoft" identity preamble;
* the identity/exam composition and its 15% training variation are deterministic.
"""

from __future__ import annotations

import copy
import hashlib

import pytest

from cosimo_ft import chat
from cosimo_ft import config as config_mod

from conftest import CHAT_TEMPLATE_PATH, EOS_TOKEN, FakeTokenizer

QUESTION = "A bond pays 5% annually. What is its current yield at a price of 98?"
COMPLETION = "Current yield = coupon / price.\n\nFINAL ANSWER: 5.10%"


# --------------------------------------------------------------------------
# compose_system — the two blocks
# --------------------------------------------------------------------------


def test_compose_system_is_identity_then_exam_protocol(cfg):
    identity = config_mod.get(cfg, "prompt.identity").strip()
    protocol = config_mod.get(cfg, "prompt.exam_protocol").strip()
    assert chat.compose_system(cfg) == f"{identity}\n\n{protocol}"


def test_the_exam_block_carries_the_grading_contract(cfg):
    tag = config_mod.get(cfg, "prompt.final_answer_tag")
    identity = config_mod.get(cfg, "prompt.identity")
    assert tag in chat.compose_system(cfg)
    assert tag not in identity, "the FINAL ANSWER contract belongs to the exam block"


def test_identity_is_present_without_the_exam_block(cfg):
    system = chat.compose_system(cfg, exam=False)
    assert system.startswith("You are Cosimo")
    assert config_mod.get(cfg, "prompt.final_answer_tag") not in system


def test_short_identity_still_carries_the_exam_block(cfg):
    short = chat.compose_system(cfg, short=True)
    assert short.startswith("You are Cosimo")
    assert config_mod.get(cfg, "prompt.final_answer_tag") in short
    assert len(short) < len(chat.compose_system(cfg))


def test_missing_identity_is_refused(cfg):
    broken = copy.deepcopy(cfg)
    broken["prompt"]["identity"] = "  "
    with pytest.raises(ValueError):
        chat.compose_system(broken)


# --------------------------------------------------------------------------
# system_for_record — deterministic training-time variation
# --------------------------------------------------------------------------


def test_system_for_record_is_deterministic(cfg):
    first = chat.system_for_record(cfg, "cfa_l1-000123")
    second = chat.system_for_record(cfg, "cfa_l1-000123")
    assert first == second


def test_system_for_record_returns_one_of_the_two_compositions(cfg):
    full = chat.compose_system(cfg)
    short = chat.compose_system(cfg, short=True)
    for index in range(200):
        assert chat.system_for_record(cfg, f"rec-{index:06d}") in (full, short)


def test_short_identity_rate_matches_the_configured_variation_rate(cfg):
    rate = float(config_mod.get(cfg, "prompt.variation_rate"))
    short = chat.compose_system(cfg, short=True)
    ids = [f"cfa_l1-{index:06d}" for index in range(20000)]
    observed = sum(1 for i in ids if chat.system_for_record(cfg, i) == short) / len(ids)
    assert observed == pytest.approx(rate, abs=0.01)


def test_variation_rate_zero_always_uses_the_full_identity(cfg):
    never = copy.deepcopy(cfg)
    never["prompt"]["variation_rate"] = 0.0
    full = chat.compose_system(never)
    assert all(chat.system_for_record(never, f"rec-{i}") == full for i in range(200))


def test_id_fraction_is_bounded():
    values = [chat.id_fraction(f"rec-{i}") for i in range(500)]
    assert all(0.0 <= v < 1.0 for v in values)


# --------------------------------------------------------------------------
# build_completion
# --------------------------------------------------------------------------


def test_build_completion_puts_the_tag_on_the_last_line():
    completion = chat.build_completion("Step 1.\nStep 2.\n", "42", "FINAL ANSWER:")
    assert completion == "Step 1.\nStep 2.\n\nFINAL ANSWER: 42"
    assert completion.splitlines()[-1] == "FINAL ANSWER: 42"


def test_build_completion_tolerates_missing_parts():
    assert chat.build_completion(None, None, "FINAL ANSWER:").strip() == "FINAL ANSWER:"


# --------------------------------------------------------------------------
# render_example — the prompt-prefix invariant
# --------------------------------------------------------------------------


def assert_prefix_invariant(tokenizer, system: str) -> dict:
    rendered = chat.render_example(tokenizer, QUESTION, COMPLETION, system)
    assert rendered["text"].startswith(rendered["prompt"])
    assert rendered["text"] == rendered["prompt"] + rendered["completion"]
    assert COMPLETION in rendered["completion"]
    assert QUESTION not in rendered["completion"], (
        "the question must stay in the prompt, or response-only masking would "
        "supervise the question"
    )
    return rendered


def test_prefix_invariant_with_the_fake_tokenizer(cfg, fake_tokenizer):
    assert_prefix_invariant(fake_tokenizer, chat.compose_system(cfg))


def test_prefix_invariant_under_the_shipped_template(cfg, jinja_tokenizer):
    rendered = assert_prefix_invariant(jinja_tokenizer, chat.compose_system(cfg))
    assert rendered["prompt"].endswith("<|assistant|>")
    assert rendered["text"].endswith(EOS_TOKEN)


def test_shipped_template_matches_the_fake_tokenizer(cfg, jinja_tokenizer):
    system = chat.compose_system(cfg)
    real = chat.render_example(jinja_tokenizer, QUESTION, COMPLETION, system)
    fake = chat.render_example(FakeTokenizer(), QUESTION, COMPLETION, system)
    assert real == fake, "the fake tokenizer no longer models the shipped template"


def test_render_prompt_ends_with_the_generation_marker(cfg, jinja_tokenizer):
    prompt = chat.render_prompt(jinja_tokenizer, QUESTION, chat.compose_system(cfg))
    assert prompt.endswith(config_mod.get(cfg, "chat.response_part"))
    assert config_mod.get(cfg, "chat.instruction_part") in prompt


def test_template_drift_is_refused(cfg):
    class DriftingTokenizer(FakeTokenizer):
        """A template whose full rendering is not prefixed by its prompt."""

        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            text = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
            return text if kwargs.get("add_generation_prompt") else "PREAMBLE" + text

    with pytest.raises(ValueError, match="chat template drift"):
        chat.render_example(
            DriftingTokenizer(), QUESTION, COMPLETION, chat.compose_system(cfg)
        )


# --------------------------------------------------------------------------
# the vendor identity preamble must not survive anywhere
# --------------------------------------------------------------------------


def test_the_shipped_template_file_has_no_vendor_preamble(chat_template_text):
    assert "Microsoft" not in chat_template_text
    assert "Your name is Phi" not in chat_template_text


@pytest.mark.parametrize("short", [False, True])
def test_no_rendered_training_prompt_contains_microsoft(cfg, jinja_tokenizer, short):
    system = chat.compose_system(cfg, short=short)
    rendered = chat.render_example(jinja_tokenizer, QUESTION, COMPLETION, system)
    for part in ("prompt", "completion", "text"):
        assert "Microsoft" not in rendered[part]


def test_the_vendor_template_would_have_injected_microsoft(cfg, vendor_tokenizer):
    # Negative control: without this, the assertion above could pass for the
    # wrong reason (e.g. a template that renders nothing at all).
    prompt = chat.render_prompt(vendor_tokenizer, QUESTION, chat.compose_system(cfg))
    assert "Microsoft" in prompt


# --------------------------------------------------------------------------
# template loading, hashing and installation
# --------------------------------------------------------------------------


def test_chat_template_hash_is_stable_and_content_addressed(cfg, chat_template_text):
    digest = chat.chat_template_hash(cfg)
    assert digest == chat.chat_template_hash(cfg)
    assert len(digest) == 12 and int(digest, 16) >= 0
    expected = hashlib.sha256(
        chat_template_text.strip("\n").encode("utf-8")
    ).hexdigest()[:12]
    assert digest == expected


def test_chat_template_hash_follows_the_content_not_the_path(cfg, tmp_path):
    copied = tmp_path / "copy.jinja"
    copied.write_text(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    same = copy.deepcopy(cfg)
    same["chat"]["template_path"] = str(copied)
    assert chat.chat_template_hash(same) == chat.chat_template_hash(cfg)

    changed = tmp_path / "changed.jinja"
    changed.write_text("{{ 'different' }}", encoding="utf-8")
    other = copy.deepcopy(cfg)
    other["chat"]["template_path"] = str(changed)
    assert chat.chat_template_hash(other) != chat.chat_template_hash(cfg)


def test_missing_template_file_is_an_error(cfg, tmp_path):
    broken = copy.deepcopy(cfg)
    broken["chat"]["template_path"] = str(tmp_path / "absent.jinja")
    with pytest.raises(FileNotFoundError):
        chat.load_chat_template(broken)


def test_null_template_path_falls_back_to_the_vendor_template(cfg, fake_tokenizer):
    fallback = copy.deepcopy(cfg)
    fallback["chat"]["template_path"] = None
    assert chat.load_chat_template(fallback) is None
    assert chat.chat_template_hash(fallback) is None
    assert chat.apply_chat_template_override(fake_tokenizer, fallback) is False
    assert fake_tokenizer.chat_template is None


def test_override_installs_the_harness_template(cfg, fake_tokenizer):
    assert chat.apply_chat_template_override(fake_tokenizer, cfg) is True
    assert fake_tokenizer.chat_template == chat.load_chat_template(cfg)
    assert "Microsoft" not in fake_tokenizer.chat_template
