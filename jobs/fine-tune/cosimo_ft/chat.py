"""Chat rendering and system-prompt composition.

Rendering always goes through the tokenizer's chat template, so the harness
makes no assumption about marker strings and the base model and every fine-tuned
checkpoint are prompted identically. Only ``apply_chat_template`` is required of
the tokenizer object, which lets the unit tests pass a fake.

Two things are deliberate here:

* The vendor template is overridden (``configs/chat_template.jinja``). The stock
  Phi-4-mini-reasoning template hardcodes "Your name is Phi, an AI math expert
  developed by Microsoft." ahead of every system message, which contradicts the
  identity being trained. The override is structurally identical, and it is
  applied to the base model too, so comparability comes from both sides seeing
  the same prompt rather than from using the vendor default.
* The system message is two blocks: ``prompt.identity`` (the persona, on every
  example) and ``prompt.exam_protocol`` (only for exam-format items, and the
  carrier of the ``FINAL ANSWER:`` grading contract).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from . import config as config_mod

logger = logging.getLogger(__name__)

# 2**64, the denominator for turning a 16-hex-digit digest prefix into [0, 1).
_HASH_SPACE = 1 << 64


def compose_system(cfg: dict, *, short: bool = False, exam: bool = True) -> str:
    """Compose the system message from the identity and task blocks.

    ``short`` picks the one-line identity (training variation only); ``exam``
    appends the exam protocol that carries the ``FINAL ANSWER:`` contract.
    """
    key = "prompt.identity_short" if short else "prompt.identity"
    identity = str(config_mod.get(cfg, key, "")).strip()
    if not identity:
        raise ValueError(
            f"{key} is empty; the identity must be present on every example"
        )
    if not exam:
        return identity
    protocol = str(config_mod.get(cfg, "prompt.exam_protocol", "")).strip()
    return f"{identity}\n\n{protocol}" if protocol else identity


def id_fraction(record_id: str) -> float:
    """Map a record id deterministically into [0, 1) via sha256."""
    digest = hashlib.sha256(str(record_id).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / _HASH_SPACE


def system_for_record(cfg: dict, record_id: str) -> str:
    """The system message for one *training* example.

    A deterministic ``prompt.variation_rate`` fraction of ids gets the short
    identity, so the model does not become brittle to one exact ~600-token
    string. Evaluation never calls this: it always uses the full identity via
    ``compose_system(cfg)``.
    """
    rate = float(config_mod.get(cfg, "prompt.variation_rate", 0.0) or 0.0)
    return compose_system(cfg, short=id_fraction(record_id) < rate, exam=True)


def load_chat_template(cfg: dict) -> str | None:
    """Read ``chat.template_path``; None when the vendor template should stand."""
    path = config_mod.get(cfg, "chat.template_path")
    if not path:
        return None
    resolved = config_mod.harness_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"chat template not found: {resolved}")
    return resolved.read_text(encoding="utf-8").strip("\n")


def chat_template_hash(cfg: dict) -> str | None:
    """12-hex sha256 of the harness chat template, or None when unset.

    Recorded in ``split_manifest.json`` and in every ``metrics.json`` so two
    artifacts can be proven to have been produced with the same prompt surface.
    """
    template = load_chat_template(cfg)
    if template is None:
        return None
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


def apply_chat_template_override(tokenizer: Any, cfg: dict) -> bool:
    """Install the harness chat template on ``tokenizer``. Call before rendering.

    Returns True when the override was applied. Exported adapters and merged
    checkpoints must be saved with this template so serving matches training.
    """
    template = load_chat_template(cfg)
    if template is None:
        logger.warning(
            "chat.template_path is null: falling back to the tokenizer's own chat "
            "template. For unsloth/Phi-4-mini-reasoning that template hardcodes the "
            "vendor identity preamble ('Your name is Phi, an AI math expert developed "
            "by Microsoft.') ahead of every system message, which contradicts the "
            "Cosimo identity being trained."
        )
        return False
    tokenizer.chat_template = template
    logger.info(
        "applied chat template override from %s",
        config_mod.get(cfg, "chat.template_path"),
    )
    return True


def build_completion(reasoning_trace: str, answer: str, tag: str) -> str:
    """Assemble the supervised target: reasoning, blank line, final-answer line."""
    return f"{(reasoning_trace or '').rstrip()}\n\n{tag} {(answer or '').strip()}"


def build_messages(question: str, system: str) -> list[dict]:
    """Build the two-message prompt (system + user) fed to the chat template."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def render_prompt(tokenizer: Any, question: str, system: str) -> str:
    """Render the prompt string the model is asked to continue."""
    return tokenizer.apply_chat_template(
        build_messages(question, system),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_conversation(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render an arbitrary message list, optionally with bound tool schemas.

    ``tools`` is passed as the template's top-level ``tools`` variable, which is
    what transformers and therefore vLLM populate from an OpenAI request. It is
    omitted entirely when None so tokenizers whose ``apply_chat_template`` does
    not accept the keyword (the test fakes) still work.
    """
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def render_tool_example(
    tokenizer: Any, messages: list[dict], tools: list[dict] | None = None
) -> dict:
    """Render a multi-turn tool conversation as ``{"prompt", "completion", "text"}``.

    The split point is the *first* assistant turn: everything before it is the
    prompt, everything from it on is the supervised completion. Interior tool
    results inside the completion are masked at training time by
    ``train_on_responses_only``, which splits on the ``<|user|>`` / ``<|assistant|>``
    markers -- and the chat template renders tool results as ``<|user|>`` turns
    precisely so that this works without reconfiguring the masking.

    The same ``text == prompt + completion`` invariant as :func:`render_example`
    is enforced, for the same reason.
    """
    first_assistant = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"), None
    )
    if first_assistant is None:
        raise ValueError("a tool example needs at least one assistant turn")

    prompt = render_conversation(
        tokenizer, messages[:first_assistant], tools, add_generation_prompt=True
    )
    full = render_conversation(tokenizer, messages, tools, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError(
            "chat template drift: the rendered conversation does not start with "
            "the rendered prompt, so the completion cannot be isolated. "
            f"prompt={prompt[:200]!r} full={full[:200]!r}"
        )
    return {"prompt": prompt, "completion": full[len(prompt) :], "text": full}


def render_example(tokenizer: Any, question: str, completion: str, system: str) -> dict:
    """Render a full training example.

    Returns ``{"prompt", "completion", "text"}`` where ``text == prompt + completion``.
    The completion is *extracted* from the fully rendered conversation rather than
    guessed, so it carries whatever turn-end / EOS markers the template appends.
    """
    prompt = render_prompt(tokenizer, question, system)
    messages = build_messages(question, system) + [
        {"role": "assistant", "content": completion}
    ]
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        raise ValueError(
            "chat template drift: the rendered conversation does not start with "
            "the rendered prompt, so the completion cannot be isolated. "
            f"prompt={prompt[:200]!r} full={full[:200]!r}"
        )
    return {"prompt": prompt, "completion": full[len(prompt) :], "text": full}
