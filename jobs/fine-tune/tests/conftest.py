"""Shared fixtures for the harness unit tests.

The suite is CPU-only and offline: no torch, no GPU, no model or dataset
download. It exercises the pure-logic modules (``config``, ``chat``,
``data_schema``, ``splits``, ``grading``) and the shipped YAML/Jinja assets.

Run it from the repository root with the host venv::

    .venv/bin/python -m pytest jobs/fine-tune/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[1]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from cosimo_ft import config as config_mod  # noqa: E402

CONFIG_DIR = HARNESS_ROOT / "configs"
CHAT_TEMPLATE_PATH = CONFIG_DIR / "chat_template.jinja"

# The vendor template shipped with unsloth/Phi-4-mini-reasoning, verbatim. The
# student moved to Qwen3.8 (configs/base.yaml; the Phi run is archived as
# configs/base.phi4.yaml), but this stays as the negative control: the tests
# asserting the harness template is free of a vendor identity preamble are
# meaningless unless the same assertion fails for a template that carries one.
VENDOR_CHAT_TEMPLATE = (
    "{{ '<|system|>Your name is Phi, an AI math expert developed by Microsoft.' }}"
    "{% for message in messages %}{% if message['role'] == 'system' %}"
    " {{ message['content'] }}"
    "{% if 'tools' in message and message['tools'] is not none %}"
    "{{ '<|tool|>' + message['tools'] + '<|/tool|>' }}{% endif %}{% endif %}"
    "{% endfor %}{{ '<|end|>' }}"
    "{% for message in messages %}{% if message['role'] != 'system' %}"
    "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}"
    "{% else %}{{ eos_token }}{% endif %}"
)

# ChatML: under Qwen the turn terminator IS the EOS token, which is why the
# shipped template stops on <|im_end|> rather than appending eos_token after it.
EOS_TOKEN = "<|im_end|>"


class FakeTokenizer:
    """Stand-in for a transformers tokenizer, template semantics only.

    Reproduces what the shipped ``configs/chat_template.jinja`` does without
    needing Jinja: system turns first, then the remaining turns, each closed
    with ``<|im_end|>`` and separated by a newline, then the generation prompt
    when one was asked for. ``chat.render_*`` only ever calls
    ``apply_chat_template``, which is what makes this substitution legitimate.

    The trailing newline is emitted for every turn *except* the last one of a
    non-generation render, so a completed conversation ends exactly on the EOS
    token -- the invariant ``data_schema.to_pref_row`` strips against.
    """

    eos_token = EOS_TOKEN

    def __init__(self) -> None:
        self.chat_template: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        if tokenize:
            raise NotImplementedError("the harness always renders with tokenize=False")
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        parts = [f"<|im_start|>system\n{m['content']}<|im_end|>\n" for m in system]
        for index, message in enumerate(rest):
            last = index == len(rest) - 1
            separator = "\n" if not last or add_generation_prompt else ""
            parts.append(
                f"<|im_start|>{message['role']}\n{message['content']}"
                f"<|im_end|>{separator}"
            )
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class JinjaTokenizer:
    """Renders a real Jinja chat template, the way transformers does.

    Used to check the *shipped* template file rather than a Python restatement of
    it, so template drift is caught by the test suite.
    """

    eos_token = EOS_TOKEN

    def __init__(self, template: str) -> None:
        import json

        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        # transformers replaces Jinja's own `tojson` with a plain json.dumps.
        # Jinja's default escapes <, > and & for HTML safety, which would mangle
        # every rendered tool schema and make these tests assert a string
        # production never produces.
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs
        )
        self.chat_template = template
        self._template = env.from_string(template)

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        tools: list[dict] | None = None,
    ) -> str:
        if tokenize:
            raise NotImplementedError("the harness always renders with tokenize=False")
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            eos_token=self.eos_token,
            tools=tools,
        )


@pytest.fixture(scope="session")
def cfg() -> dict:
    """The resolved default config (configs/base.yaml only)."""
    return config_mod.load_config()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture(scope="session")
def chat_template_text() -> str:
    """The shipped harness chat template, as it lives on disk."""
    return CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def jinja_tokenizer(chat_template_text: str) -> JinjaTokenizer:
    """A tokenizer backed by the shipped template; skipped when Jinja is absent."""
    pytest.importorskip("jinja2", reason="jinja2 is not installed in this venv")
    return JinjaTokenizer(chat_template_text)


@pytest.fixture
def vendor_tokenizer() -> JinjaTokenizer:
    """A tokenizer backed by the vendor template, used as a negative control."""
    pytest.importorskip("jinja2", reason="jinja2 is not installed in this venv")
    return JinjaTokenizer(VENDOR_CHAT_TEMPLATE)
