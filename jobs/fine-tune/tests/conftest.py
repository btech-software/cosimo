"""Fixtures for the harness unit tests.

Fixtures only. The helpers they wrap -- ``FakeTokenizer``, ``JinjaTokenizer``,
``CONFIG_DIR``, ``EOS_TOKEN``, ``VENDOR_CHAT_TEMPLATE`` -- live in
``harness_fixtures.py``, because a module named ``conftest`` is not safely
importable by name when more than one test tree is in the same run. See that
module's docstring for the mechanism; ``test_suite_hygiene.py`` guards it.

The suite is CPU-only and offline: no torch, no GPU, no model or dataset
download.

Run it from the repository root with the host venv::

    .venv/bin/python -m pytest jobs/fine-tune/tests -q
"""

from __future__ import annotations

import pytest

# Puts HARNESS_ROOT on sys.path as an import side effect, which is what makes
# `from cosimo_ft import ...` work in every test module. conftest.py is imported
# before any of them, so this is the earliest hook available.
from harness_fixtures import (
    CHAT_TEMPLATE_PATH,
    VENDOR_CHAT_TEMPLATE,
    FakeTokenizer,
    JinjaTokenizer,
)

from cosimo_ft import config as config_mod  # noqa: E402


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
