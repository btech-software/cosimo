"""Guards on the test suite itself, not on the harness.

One rule, and it is here because breaking it produces a failure that looks like
a code bug and is reproducible only under a particular command line.

``conftest.py`` is imported by pytest as a top-level module literally named
``conftest``, and pytest pre-imports the conftests of every initial argument
before collecting anything. Two test trees in one run therefore both claim that
name, and the last one imported wins in ``sys.modules``. So a
``from conftest import ...`` anywhere in this suite resolves against whichever
directory happened to be named last::

    pytest dataset/tests/v3 jobs/fine-tune/tests   # jobs' conftest wins, passes
    pytest jobs/fine-tune/tests dataset/tests/v3   # dataset's wins, ImportError

Same tests, same code, opposite results. ``make v3-test`` uses the first order,
which is why this hid: the shipped command works and the obvious reordering of
its arguments does not.

Shared helpers therefore live in ``harness_fixtures.py``, whose name nothing
else in the repository claims.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

# Every module in this directory except conftest.py itself, which is allowed to
# import the helpers -- it is the one file pytest resolves by path rather than
# by module name.
SUITE_MODULES = sorted(
    path
    for path in TESTS_DIR.glob("*.py")
    if path.name not in {"conftest.py", "harness_fixtures.py"}
)


def imported_module_names(path: Path) -> set[str]:
    """Top-level module names the file imports, from its AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_suite_has_modules_to_check():
    """A glob that matched nothing would make every assertion below vacuous."""
    assert len(SUITE_MODULES) >= 5


@pytest.mark.parametrize("path", SUITE_MODULES, ids=lambda p: p.name)
def test_no_test_module_imports_conftest_by_name(path: Path):
    assert "conftest" not in imported_module_names(path), (
        f"{path.name} imports `conftest` by module name. Two test trees in one "
        "pytest run both bind that name and the last one imported wins, so this "
        "passes or fails depending on argument order. Import from "
        "`harness_fixtures` instead."
    )


def test_the_helpers_module_is_not_named_conftest():
    """The fix itself: helpers live under a name no other tree claims."""
    helpers = TESTS_DIR / "harness_fixtures.py"
    assert helpers.is_file()
    # If another conftest.py appears beside it, the collision is back.
    assert [p.name for p in TESTS_DIR.glob("conftest.py")] == ["conftest.py"]
