"""The committed pair fixture's duties: regenerate byte-identical, and prove
its meta honest about the crimes it has seen.

The drift test is the same argument the prose and implementation fixtures
make -- a fixture nobody can regenerate is folklore. This lane adds one
test the others have no need of: the walk must have *met every licensed
crime*. A preference fixture that never drew ``overconfident_abstention_fail``
has not tested that detector, and a suite that passes over it would be
agreeing with a table it never interrogated.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_preference_fixture as pref_harness  # noqa: E402
from pipelines.v3 import config  # noqa: E402
from pipelines.v3.verification.preference import PITFALL_INSTRUCTIONS  # noqa: E402

COMMITTED = os.path.join(_HERE, "fixtures", pref_harness.PREF_FIXTURE_NAME)


def _regenerate_bytes(payload: dict, tmp_path) -> bytes:
    path = os.path.join(str(tmp_path), "regen.json")
    with open(path, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    with open(path, "rb") as handle:
        return handle.read()


def test_the_committed_fixture_is_byte_identical_to_its_harness(tmp_path):
    payload = pref_harness.build_fixture(
        pref_harness.DEFAULT_TYPES, pref_harness.DEFAULT_LIMIT
    )
    with open(COMMITTED, "rb") as handle:
        committed = handle.read()
    assert _regenerate_bytes(payload, tmp_path) == committed, (
        "the committed fixture drifted from its harness; regenerate it with "
        ".venv/bin/python dataset/tests/v3/fixtures/make_preference_fixture.py "
        "and inspect the diff before accepting it"
    )


def test_the_walk_met_every_kind_and_every_licensed_crime():
    with open(COMMITTED, encoding="utf8") as handle:
        payload = json.load(handle)
    meta = payload["meta"]
    assert meta["entries"] == 2 * meta["pairs"], (
        "both sides or none: no half-pairs in the table"
    )
    assert payload["model"] == pref_harness.DUMMY_MODEL
    assert set(meta["types"]) == set(config.PREF_PROBABILITIES), (
        "a parent kind the walk never visited is a lane the pair board never read"
    )
    assert set(meta["pitfalls"]) == set(PITFALL_INSTRUCTIONS), (
        "a detector the fixture never drew is a detector the suite never ran"
    )
    assert meta["pairs"] >= 20, "the committed walk is thinner than the agreed sample"
