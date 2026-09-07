#!/usr/bin/env python3
"""Generate ``teacher_echo.json``, the committed offline teacher fixture.

The echo oracle is the self-checking half of the offline suite: for one
pinned pack it answers the *real* prompt that :mod:`pipelines.v3.teacher`
would send, with text assembled only from the pack's own fields -- the question
verbatim plus one sentence per ``must_mention`` item. That makes it clean under
the invented-number gate *by construction*, and it means a fixture run and a
live run of PR2's renderer differ in one variable: who wrote the prose.

Determinism: the pack is pinned to one coordinate
(``valuation.equity.dcf`` / ``mature_consumer`` / variant 0); the request hash
is :func:`pipelines.v3.teacher.client.canonical_request` over the exact
messages the routed teacher would post. Run from repo root:

    .venv/bin/python dataset/tests/v3/fixtures/make_teacher_echo.py

The committed json must match byte-for-byte; test_teacher_client enforces it.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_TESTS = os.path.dirname(_HERE)
DATASET = os.path.dirname(os.path.dirname(_V3_TESTS))
for _p in (DATASET, os.path.dirname(DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.teacher import prompts, routing  # noqa: E402
from pipelines.v3.teacher.client import canonical_request  # noqa: E402

PINNED = ("valuation.equity.dcf", "mature_consumer", 0)
FIXTURE_NAME = "teacher_echo.json"

#: The offline model name. Not in routing's defaults on purpose: no live
#: deployment should ever be able to *accidentally* name a model that only the
#: replay file defines.
ECHO_MODEL = "fixture-echo"


def echo_text(pack: dict) -> str:
    """A completion with zero room to be wrong about a number.

    Question verbatim (the pack guarantees every token in it is allowed --
    test_packs_registry pins that), then every must_mention point as its own
    sentence. Nothing else: no numbers of our own scaffolding, not even a
    count of the points covered.
    """
    sentences = [pack["question"].strip()]
    sentences.extend(point.strip().rstrip(".") + "." for point in pack["must_mention"])
    return "\n".join(sentences)


def request_body(pack: dict) -> dict:
    """The exact body the routed analysis teacher would post for *pack*.

    Shared between :func:`build_fixture` and the tests that replay against
    it, so the hash recorded here is the hash a real offline render produces
    -- the fixture can not be "nearly" the request, matching is exact or the
    entry is dead weight that will never fire.
    """
    decision = routing.route("analysis")
    body = {
        "model": ECHO_MODEL,
        "messages": prompts.render_brief(pack, kind="analysis"),
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    if decision.think:
        body["thinking"] = {"type": "enabled"}
    return body


def build_fixture() -> dict:
    work_type, family, variant = PINNED
    pack = compute_pack(work_type, family, variant).to_dict()
    answer = {
        "model": ECHO_MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": echo_text(pack)},
            }
        ],
        "usage": {"total_tokens": max(1, len(echo_text(pack).split()))},
    }
    entries = {canonical_request(request_body(pack)): answer}
    # The wildcard is the smoke run's escape valve: renderers may send briefs
    # for packs this file never covered, and the run must not stop -- but the
    # text is still *from the file*, replayed, never composed at call time.
    entries["*"] = {
        "model": ECHO_MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Understood."},
            }
        ],
        "usage": {"total_tokens": 1},
    }
    return {
        "version": 1,
        "pinned": list(PINNED),
        "model": ECHO_MODEL,
        "entries": entries,
    }


def main() -> int:
    payload = build_fixture()
    out = os.path.join(_HERE, FIXTURE_NAME)
    with open(out, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(f"wrote {out}: {len(payload['entries'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
