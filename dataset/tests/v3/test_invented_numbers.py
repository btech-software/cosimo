"""The invented-number gate: the rule the whole v3 thesis rests on.

Three layers of confidence, in order of strangeness:

1. arithmetic of the gate itself (tolerance, scaling, fail-closed);
2. the smoke packs' own questions must pass their own ``allowed_numbers`` --
   if fact-computer print discipline and the gate ever disagree about what
   "the same number" means, the packs become unsatisfiable, so this file is
   where the disagreement dies;
3. the committed teacher fixture's answer passes, for the pinned pack, the
   same gate -- the offline render run of PR2 will therefore exercise a
   completion that this file has already proved clean.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from pipelines.v3.packs import FAMILIES, PackError, compute_pack
from pipelines.v3.verification.invented_numbers import (
    REL_TOLERANCE,
    clean,
    invented_numbers,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "fixtures"))
import make_teacher_echo  # noqa: E402


def test_plain_allowed_values_pass_under_every_spelling():
    allowed = [0.03, 0.125, 1250000.0]
    assert clean("the margin is 3.00% and the ratio reads 0.125", allowed)
    assert clean("AUM of 1,250,000.00 this year", allowed)
    assert clean("flat 0.03 growth", allowed)


def test_an_invented_number_is_named_not_silently_dropped():
    got = invented_numbers("we assume a terminal multiple of 14.2x", [3.0, 0.03, 0.125])
    assert got == ["14.2"]


def test_repeats_are_reported_once_kept_in_order():
    got = invented_numbers("7.13 then 8.14 then 7.13 again", [3.0])
    assert got == ["7.13", "8.14"]


def test_tolerance_is_relative_not_absolute():
    allowed = [1000.0]
    assert clean("about 1004", allowed)  # 0.4% < 0.5% relative
    assert invented_numbers("about 1060", allowed) == ["1060"]  # 6% off
    assert invented_numbers("about 60", allowed) == ["60"]  # a real ratio slip
    assert 0 < REL_TOLERANCE <= 0.01


def test_scaling_is_limited_to_the_honest_spellings():
    # 0.03 may honestly appear as 0.03 (x1) or 3.0% (x100); a 30.0 "per
    # mille" reading (x1000) is not one of them -- a unit hallucination must
    # not be able to pass the gate by reformatting itself.
    assert clean("a 3.0% share of 0.03", [0.03])
    assert clean("plain 0.03", [0.03])
    assert invented_numbers("a 30.0 per-mille claim about 0.03", [0.03]) == ["30.0"]


def test_zero_and_signs_parse_the_way_the_v2_tokenizer_does():
    assert clean("a drawdown of 0", [0.0])
    assert invented_numbers("a drawdown of -0.2", [0.0]) == ["-0.2"]


def test_the_gate_fails_closed_on_a_missing_or_broken_contract():
    with pytest.raises(ValueError, match="refusing to grade"):
        invented_numbers("margin 3%", [])
    with pytest.raises(ValueError, match="refusing to grade"):
        invented_numbers("margin 3%", [float("nan")])
    # None entries are noise the pack author can no longer mean; they must not
    # widen into "anything goes".
    with pytest.raises(ValueError, match="refusing to grade"):
        invented_numbers("margin 3%", [None, float("nan")])


def test_every_smoke_pack_questions_pass_their_own_gate():
    """Print discipline and the gate agree on what counts as the same number."""
    seen = 0
    for work_type, families in sorted(FAMILIES.items()):
        for family in families:
            for variant in range(6):
                try:
                    pack = compute_pack(work_type, family, variant)
                except PackError:
                    continue
                seen += 1
                offenders = invented_numbers(pack.question, pack.number_set())
                assert offenders == [], (
                    f"{pack.scenario_id}#{variant} question quotes {offenders}, "
                    "outside its own allowed_numbers"
                )
    assert seen >= 40, (
        "sweep too thin to prove the agreement (did the registry shrink?)"
    )


def test_the_committed_fixture_answer_is_clean_under_the_same_gate():
    path = os.path.join(_HERE, "fixtures", "teacher_echo.json")
    with open(path, encoding="utf8") as handle:
        fixture = json.load(handle)
    work_type, family, variant = fixture["pinned"]
    pack = compute_pack(work_type, family, variant)
    answer = fixture["entries"][
        make_teacher_echo.canonical_request(
            make_teacher_echo.request_body(pack.to_dict())
        )
    ]
    content = answer["choices"][0]["message"]["content"]
    assert clean(content, pack.number_set())
    assert pack.question in content
    for point in pack.must_mention:
        assert point.rstrip(".") in content
