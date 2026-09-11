"""The trajectory gate: every discipline fires on its own tamper, silence on merit.

Spec §5.8's verification, made into a decision table: each class of violation
-- shape, budget, mode, replay, grounding, acknowledgement, and the clauses
shared with every row of the corpus -- gets exactly one tamper that lights it
and nothing else, so a red board row always names one cause. The
*untampered* rows of every scheduled shape (no-call, clean, and four of the
five faults) must read all-clear through the same gate that will refuse them
in ``verify_v3`` -- one policy, three call sites, no drift between the
families.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_agentic_fixture as harness  # noqa: E402
from cosimo.tools import SCHEMAS  # noqa: E402
from pipelines.v3.oracle import faults, runtime  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.agentic import render_agentic_row  # noqa: E402
from pipelines.v3.verification.agentic import (  # noqa: E402
    GROUNDING_TAG,
    REPLAY_TAG,
    gate_violations,
    trajectory_violations,
)

PINNED = ("valuation.equity.dcf", "mature_consumer", 0)
RANK_CLEAN, RANK_NO_CALL = 1, 0
RANK_EMPTY, RANK_STALE, RANK_TICKER, RANK_RATE = 2, 7, 12, 17


def _row(rank: int, work_type: str = PINNED[0], family: str = PINNED[1]):
    """A shipped agentic row from the scripted dummy, plus its recomputed pack.

    The pack here is ``compute_pack``'s fresh word -- the same authority
    ``verify_v3`` grades against -- not a copy of anything the row carries.
    """
    pack = compute_pack(work_type, family, 0).to_dict()
    outcome = render_agentic_row(
        harness.ScriptedTeacher(pack, rank, {}),
        {**pack, "id": "t", "verification": {}},
        rank=rank,
    )
    assert outcome["row"] is not None, outcome["dead_letter"]["violations"]
    return pack, outcome["row"]


def _judge(row: dict, pack: dict, **over) -> list[str]:
    render_field = row["verification"]["render"]
    return gate_violations(
        pack,
        row["messages"],
        mode=over.get("mode", render_field["mode"]),
        fault=over.get("fault", render_field["fault"]),
    )


def _call_tool_pair(row: dict, index: int = 0):
    """The (assistant-call, tool-result) pair at *index*, wire form intact."""
    pairs = []
    pending = None
    for message in row["messages"]:
        if message.get("tool_calls"):
            pending = message
        elif message.get("role") == "tool" and pending is not None:
            pairs.append((pending, message))
            pending = None
    return pairs[index]


# ---------------------------------------------------------------- silence


@pytest.mark.parametrize(
    "rank", [RANK_NO_CALL, RANK_CLEAN, RANK_EMPTY, RANK_STALE, RANK_TICKER, RANK_RATE]
)
def test_the_shipped_rows_read_all_clear_through_the_same_gate(rank):
    """The rows that shipped under the fixture's dummy are the rows the board
    must accept without a word -- else the loop and the gate disagree, and
    the fixture is theatre."""
    pack, row = _row(rank)
    assert _judge(row, pack) == []
    assert (
        trajectory_violations(
            pack,
            row["messages"],
            mode=row["verification"]["render"]["mode"],
            fault=row["verification"]["render"]["fault"],
        )
        == []
    )


def test_gate_and_trajectory_are_one_policy_two_names():
    pack, row = _row(RANK_STALE)
    assert _judge(row, pack) == trajectory_violations(
        pack,
        row["messages"],
        mode="faulted",
        fault="stale_asof",
    )


# ---------------------------------------------------------------- shape


def test_a_foreign_role_is_named_not_tolerated():
    pack, row = _row(RANK_CLEAN)
    intruder = dict(row["messages"][2])
    intruder["role"] = "observer"
    row["messages"][2] = intruder
    assert any("observer" in v and "assistant and tool" in v for v in _judge(row, pack))


def test_an_empty_final_turn_is_refused():
    pack, row = _row(RANK_CLEAN)
    row["messages"][-1]["content"] = "   "
    assert any("final assistant turn is empty" in v for v in _judge(row, pack))


def test_a_call_without_its_result_is_a_lie_about_the_server():
    pack, row = _row(RANK_CLEAN)
    row["messages"] = [m for m in row["messages"] if m.get("role") != "tool"]
    verdict = _judge(row, pack)
    assert any("results were returned" in v for v in verdict)


# ---------------------------------------------------------------- budget


def test_seven_calls_are_one_too_many():
    pack, row = _row(RANK_CLEAN)
    pair = _call_tool_pair(row)
    for _ in range(5):
        row["messages"].insert(-1, dict(pair[0]))
        row["messages"].insert(-1, dict(pair[1]))
    verdict = _judge(row, pack)
    assert any("exceed the budget" in v for v in verdict)


# ---------------------------------------------------------------- mode


def test_a_no_call_row_that_called_is_waste_named_as_such():
    """§5.8 verbatim: the pack already carried the data; the call was waste."""
    pack, row = _row(RANK_NO_CALL)
    call = {"name": "get_fundamentals", "arguments": {"symbol": "YCBK"}}
    clean = runtime.execute(runtime.Call(call["name"], call["arguments"]), pack)
    row["messages"].insert(
        -1,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_fundamentals",
                        "arguments": call["arguments"],
                    },
                }
            ],
        },
    )
    row["messages"].insert(
        -1,
        {
            "role": "tool",
            "name": "get_fundamentals",
            "content": faults.canonical(clean),
        },
    )
    verdict = _judge(row, pack)
    assert len(verdict) == 1, verdict
    assert "waste" in verdict[0] and "no_call" in verdict[0]


def test_a_looping_row_that_never_looped_is_named_as_such():
    pack, row = _row(RANK_CLEAN)
    # §A: the shipped transcript opens on the user's goal, so the "goal then
    # answer, nothing between" shape is two turns rather than three.
    row["messages"] = [row["messages"][0], row["messages"][-1]]
    verdict = _judge(row, pack)
    assert any("no tool was called at all" in v for v in verdict)


# ---------------------------------------------------------------- replay


def test_one_edited_digit_in_a_stored_block_fails_the_replay():
    """A trajectory is a claim about what the oracle said; the claim is
    audited by saying it again. One unit on one digit is enough."""
    pack, row = _row(RANK_CLEAN)
    _assistant, block = _call_tool_pair(row)
    payload = json.loads(block["content"])
    payload["rows"]["enterprise_value_m"] = (
        float(payload["rows"]["enterprise_value_m"]) * 1.0001
    )
    block["content"] = faults.canonical(payload)
    verdict = _judge(row, pack)
    assert len(verdict) == 1, verdict
    assert verdict[0].startswith(REPLAY_TAG)


def test_a_fabricated_block_fails_the_replay_even_when_well_shaped():
    pack, row = _row(RANK_CLEAN)
    _assistant, block = _call_tool_pair(row)
    block["content"] = faults.canonical(
        {"rows": {"made_up": 1.5}, "as_of": pack["as_of"]}
    )
    verdict = _judge(row, pack)
    assert any(v.startswith(REPLAY_TAG) for v in verdict)


# ---------------------------------------------------------------- grounding


def test_a_number_from_neither_pack_nor_results_is_invention():
    pack, row = _row(RANK_CLEAN)
    row["answer"] = row["answer"] + " The risk-free rate is 8153.7729 percent."
    row["messages"][-1]["content"] = row["answer"]
    verdict = _judge(row, pack)
    assert len(verdict) == 1, verdict
    assert verdict[0].startswith(GROUNDING_TAG)
    assert "8153.7729" in verdict[0]


def test_a_figure_quoted_from_a_faulted_block_is_ground_legally():
    """The union in "pack ∪ tool results" includes the dirty blocks: a
    trajectory may cite what it was shown -- what makes a fault survivable
    is naming it, never hiding it."""
    pack, row = _row(RANK_STALE)
    assert any("2019" in line for line in row["answer"].splitlines())
    assert _judge(row, pack) == []


# ---------------------------------------------------------------- acknowledgement


def test_an_unacknowledged_fault_is_a_dead_letter_not_a_near_miss():
    pack, row = _row(RANK_STALE)
    ack = harness._ACK_SENTENCES["stale_asof"]
    tampered = "\n".join(line for line in row["answer"].splitlines() if line != ack)
    row["answer"] = tampered
    row["messages"][-1]["content"] = tampered
    verdict = _judge(row, pack)
    assert any("never names it" in v and "stale_asof" in v for v in verdict)
    assert any("stale" in v or "as of" in v for v in verdict if "say one of" in v)


def test_a_rate_limited_call_must_be_reissued_or_waived_in_words():
    pack, row = _row(RANK_RATE)
    pairs = []
    pending = None
    for message in row["messages"]:
        if message.get("tool_calls"):
            pending = message
        elif message.get("role") == "tool" and pending is not None:
            pairs.append((pending, message))
            pending = None
    first, first_block = pairs[0]
    assert json.loads(first_block["content"]).get("__fault__") == "rate_limit"
    # drop the re-issue pair: the refusal stands unanswered
    del row["messages"][
        row["messages"].index(first) : row["messages"].index(pairs[1][1]) + 1
    ]
    verdict = _judge(row, pack)
    assert any("re-issues the identical call" in v for v in verdict)


def test_a_waiver_in_words_is_an_honest_way_to_proceed_without():
    pack, row = _row(RANK_RATE)
    pending = None
    positions = []
    for position, message in enumerate(row["messages"]):
        if message.get("tool_calls"):
            pending = position
        elif message.get("role") == "tool" and pending is not None:
            positions.append((pending, position))
            pending = None
    first_call, first_result = positions[0]
    assert (
        json.loads(row["messages"][first_result]["content"]).get("__fault__")
        == "rate_limit"
    )
    reissue_call, reissue_result = positions[1]
    del row["messages"][reissue_call : reissue_result + 1]  # drop the re-issue pair
    row["answer"] += "\nProceeded without the call; the pack alone answers this."
    row["messages"][-1]["content"] = row["answer"]
    assert _judge(row, pack) == []


# ---------------------------------------------------------------- the shared clauses


def test_an_ignored_must_mention_is_missed_by_name():
    """Dropping a point's content from the answer must be named by the gate.

    The tamper deletes every word of the anchor, not the one line that used to
    equal it verbatim. That older tamper stopped meaning anything when
    ``must_mention`` became anchors matched on content words in any order
    (verification/prose.py): removing a single line left the anchor's terms
    scattered through the rest of the answer, so the gate correctly still saw
    coverage and the test failed while measuring nothing.
    """
    import re

    pack, row = _row(RANK_CLEAN)
    point = pack["must_mention"][0]
    terms = {w for w in re.findall(r"[a-z0-9]+", point.casefold().replace("-", " "))}
    tampered = " ".join(
        word
        for word in row["answer"].split()
        if not any(
            term in word.casefold().replace("-", " ") for term in terms if len(term) > 3
        )
    )
    row["answer"] = tampered
    row["messages"][-1]["content"] = tampered
    verdict = _judge(row, pack)
    assert any("must_mention not covered" in v and repr(point) in v for v in verdict)


def test_an_asserted_forbidden_claim_is_hit_by_name():
    pack, row = _row(RANK_CLEAN)
    assert _judge(row, pack) == []  # silent on the pack's own taboos
    claim = pack["forbidden_claims"][0]
    tampered = row["answer"] + "\n" + claim
    row["answer"] = tampered
    row["messages"][-1]["content"] = tampered
    verdict = _judge(row, pack)
    assert any("forbidden claim asserted" in v and repr(claim) in v for v in verdict)


def test_the_exam_tag_does_not_belong_behind_a_desk_answer():
    pack, row = _row(RANK_CLEAN)
    row["answer"] = "FINAL ANSWER: " + row["answer"]
    row["messages"][-1]["content"] = row["answer"]
    assert any("exam contract" in v for v in _judge(row, pack))


def test_tool_markup_in_the_final_turn_is_refused():
    pack, row = _row(RANK_CLEAN)
    row["answer"] = (
        row["answer"]
        + '\n<tool_call>{"name": "get_fundamentals", "arguments": {}}</tool_call>'
    )
    row["messages"][-1]["content"] = row["answer"]
    assert any("tool-call markup" in v for v in _judge(row, pack))


# ---------------------------------------------------------------- the row's own claims


def test_advertised_schemas_resolve_in_the_registry():
    for rank in (RANK_NO_CALL, RANK_CLEAN, RANK_RATE):
        _pack, row = _row(rank)
        advertised = {s["function"]["name"] for s in row["tool_schemas"]}
        assert advertised <= set(SCHEMAS)
        assert advertised == set(row["tool_names"])


def test_the_render_ledger_says_who_ran_and_what_was_executed():
    _pack, row = _row(RANK_RATE)
    render_field = row["verification"]["render"]
    assert render_field["kind"] == "agentic"
    assert render_field["mode"] == "faulted"
    assert render_field["fault"] == "rate_limit"
    assert render_field["tool_calls"] == 3  # refuse, re-issue, the second ask
    assert render_field["attempts"] == render_field["tool_calls"] + 1
