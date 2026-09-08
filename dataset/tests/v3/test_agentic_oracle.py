"""The oracle's two halves: honest projections (runtime) and the dirty layer (faults).

The load-bearing property of :mod:`pipelines.v3.oracle.runtime` is one line
of the spec read strictly -- *every number in a result is a projection of a
number the pack already owns*. It is tested here at the source, walking the
parsed payload (keys are labels, not quantities: a key like ``z95`` must not
be read as a number) and demanding set membership against the recomputed
pack's own number set. Everything else in the agentic verification stack --
the replay, the grounding subset, the fixture -- rests on it, and none of
them could say so as confidently if the oracle could mint figures of its own.

:mod:`pipelines.v3.oracle.faults` is tested as the planner it claims to be:
a pure function of the selector's ordinal, whose arithmetic makes the PR3
gate's mix exact rather than approximate, and five degradations each of
which degrades only the block, never the data's honesty about the pack.
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
from pipelines.v3.oracle import faults, runtime  # noqa: E402
from pipelines.v3.oracle.runtime import Call, OracleError  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.verification.invented_numbers import invented_numbers  # noqa: E402
from pipelines.v3.verification.prose import whitelist_for  # noqa: E402

PINNED = ("valuation.equity.dcf", "mature_consumer", 0)
BOOK = ("risk.market.var_es", "rates_book", 0)
TCA = ("execution.tca.arrival", "large_cap_intraday", 0)


def _policy(pack: dict, rank: int = 1):
    """The fixture's scripted policy, borrowed to *ask* the oracle with."""
    return harness.ScriptedTeacher(pack, rank, {})


def _values(node, floor=None):
    """Every numeric value in a parsed payload, keys never read as numbers."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _values(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _values(value)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield float(node)


# ---------------------------------------------------------------- runtime


def test_every_registered_tool_has_an_executant_and_vice_versa():
    """ "A tool not in the serving app does not appear in the corpus" -- and
    the converse: the corpus must be able to answer every tool it advertises,
    or the conversations it ships describe a server that cannot speak."""
    assert set(runtime.SCHEMA_NAMES) == set(runtime._EXECUTANTS)
    for tools in runtime.TOOL_AVAILABILITY.values():
        assert set(tools) <= set(runtime.SCHEMA_NAMES)


def test_projections_never_mint_new_figures():
    """Walked from every answerable call of every work type, every number in
    the block is a number of the pack -- not merely *like* one."""
    for work_type, family in (
        PINNED[:2],
        BOOK[:2],
        TCA[:2],
        ("valuation.equity.multiples", "specialty_retail"),
        ("portfolio.attribution.brinson_carino", "global_equity_long"),
    ):
        pack = compute_pack(work_type, family, 0).to_dict()
        authority = pack_number_set(pack)
        policy = _policy(pack)
        needed = list(runtime.tools_for(work_type))
        taken: list[str] = []
        for occurrence, name in enumerate(needed):
            call = policy._call_for(name, occurrence, len(needed), taken)
            taken.append(name)
            result = runtime.execute(Call(name, call["arguments"]), pack)
            for value in _values(json.loads(faults.canonical(result))):
                assert any(
                    abs(value - a) <= 1e-12 * max(1.0, abs(a)) for a in authority
                ), f"{work_type}/{name} minted {value!r}, which the pack does not own"


def pack_number_set(pack: dict) -> set:
    return {float(x) for x in pack.get("allowed_numbers") or []}


def test_unknown_tool_raises_rather_than_empties():
    """An empty answer is the pack saying "I have none"; an unknown tool is a
    programming bug, and the two must not sound alike in the shard."""
    pack = compute_pack(*PINNED).to_dict()
    with pytest.raises(OracleError):
        runtime.execute(Call("get_weather", {}), pack)


def test_a_mistaken_identity_answers_honestly_and_emptily():
    """Called for a symbol the scenario does not name: no rows, a note, and
    not one figure the pack does not author -- the gate must be able to tell
    an honest empty from a fabricated block, and an empty speaks no digits."""
    pack = compute_pack(*PINNED).to_dict()
    result = runtime.execute(
        Call("get_fundamentals", {"symbol": "NOPE", "metrics": ["wacc"]}), pack
    )
    text = faults.canonical(result)
    assert result["rows"] == {}
    assert "no rows" in result["note"]
    assert invented_numbers(text, pack["allowed_numbers"], whitelist_for(pack)) == []


def test_compute_metrics_echoes_the_scenarios_own_inputs_when_args_disagree():
    """The correcting echo, not a recomputed valuation: a caller arguing from
    someone else's wacc is told what the scenario says, in the scenario's own
    figures -- so a mistaken call is data the teacher can retry against."""
    pack = compute_pack(*PINNED).to_dict()
    result = runtime.execute(
        Call(
            "compute_metrics",
            {
                "fcff": pack["computed"]["fcff_year1_m"],
                "wacc": 0.4142,
                "terminal_growth": pack["inputs"]["terminal_growth"],
                "proj_years": pack["inputs"]["explicit_years"],
            },
        ),
        pack,
    )
    assert "expected" in result and "do not match" in result["status"]
    authority = pack_number_set(pack)
    for value in _values(result["expected"]):
        assert any(abs(value - a) <= 1e-12 * max(1.0, abs(a)) for a in authority)


def test_matching_arguments_yield_the_derived_block():
    pack = compute_pack(*PINNED).to_dict()
    inputs, computed = pack["inputs"], pack["computed"]
    result = runtime.execute(
        Call(
            "compute_metrics",
            {
                "fcff": computed["fcff_year1_m"],
                "wacc": inputs["wacc"],
                "terminal_growth": inputs["terminal_growth"],
                "proj_years": inputs["explicit_years"],
            },
        ),
        pack,
    )
    assert result["method"] == "two-stage dcf"
    assert result["rows"].get("enterprise_value_m") == computed["enterprise_value_m"]


# ---------------------------------------------------------------- faults


def test_schedule_of_is_a_pure_function_and_the_mix_is_exact():
    """The PR3 gate's line, as arithmetic: within twenty consecutive ranks,
    four no-call conversations, four faulted ones, and twelve clean loops --
    'every fifth job', written down and believed."""
    schedules = [faults.schedule_of(rank) for rank in range(20)]
    modes = [schedule.mode for schedule in schedules]
    assert modes.count("no_call") == 4
    assert modes.count("faulted") == 4
    assert modes.count("clean") == 12
    assert modes == [faults.schedule_of(rank).mode for rank in range(20)]  # pure
    no_call = [rank for rank in range(20) if modes[rank] == "no_call"]
    faulted = [rank for rank in range(20) if modes[rank] == "faulted"]
    assert not set(no_call) & set(faulted)  # a no-call job has nothing to fault
    drawn = {faults.schedule_of(rank).fault for rank in faulted}
    assert len(drawn) == 4, "the rotation must not stall on one fault"


def test_schedule_of_refuses_an_unsorted_rank():
    with pytest.raises(faults.FaultError):
        faults.schedule_of(-1)


def test_the_five_faults_are_the_taxonomy_and_each_tags_itself():
    assert faults.FAULTS == (
        "empty_result",
        "stale_asof",
        "wrong_ticker",
        "rate_limit",
        "schema_drift",
    )
    assert set(faults.FAULT_ACK) == set(faults.FAULTS)
    clean = {"rows": {"w": 1.5}, "as_of": "2024-06-28"}
    for fault in faults.FAULTS:
        polluted = json.loads(
            faults.canonical(faults.inject("get_fundamentals", clean, fault))
        )
        assert polluted["__fault__"] == fault
        assert polluted["tool"] == "get_fundamentals"


def test_injection_is_deterministic_and_pure():
    clean = {"rows": {"w": 1.5}}
    first = faults.canonical(faults.inject("get_positions", clean, "empty_result"))
    second = faults.canonical(faults.inject("get_positions", clean, "empty_result"))
    assert first == second
    assert clean == {"rows": {"w": 1.5}}  # the clean block was not corrupted


def test_stale_asof_carries_the_stale_stamp_and_an_acknowledgable_date():
    clean = {"rows": {"w": 1.5}, "as_of": "2024-06-28"}
    polluted = json.loads(
        faults.canonical(faults.inject("get_returns_series", clean, "stale_asof"))
    )
    assert polluted["as_of"] == faults.STALE_AS_OF
    assert polluted["stale"] is True


def test_wrong_ticker_rotates_the_identifier_and_names_the_mismatch():
    clean = {"rows": {"w": 1.5}, "symbol": "QANX"}
    polluted = json.loads(
        faults.canonical(
            faults.inject("get_fundamentals", clean, "wrong_ticker", requested="QANX")
        )
    )
    assert polluted["symbol"] != "QANX"
    assert polluted["symbol"].isalpha()  # rotated through the alphabet, digit-free
    assert "not the one requested" in polluted["warning"]
    assert polluted["requested"] == "QANX"


def test_schema_drift_renames_fields_and_keeps_every_figure():
    """The field *names* drift; the figures do not move -- so a drifted block
    is survivable precisely because the numbers in it are still the pack's."""
    clean = {"rows": {"portfolio_return": 0.01, "active_bps": 12.0}}
    polluted = json.loads(
        faults.canonical(faults.inject("get_positions", clean, "schema_drift"))
    )
    assert set(polluted["rows"]) == {"portfolio_retur", "active_bp"}
    assert sorted(_values(polluted["rows"])) == sorted(_values(clean["rows"]))


def test_rate_limit_refuses_once_and_says_how_to_be_answered_again():
    """The refusal is the whole lesson: an error block with no figure to
    quote and a note in words the gate's vocabulary will recognise, so the
    desk can re-issue the identical call and be answered."""
    clean = {"rows": {"w": 1.5}}
    polluted = json.loads(
        faults.canonical(faults.inject("get_fundamentals", clean, "rate_limit"))
    )
    assert "rows" not in polluted  # refused at the gate: no block to quote from
    assert invented_numbers(faults.canonical(polluted), [1.5], ()) == []
    assert any(
        token in faults.canonical(polluted).casefold()
        for token in ("too many requests", "unavailable")
    )


def test_canonical_is_the_one_byte_form_all_three_call_sites_share():
    """Loop, gate and board each dump a tool result; this is the dump they all
    must reach -- sorted keys, tight separators, text unescaped."""
    blob = faults.canonical({"b": 1, "a": {"z": [1.5]}, "u": "café"})
    assert blob == '{"a":{"z":[1.5]},"b":1,"u":"café"}'


def test_injecting_an_unknown_fault_is_an_error_not_a_quiet_clean_block():
    with pytest.raises(faults.FaultError):
        faults.inject("get_positions", {}, "network_partition")


def test_tools_for_maps_the_answerable_surface_per_work_type():
    assert runtime.tools_for("valuation.equity.dcf") == (
        "get_fundamentals",
        "compute_metrics",
    )
    assert set(runtime.tools_for("risk.market.var_es")) == {
        "get_returns_series",
        "get_positions",
    }
    with pytest.raises(OracleError):
        runtime.tools_for("mythology.oracle.dodona")
