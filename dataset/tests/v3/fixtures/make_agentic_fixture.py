#!/usr/bin/env python3
"""Generate ``agentic_fixture.json``: the offline dummy for the agentic stage.

The prose fixture (PR2) proves a *one-shot* completion replays. This one
proves the *loop*: twenty real agentic jobs from the real plan driven through
``render.agentic_stage`` with no network and no GPU -- the arch spec §10 PR3
gate ("20 conversations, 4 with faults, 4 no-call; every final number
grounded") is a pytest over this file.

The dummy is *composed, not captured*, and every reply it gives is the reply
the scripted policy would give from the transcript alone: the teacher object
here derives each answer from the message list and the job's rank, never from
side channels, so an identical ask must produce an identical reply -- and the
build *asserts* that (a body captured twice with different answers is a
broken harness, not a flaky corpus) and re-runs every conversation a second
time through the *real* ``Teacher``/``FixtureTransport`` pair, requiring the
rows byte for byte. A dummy that flunks the gate, or replays differently,
fails this build; it must never fail the corpus.

The scripted policy is deliberately pedestrian, and it is the *loop* that is
under test, not wit: propose the tools the work type answers with, one call
each; re-issue the identical call once when a block comes back rate-limited
(the desk's own reflex, taught by the brief); then close with question,
``must_mention``, the pack's own figures at fixed point, and -- when a fault
was injected -- a sentence naming it, because an unacknowledged fault is a
dead letter by design and a dummy that could not acknowledge one would be
proving nothing.

Determinism: no wall clock, no RNG, no environment reads beyond the deliberate
pin of the lane to the dummy's own model name (the ``make_teacher_echo.py``
convention, so the replay table cannot hash differently on boxes with
different ``TEACHER_*`` defaults). Run from the repo root::

    .venv/bin/python dataset/tests/v3/fixtures/make_agentic_fixture.py

The committed json must match byte-for-byte; the drift test enforces it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_TESTS = os.path.dirname(_HERE)
DATASET = os.path.dirname(os.path.dirname(_V3_TESTS))
for _p in (DATASET, os.path.dirname(DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The completion budget is part of every request body this file hashes, and
# `client.DEFAULT_MAX_TOKENS` reads COSIMO_V3_MAX_TOKENS at import time. A
# developer with that set -- `.env.example` ships it -- would otherwise
# capture their own budget into the committed table and rekey every entry.
# Dropping it here keeps the promise this module's docstring makes: the bytes
# hash the same on every box.
os.environ.pop("COSIMO_V3_MAX_TOKENS", None)

from pipelines.v3 import config, inventory  # noqa: E402
from pipelines.v3.oracle import faults, runtime  # noqa: E402
from pipelines.v3.oracle.runtime import _entity_identity  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.agentic import (  # noqa: E402
    render_agentic_row,
    select_agentic_jobs,
    select_agentic_sample,
)
from pipelines.v3.teacher.client import (  # noqa: E402
    FixtureTransport,
    Teacher,
    _parse,
    build_body,
    canonical_request,
)
from pipelines.v3.verification.agentic import gate_violations  # noqa: E402

AGENTIC_FIXTURE_NAME = "agentic_fixture.json"
DEFAULT_LIMIT, DEFAULT_FAULTED, DEFAULT_NO_CALL = 20, 4, 4

#: The offline model name, pinned into every request body this file keys on
#: (never a deployment lane name -- the replay table's bytes must not depend
#: on a box's ``TEACHER_*`` defaults). The agentic lane is the reasoning one,
#: so an offline run hits these entries by setting ``TEACHER_REASONING`` to
#: this value, exactly the convention ``make_teacher_echo.py`` established.
DUMMY_MODEL = "fixture-agentic"


def _dec(value: float) -> str:
    """Fixed point, no sci notation: the spelling the number gate parses without
    ambiguity (shared policy with ``make_prose_fixture``)."""
    text = f"{float(value):.10f}".rstrip("0")
    if text.endswith("."):
        text = text[:-1]
    return text or "0"


def _scalars(node: dict) -> dict:
    return {
        key: value
        for key, value in node.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


_ACK_SENTENCES = {
    "empty_result": "The block came back with no data; nothing rests on it.",
    "stale_asof": (
        "That response is stale as of 2019-01-01, an outdated stamp; I flag it "
        "and rely on the pack's own as-of."
    ),
    "wrong_ticker": (
        "The block carries a wrong ticker -- a mismatch, not the identifier I "
        "asked for; its figures are another entity's and are not used."
    ),
    "rate_limit": (
        "The first call hit a rate limit and was unavailable; re-issued once, "
        "the block below is what the desk relies on."
    ),
    "schema_drift": (
        "The block arrived with unrecognised, renamed fields -- schema drift; I "
        "read only the figures the registry knows."
    ),
}


class ScriptedTeacher:
    """The dummy: one duck, the ``Teacher.complete`` shape, no side channels.

    Every reply is a pure function of ``(messages, rank)`` -- the transcript
    says what has been asked and answered, the rank (the selector's ordinal)
    says what the schedule planned. ``entries`` collects the
    ``canonical_request(body) -> payload`` pairs the real transport will be
    asked to replay; a collision with a differing payload is raised, because
    two identical asks answering differently is the one bug a replay corpus
    must never ship.
    """

    def __init__(self, pack: dict, rank: int, entries: dict):
        self.pack = pack
        self.rank = rank
        self.schedule = faults.schedule_of(rank)
        self.entries = entries
        self.asked = 0

    # -- the wire, one form both directions ----------------------------------
    def complete(self, messages, *, model, temperature, max_tokens, think, extra=None):
        payload = self._reply(messages)
        body = build_body(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            extra=extra,
        )
        key = canonical_request(body)
        prior = self.entries.get(key)
        if prior is not None and prior != payload:
            raise AssertionError(
                f"impure dummy: the same ask hashed to {key[:16]} twice with "
                "different answers -- the policy reads a side channel"
            )
        self.entries[key] = payload
        self.asked += 1
        return _parse(payload, model)

    def _payload(self, content: str, calls: list[dict] | None) -> dict:
        message: dict = {"content": content, "role": "assistant"}
        if calls:
            message["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in calls
            ]
        else:
            message["reasoning_content"] = "Checked line by line against the pack."
        return {
            "choices": [{"finish_reason": "stop", "message": message}],
            "model": DUMMY_MODEL,
            "usage": {"total_tokens": max(1, len(content.split()))},
        }

    def _reply(self, messages: list[dict]) -> dict:
        calls, results = self._transcript(messages)
        if len(calls) != len(results):
            raise AssertionError("the loop reported a call without its result")
        if self.schedule.mode == "no_call":
            if calls:
                raise AssertionError(
                    "a no_call job proposed a call; the policy is pure"
                )
            return self._payload(self._final_text(), None)
        if results and self._fault(results[-1]) == "rate_limit":
            if len(calls) < 2 or self._key(calls[-1]) != self._key(calls[-2]):
                refused = calls[-1]["function"]
                return self._payload(
                    "", [{"name": refused["name"], "arguments": refused["arguments"]}]
                )  # re-issue, identical
        planned = self._planned(calls, results)
        needed = list(runtime.tools_for(self.pack["work_type"]))
        if len(planned) < len(needed):
            occurrence = len(planned)
            name = needed[occurrence]
            taken = [c["function"]["name"] for c in planned]
            return self._payload(
                "", [self._call_for(name, occurrence, len(needed), taken)]
            )
        return self._payload(self._final_text(), None)

    # -- transcript arithmetic -------------------------------------------------
    @staticmethod
    def _transcript(messages: list[dict]) -> tuple[list[dict], list[dict]]:
        calls = [c for m in messages for c in (m.get("tool_calls") or [])]
        results = [m for m in messages if m.get("role") == "tool"]
        return calls, results

    @staticmethod
    def _key(call: dict) -> str:
        function = call["function"]
        return faults.canonical(
            {"arguments": function["arguments"], "name": function["name"]}
        )

    @staticmethod
    def _fault(result: dict) -> str | None:
        try:
            return json.loads(result.get("content") or "{}").get("__fault__")
        except ValueError:
            return None

    @classmethod
    def _planned(cls, calls: list[dict], results: list[dict]) -> list[dict]:
        """The calls excluding rate-limit re-issues (which are not new asks)."""
        planned: list[dict] = []
        for position, call in enumerate(calls):
            if (
                position >= 1
                and cls._key(call) == cls._key(calls[position - 1])
                and position - 1 < len(results)
                and cls._fault(results[position - 1]) == "rate_limit"
            ):
                continue
            planned.append(call)
        return planned

    def _call_for(
        self, name: str, occurrence: int, needed: int, taken: list[str]
    ) -> dict:
        """The call for the *occurrence*-th planned ask of *name*, from pack data."""
        pack = self.pack
        inputs, computed = pack.get("inputs") or {}, pack.get("computed") or {}
        identity = _entity_identity(pack)
        if name == "get_fundamentals":
            available = sorted({*_scalars(inputs), *_scalars(computed)})
            repeats = taken.count(name)
            parts = max(needed, 1)
            size = -(-len(available) // parts) or 1
            start = repeats * size
            metrics = available[start : start + size] or available[:1]
            return {"name": name, "arguments": {"symbol": identity, "metrics": metrics}}
        if name == "compute_metrics":
            return {
                "name": name,
                "arguments": {
                    "fcff": computed.get("fcff_year1_m"),
                    "wacc": inputs.get("wacc"),
                    "terminal_growth": inputs.get("terminal_growth"),
                    "proj_years": inputs.get("explicit_years"),
                },
            }
        if name == "get_returns_series":
            return {"name": name, "arguments": {"portfolio_id": identity}}
        if name == "get_positions":
            return {"name": name, "arguments": {"book_id": identity}}
        if name == "get_transaction_cost":
            return {
                "name": name,
                "arguments": {
                    "symbol": inputs.get("symbol"),
                    "side": inputs.get("side"),
                    "shares": inputs.get("shares"),
                    "arrival_price": inputs.get("arrival_price"),
                },
            }
        raise AssertionError(f"the scripted policy does not know {name!r}")

    def _final_text(self) -> str:
        pack = self.pack
        lines = [pack["question"].strip()]
        lines += [
            point.strip().rstrip(".") + "." for point in pack.get("must_mention") or []
        ]
        facts = sorted(
            _scalars(pack.get("computed") or {}),
            key=lambda key: (
                0 if abs(float(pack["computed"][key])) >= 1e-4 else 1,
                key,
            ),
        )
        for key in facts[:4]:
            value = float(pack["computed"][key])
            spelling = _dec(value)
            if spelling == "0" and value != 0.0:
                continue  # too small to quote in plain decimals; skip, do not invent
            lines.append(
                f"The {key.replace('_', ' ')} stands at {spelling}, per the pack."
            )
        if self.schedule.fault:
            lines.append(_ACK_SENTENCES[self.schedule.fault])
        lines.append(
            "Every figure above is the pack's own or a block received; nothing "
            "here is drawn from outside them."
        )
        text = "\n".join(lines)
        problems = gate_violations(
            pack,
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}]
            + [{"role": "assistant", "content": text}],
            mode="no_call" if self.schedule.mode == "no_call" else "clean",
            fault=None,
        )
        # Grounding-and-coverage preview: the loop's own gate judges the full
        # transcript; this pre-checks the only parts the dummy itself writes,
        # so a composition bug fails the build, not the corpus.
        # Grounding is deliberately *not* previewed: the closing text of a
        # faulted row may cite a figure that lives only in the faulted block
        # it is naming, and a preview without that block would report an
        # honest quotation as an invention. The loop's own gate, judging the
        # whole transcript, is the authority -- a violation there dead-letters
        # the row and fails this build all the same.
        offenders = [p for p in problems if "must_mention" in p or "forbidden" in p]
        if offenders:
            raise AssertionError(
                f"dummy closing text for {pack['scenario_id']} variant "
                f"{pack['variant']} breaks its own contract: {offenders}"
            )
        return text


def build_fixture(limit: int, faulted: int, no_call: int) -> dict:
    """The PR3-gate sample of the real plan, driven to a replay table."""
    plan = inventory.load_plan(config.taxonomy_path())
    jobs = inventory.expand_jobs(plan)
    sample = select_agentic_sample(jobs, limit=limit, faulted=faulted, no_call=no_call)
    full_slice = list(select_agentic_jobs(jobs))
    ranks_walked = max(rank for rank, _job in sample) + 1 if sample else 0

    os.environ[config.TEACHER_REASONING_ENV] = DUMMY_MODEL
    entries: dict[str, dict] = {}
    rows: dict[int, dict] = {}
    for rank, job in sample:
        pack = compute_pack(job.work_type, job.family, job.variant).to_dict()
        outcome = render_agentic_row(
            ScriptedTeacher(pack, rank, entries),
            {**pack, "id": "fixture", "verification": {}},
            rank=rank,
        )
        if outcome["row"] is None:
            raise AssertionError(
                f"the dummy flunked its own gate at rank {rank} "
                f"({job.work_type}/{job.family}/variant {job.variant}): "
                f"{outcome['dead_letter']['violations']}"
            )
        rows[rank] = outcome["row"]

    # -- second time through the real pair: what was asked must replay ------
    teacher = Teacher(FixtureTransport(entries=dict(entries)))
    for rank, job in sample:
        pack = compute_pack(job.work_type, job.family, job.variant).to_dict()
        outcome = render_agentic_row(
            teacher, {**pack, "id": "fixture", "verification": {}}, rank=rank
        )
        if outcome["row"] is None:
            raise AssertionError(
                f"replay dead-lettered rank {rank}: {outcome['dead_letter']['violations']}"
            )
        if json.dumps(rows[rank], sort_keys=True) != json.dumps(
            outcome["row"], sort_keys=True
        ):
            raise AssertionError(
                f"replay diverged at rank {rank}: the fixture does not recreate "
                "the conversation it captured"
            )

    # -- the mix, counted from what shipped, not from what was planned ------
    modes = [row["verification"]["render"]["mode"] for row in rows.values()]
    faults_seen = sorted(
        {
            row["verification"]["render"]["fault"]
            for row in rows.values()
            if row["verification"]["render"]["fault"]
        }
    )
    if (
        modes.count("no_call") != no_call
        or modes.count("faulted") != faulted
        or modes.count("clean") != limit - faulted - no_call
    ):
        raise AssertionError(
            f"the gate's mix came out {modes.count('clean')} clean, "
            f"{modes.count('faulted')} faulted, {modes.count('no_call')} no-call "
            f"-- required {limit - faulted - no_call}/{faulted}/{no_call}"
        )
    if len(faults_seen) != faulted:
        raise AssertionError(
            f"the faulted slice drew {faults_seen}, not {faulted} distinct faults "
            "-- the rotation has stalled on a repeat"
        )
    return {
        "entries": entries,
        "model": DUMMY_MODEL,
        "version": 1,
        "meta": {
            "entries": len(entries),
            "ranks_walked": ranks_walked,
            "slice_size": len(full_slice),
            "limit": limit,
            "faulted": faulted,
            "no_call": no_call,
            "faults": faults_seen,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(_HERE, AGENTIC_FIXTURE_NAME))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--faulted", type=int, default=DEFAULT_FAULTED)
    parser.add_argument("--no-call", dest="no_call", type=int, default=DEFAULT_NO_CALL)
    args = parser.parse_args(argv)
    payload = build_fixture(args.limit, args.faulted, args.no_call)
    with open(args.out, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    meta = payload["meta"]
    print(
        f"wrote {args.out}: {meta['entries']} replay entries for "
        f"{meta['limit']} conversations ({meta['no_call']} no-call, "
        f"{meta['faulted']} faulted: {', '.join(meta['faults'])}); "
        f"walked {meta['ranks_walked']} of {meta['slice_size']} agentic ranks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
