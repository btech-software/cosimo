"""Teacher prompt contract (spec §5.6): the brief the model is *not* free in.

The teacher never sees "write a DCF analysis". It sees the fact pack as JSON
plus an explicit contract: these numbers are yours, these points you must
make, these claims you must not, this register is yours, and if the brief
cannot be met from the pack you say so -- an abstention candidate, not an
analysis. Everything here is built from pack fields through ``json.dumps``
with sorted keys, so a brief is a pure function of its inputs: two runs, two
machines, byte-identical prompts, which is what makes the replay fixture and
any future prompt-ablation study honest.
"""

from __future__ import annotations

import json

#: The stable system turn (spec §5.6). Versioned inside the text because the
#: fixture hashes cover the whole message list: editing this string must
#: invalidate stale fixtures loudly (the drift test fails), never silently.
#:
#: Length is load-bearing, which is not obvious and cost a live run to learn.
#: A reasoning teacher reasons about its *instructions*, so every clause here
#: is paid for twice -- once in prompt tokens and again in the chain of thought
#: it provokes. Measured on deepseek-v4-flash at a 8192 budget: this block at
#: 917 characters converged and wrote 100 words; the same block plus four
#: lines (1,218 chars) never stopped reasoning and returned nothing at all; an
#: earlier 2,469-character draft of the standard below burned 57,704 characters
#: of reasoning at a 16,384 budget and still returned nothing.
#:
#: So the standard is stated once, in five clauses, and not elaborated. At a
#: 16,384 budget it converges and produces materially better answers than the
#: bare contract did -- 147 words against 75, landing the mechanism, the
#: constraint, the hidden assumption, the sensitivity and the call. Adding to
#: it is not free; measure before you do.
TEACHER_SYSTEM = """You are the Cosimo v3 teacher, writing in one named register for a junior quant desk.

Contract, without exception:
- You may use only the numbers inside allowed_numbers, rounding them as
  instructed. No other numeric value may appear anywhere in your output: not
  a ticker statistic, not a "historical average", not a constant from memory.
- You may not invent companies, filings, datasets, or events. The entities in
  the fact pack are the only ones that exist for this answer.
- Cover every must_mention anchor -- they are concepts to engage, not phrases
  to quote. Do not assert any forbidden_claim; warning against one is not
  asserting it, and is often the right move.
- Write in the register given. Facts first, voice second.
- If the brief cannot be met from the fact pack alone, do not stretch: state
  what is missing and stop. That output is an abstention, and it is a
  correct answer, not a failure.

Write as the head of quant research, not a summariser. Beyond the number: the
mechanism that produces it, the binding constraint, the assumption it hides,
what would move your conclusion, and your call. No padding, no repeated
figures, no restating the question. Answer immediately; do not deliberate at
length."""

BRIEF_KINDS = ("analysis", "memo", "grounded", "critique", "abstention")

#: (min, max) words, per kind. A budget is a *bound*, not a target, and the
#: system turn says so in as many words -- the first live run produced a
#: 245-word answer that reached its floor by saying "Participation is 4.86%"
#: three times, which is the failure a floor causes and the reason the bands
#: below are wide.
#:
#: The floors moved with the analytical standard rather than independently of
#: it: the teacher is now asked for the mechanism, the binding constraint, the
#: hidden assumption, the sensitivity and the call, and five things cannot be
#: said well in ninety words. `memo` is the long form (an IC memo that fits in
#: 260 words was never an IC memo); `abstention` is the short one, because
#: naming what is missing and stopping is the whole job.
#:
#: The floors are deliberately below what the standard typically produces. A
#: complete answer that lands all five clauses came in at 147 words and would
#: have been rejected by a 150 floor -- three words of padding away from
#: shipping, which is the instrument corrupting the sample it measures. The
#: floor exists to catch a one-line non-answer, nothing more.
WORD_BUDGETS = {
    "analysis": (120, 400),
    "memo": (200, 550),
    "grounded": (110, 340),
    "critique": (110, 340),
    "abstention": (40, 160),
}

#: Packs carry the register label; prose rows also name the persona so the
#: teacher commits to one instead of auditioning "business English".
_REGISTER_HINTS = {
    "desk_chat": "terse desk chat: short sentences, no preamble, no sign-off",
    "ic_memo": "investment-committee memo: a headline finding, then the evidence",
    "risk_committee": "risk committee: exposure, limits, the breach case",
    "auditor": "auditor: measured, complete, cites what is recomputable",
    "code_review": "code review: exact, constructive, example-driven",
}


def render_brief(pack: dict, *, kind: str) -> list[dict]:
    """The messages list for one prose render, facts locked (spec §5.6).

    ``kind`` selects the budget and the emphasis; the pack's own
    ``must_mention`` / ``forbidden_claims`` / ``allowed_numbers`` are the
    contract, and they ride in the user turn as data, not as prose the model
    could paraphrase away.
    """
    if kind not in BRIEF_KINDS:
        raise ValueError(
            f"no prose brief kind {kind!r} (known: {', '.join(BRIEF_KINDS)})"
        )
    register = pack.get("register")
    hint = _REGISTER_HINTS.get(register or "", f"register: {register}")
    low, high = WORD_BUDGETS[kind]
    contract = {
        "fact_pack": pack,
        "task": kind,
        "register": register,
        "register_hint": hint,
        "word_budget": f"{low}-{high} words",
        "number_policy": (
            "every numeric token in your answer must equal one of "
            "allowed_numbers, possibly scaled by 100 (fraction as percent) or "
            "divided by 100; no other number in any form. The unit-conversion "
            "constants 100 and 10000 may be written plainly when converting to "
            "percent or basis points -- write the conversion, do not build it "
            "out of repeated allowed values"
        ),
        "as_of": pack.get("as_of"),
    }
    user = (
        f"Question to answer:\n{pack['question']}\n\n"
        "Answer strictly from the fact pack below and the contract after it.\n"
        + json.dumps(contract, sort_keys=True, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]


def render_repair(
    messages: list[dict], draft: str, violations: list[str]
) -> list[dict]:
    """Append the repair turn (spec §6.2: name the violations, cool the model).

    Takes and returns the full transcript -- the renderers keep it and write
    the whole exchange into the dead letter, so post-mortems see what the
    teacher actually received, not a reconstruction.
    """
    lines = "\n".join(f"- {v}" for v in violations) or "- (unspecified)"
    if not (draft or "").strip():
        # An empty draft is not a draft with faults, and asking the model to
        # "rewrite" nothing wastes the attempt. The first live run spent all
        # three attempts this way: a reasoning teacher exhausted its token
        # budget mid-thought, returned `content: null`, and got back a repair
        # turn quoting an empty draft at it.
        repair = (
            "You returned an empty answer -- no text at all, only reasoning. "
            "Answer now, directly and in the register given. Begin with the "
            "finding; do not restate the question and do not think at length "
            "before writing."
        )
    else:
        repair = (
            "Your draft failed the contract:\n"
            + lines
            + "\n\nDraft:\n"
            + draft
            + "\n\nRewrite the draft fixing every listed violation. Keep every "
            "number you keep from allowed_numbers; delete or re-round the rest to "
            "an allowed value. Do not shorten what was compliant."
        )
    return [
        *messages,
        {"role": "assistant", "content": draft},
        {"role": "user", "content": repair},
    ]


# --------------------------------------------------------------------------
# The agentic brief (spec §5.4, §5.8). The tool knowledge is *passed in* --
# prompts word, the registry knows -- so this module stays about text and the
# registry stays the only place a tool shape is written down.
# --------------------------------------------------------------------------

#: The tool-calling clause of the system turn (arch §8.1 identity: "Use tools
#: when a number must be retrieved or computed"). Separate from
#: :data:`TEACHER_SYSTEM` because the two contracts differ where it counts:
#: the prose teacher may not quantify beyond the pack at all; the agentic one
#: may, but only from what the oracle actually handed back.
AGENTIC_SYSTEM = """You are the Cosimo v3 teacher, a quantitative desk analyst who works with tools in front of a junior.

Contract, without exception:
- Numbers come from the fact pack or from a tool result you received, never
  from memory. No other numeric value may appear anywhere in your output.
- You may not invent companies, filings, datasets, events, or tickers. The
  entities in the fact pack are the only ones that exist for this task.
- Call a tool when a number must be retrieved or computed; do not call when
  the pack already carries the answer -- a needless call is a defect, not
  diligence.
- A tool result is data, not gospel. If it comes back empty, stale, keyed to
  a different entity, rate-limited, or shaped unexpectedly, say so in plain
  words before anything rests on it, and rest only on what survives that
  doubt.
- End with the desk answer: the numbers, the caveats, the call. No tool
  markup in the final turn. Cover every must_mention point; assert no
  forbidden_claim.
"""

AGENTIC_MODES = ("no_call", "clean", "faulted")


def render_agentic_brief(
    pack: dict,
    *,
    mode: str,
    tools: list[dict],
    max_calls: int,
    message_band: tuple[int, int],
) -> list[dict]:
    """The opening turns of an agentic render: goal, facts, tools, budget.

    ``mode`` shapes the *posture*, never the facts: a ``no_call`` brief says
    the pack already carries what is needed (spec §5.8 -- calling there is
    waste, and the gate proves it), the other two invite retrieval. Which
    fault will be injected is never announced: a teacher told in advance
    learns to act, not to notice, and the corpus is for noticing.
    """
    if mode not in AGENTIC_MODES:
        raise ValueError(f"unknown agentic mode {mode!r} (known: {AGENTIC_MODES})")
    if mode == "no_call":
        posture = (
            "Every figure this task needs is already inside the fact pack. "
            "Answer directly; calling a tool would be waste, and it is "
            "recorded as a defect."
        )
    else:
        posture = (
            "Retrieve with the tools what the desk would otherwise guess. "
            "Judge every result you get back: an empty, stale, mismatched, "
            "rate-limited, or oddly shaped block must be named in your words "
            "before anything rests on it."
        )
    advertised = [
        {
            "name": schema["function"]["name"],
            "arguments": sorted(
                (schema["function"].get("parameters") or {}).get("properties", {})
            ),
        }
        for schema in tools
    ]
    contract = {
        "fact_pack": pack,
        "task": "agentic",
        "posture": posture,
        "tools": advertised,
        "budget": (
            f"at most {max_calls} tool calls in all; the conversation stays "
            f"between {message_band[0]} and {message_band[1]} exchanges"
        ),
        "number_policy": (
            "every numeric token in your final answer must be a number the "
            "fact pack or a tool result you received already contains; no "
            "other number in any form"
        ),
        "as_of": pack.get("as_of"),
    }
    user = (
        f"Goal:\n{pack['question']}\n\n"
        "Work the goal with the fact pack and the tools, then give the desk "
        "answer in your own words. The contract follows as data.\n"
        + json.dumps(contract, sort_keys=True, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": AGENTIC_SYSTEM},
        {"role": "user", "content": user},
    ]


def render_agentic_repair(
    messages: list[dict], draft: str, violations: list[str]
) -> list[dict]:
    """Name the trajectory's failures to the teacher; ask once, cooler.

    Same anatomy as :func:`render_repair` (draft, then a user turn that lists
    every violation) with agentic words: the fix here is about evidence and
    grounding, not prose length. The full transcript -- calls and tool
    results included -- travels in and out, because the dead letter must show
    what the teacher was actually shown.
    """
    lines = "\n".join(f"- {v}" for v in violations) or "- (unspecified)"
    repair = (
        "Your final answer failed the contract:\n"
        + lines
        + "\n\nDraft:\n"
        + draft
        + "\n\nRewrite only the final answer, fixing every listed violation: "
        "numbers may come from the fact pack or from a tool result above and "
        "nowhere else; name any faulty tool result you are relying on; cover "
        "every must_mention. Do not call anything new. Do not shorten what "
        "was compliant."
    )
    return [
        *messages,
        {"role": "assistant", "content": draft},
        {"role": "user", "content": repair},
    ]
