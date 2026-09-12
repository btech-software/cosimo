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

from .. import config

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

#: What each record type is *for*, in the second person. The register owns the
#: voice; this owns the move -- what the row does with the pack, and where it
#: starts.
#:
#: It exists because there was nothing. Diffing an ``analysis`` brief against a
#: ``grounded`` brief for one pack produced exactly two changes: the literal
#: ``"task"`` string, and a word budget whose bands overlap (120-400 against
#: 110-340). The docstring on :func:`render_brief` claimed "``kind`` selects
#: the budget and the emphasis"; there was no emphasis. Five record types were
#: one instruction wearing five names, and in a live nine-row sample four of
#: nine sibling pairs opened on the same four words -- the pack's headline
#: figure -- while two non-``memo`` rows gave themselves a memo title.
#:
#: Every clause here has to compose with *any* register, because one pack
#: serves all of them and the register is the family's property. So these say
#: what to do and never how to format: headings, labelled calls and length are
#: :data:`verification.register._REGISTER_SHAPE`'s to rule on, and a kind rule
#: that reached into them would recreate the contradiction this session spent
#: three rounds removing -- a teacher told to write a call by one line of its
#: brief and not to by the next has been given no brief at all.
_KIND_SHAPE = {
    "analysis": (
        "argue, do not summarise. Open on the mechanism or the binding "
        "constraint, not on the headline figure -- the reader can see the "
        "number. Name the assumption it hides and what would change your "
        "conclusion"
    ),
    "memo": (
        "write it up for a decision: the finding first, the evidence that "
        "carries it, then what you would do"
    ),
    "grounded": (
        "answer the question and stay on it. Every claim you make should be "
        "traceable to a figure in the fact pack; do not open with a "
        "write-up's framing and do not range beyond what was asked"
    ),
    "critique": (
        "judge the claim you are given: say what is wrong with it, why it is "
        "wrong, and what would have to be true for it to hold"
    ),
    "abstention": (
        "say precisely what the fact pack does not contain and stop. Do not "
        "estimate around the gap or answer a nearby question instead"
    ),
}


def kind_shape(kind: str) -> str:
    """The move :data:`_KIND_SHAPE` asks of *kind*, or ``""`` if none."""
    return _KIND_SHAPE.get(kind, "")


#: Packs carry the register label; prose rows also name the persona so the
#: teacher commits to one instead of auditioning "business English".
_REGISTER_HINTS = {
    "desk_chat": "terse desk chat: short sentences, no preamble, no sign-off",
    "ic_memo": "investment-committee memo: a headline finding, then the evidence",
    "risk_committee": "risk committee: exposure, limits, the breach case",
    "auditor": "auditor: measured, complete, cites what is recomputable",
    "code_review": "code review: exact, constructive, example-driven",
}


def desk_figures(node):
    """*node* with every float rounded to the depth a desk would print.

    The teacher quotes what the brief shows it, so the brief is where rounding
    has to happen. ``assemble_numbers`` rounds to 12 dp -- a deliberate choice,
    but a noise-suppression one ("0.30000000000000004 must not appear twice"),
    never a decision to publish twelve significant decimals. The consequence only
    surfaced with a teacher good enough to quote the pack faithfully: the first
    qwen3.8-flash-next run wrote a portfolio weight as ``0.472041725693``,
    straight from ``allowed_numbers``, and no reviewer would accept it.

    Applied to the brief's copy and nowhere else, because the stored pack's
    ``computed`` figures are load-bearing at full precision: the exam answer is
    ``computed[answer_key]`` verbatim, the oracle returns them as tool results,
    and ``verification.implementation`` pins them into generated hidden tests
    that compare within 1e-9 relative. Rounding there would quietly break every
    one of those. Rounding here changes only what the model reads, and the
    invented-number gate still admits the rounded spelling: its tolerance is
    0.5%, and six decimals on any figure in the corpus is far inside that.
    """
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        return round(node, config.PROSE_MAX_DECIMALS)
    if isinstance(node, dict):
        return {key: desk_figures(value) for key, value in node.items()}
    if isinstance(node, list):
        rounded = [desk_figures(value) for value in node]
        # allowed_numbers is a sorted set; rounding can collide two spellings of
        # the same economic figure, and two identical entries would read as a
        # contradiction in a list the model is told is exhaustive.
        if rounded and all(isinstance(v, (int, float)) for v in rounded):
            return sorted({float(v) for v in rounded})
        return rounded
    if isinstance(node, tuple):
        return tuple(desk_figures(value) for value in node)
    return node


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
    # Imported here, not at module scope: ``verification.register`` reads
    # ``WORD_BUDGETS`` from this module to size the desk_chat ceiling, so
    # the two are mutually dependent by construction -- the budget is the
    # prompt's to state and the ceiling is the gate's to enforce.
    from ..verification.register import register_shape

    register = pack.get("register")
    hint = _REGISTER_HINTS.get(register or "", f"register: {register}")
    low, high = WORD_BUDGETS[kind]
    contract = {
        "fact_pack": desk_figures(pack),
        "task": kind,
        "register": register,
        "register_hint": hint,
        # The register *gate*, in its own words, beside the hint that describes
        # the voice. The hint says what the register sounds like; this says
        # what it will be refused for -- which the brief had never carried, so
        # every shape rule was a rule the teacher could only discover by
        # failing it and paying for a retry.
        "register_rules": register_shape(register or "", kind),
        # What this record type does with the pack. Beside the register rules
        # rather than merged into them: the register is the family's and the
        # task is the row's, and one pack serves every record type.
        "task_rules": kind_shape(kind),
        "word_budget": f"{low}-{high} words",
        "number_policy": (
            "every numeric token in your answer must equal one of "
            "allowed_numbers, possibly scaled by 100 (fraction as percent) or "
            "divided by 100; no other number in any form. The unit-conversion "
            "constants 100 and 10000 may be written plainly when converting to "
            "percent or basis points -- write the conversion, do not build it "
            "out of repeated allowed values. Do no arithmetic: no totals, no "
            "differences, no ratios, no sum checks. If the parts of a "
            "decomposition are given, quote them and say what they mean; "
            "adding them up produces a number the pack does not contain. And "
            "do not re-round: write each figure at the precision it is given"
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
        # "Do not shorten what was compliant" is right for a numbers or an
        # anchor violation and exactly wrong for a length one: a draft told to
        # cut sentences and to keep its length has been given two instructions
        # it cannot both obey, and the first row of the corrected render spent
        # all three attempts at 13 sentences against a ceiling of 12. So the
        # clause is dropped when the fault *is* the length.
        from ..verification.register import is_length_violation

        too_long = any(is_length_violation(v) for v in violations)
        keep = "" if too_long else " Do not shorten what was compliant."
        repair = (
            "Your draft failed the contract:\n"
            + lines
            + "\n\nDraft:\n"
            + draft
            + "\n\nRewrite the draft fixing every listed violation. Keep every "
            "number you keep from allowed_numbers; delete or re-round the rest to "
            "an allowed value." + keep
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
