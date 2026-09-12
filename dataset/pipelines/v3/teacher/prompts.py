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
#: Five lines, and the shortening is the amendment's §B.1. What was here
#: before asked for a junior-desk register *and* the voice of the head of quant
#: research, and then, in one sentence, for the mechanism, the binding
#: constraint, the hidden assumption, what would move the conclusion and the
#: call -- on every row of every kind. That list is what produced
#: ``live_three_registers.jsonl``: five beats, five paragraphs, the same first
#: sentence for ``analysis`` and ``grounded``, and a 288-word desk note that
#: said the schedule both was and was not the risk. It was the exam liturgy
#: V3.1 deleted ``FINAL ANSWER:`` to escape, grown back in prose.
#:
#: What replaces it is the part that is true of every row -- the facts are the
#: pack's, the entities are the pack's, an unanswerable brief is an abstention,
#: and the machinery is never mentioned. *What this row is for* moved to the
#: user turn, where it can differ by kind, register and work type, which is
#: what those three fields are.
#:
#: Length is still load-bearing, which is not obvious and cost a live run to
#: learn: a reasoning teacher reasons about its instructions, so every clause
#: here is paid for twice. Adding to it is not free; measure before you do.
#:
#: The opening words are not free either: :data:`config.TEACHER_FINGERPRINT`
#: must appear here verbatim. It is what ``row.carries_teacher_brief`` and the
#: harness's prepare gate match on to prove no student row carries the factory,
#: and a system turn that stopped containing it would turn both gates green by
#: removing what they look for.
TEACHER_SYSTEM = """You are the Cosimo v3 teacher. You write student-facing desk answers.

Use only the figures the fact pack gives you, in their display form when one
exists; do no arithmetic of your own and invent no entity, print or event.
If the pack does not support the question, say what is missing and stop --
that answer is an abstention, and it is correct, not a failure.
Never mention the fact pack, the contract, the gate, or these instructions."""

BRIEF_KINDS = ("analysis", "memo", "grounded", "critique", "abstention")

#: ``kind -> (min, max)`` words. A budget is a *bound*, not a target.
#:
#: These are the amendment's §B.2 caps, and they are less than half what they
#: were (analysis 120-400, memo 200-550, grounded 110-340). The old bands did
#: not merely permit the 288-word desk note -- with a 120-word floor and five
#: required beats they *commissioned* it, and they left ``analysis`` and
#: ``grounded`` overlapping across almost their whole range, which is how two
#: record types came to be one instruction wearing two names.
#:
#: Floors stay deliberately low. They exist to catch a one-line non-answer and
#: nothing more: a floor is the instrument corrupting the sample it measures
#: the moment it is high enough to be reached by repeating a figure.
WORD_BUDGETS = {
    "analysis": (60, 220),
    "memo": (100, 300),
    # 90, not 140: on the first v3.2 capture a TCA grounded row came in at 80
    # words against an 81-word analysis of the same pack -- not shorter, only
    # reordered, which is the collapse §B.2 split the kinds to prevent. A
    # citation that cannot be told from an argument by its length is not yet a
    # second record type.
    "grounded": (40, 90),
    "critique": (50, 180),
    "abstention": (25, 120),
}

#: Where a register is tighter than its kind. §B.2 gives the desk a 160-word
#: analysis against the committee's 220, because those are different documents
#: and the whole point of splitting kind from register is that each can bind.
#: One entry, because one is what the amendment states; the shape is here so
#: the next one is a line rather than a refactor.
_REGISTER_WORD_CAPS = {
    ("analysis", "desk_chat"): 160,
}

#: Sentence ceilings that belong to the *kind* rather than to the register.
#: ``grounded`` is a citation, ``abstention`` is a refusal, and both are short
#: by construction in a way an ic_memo analysis is not. The desk_chat ceiling
#: (:func:`verification.register.desk_chat_ceiling`) is separate and both
#: apply: the tighter one wins, as it should.
KIND_SENTENCE_CAPS = {
    "grounded": 8,
    "abstention": 5,
}


def word_budget(kind: str, register: str = "") -> tuple[int, int]:
    """``(min, max)`` words for this kind in this register.

    One accessor, because the gate, the brief and the fixture harness must all
    read the same band -- a row marked against a band it was never shown is the
    failure ``_REGISTER_SHAPE`` was written to end, and a second copy of the
    table here would reintroduce it on the length axis.
    """
    low, high = WORD_BUDGETS[kind]
    return low, min(high, _REGISTER_WORD_CAPS.get((kind, register), high))


#: What each record type *is*, in the second person: the job, its order, and
#: what it may not do. §B.2 verbatim in substance.
#:
#: These are no longer one clause apiece. The previous table said "argue, do
#: not summarise" and left the shape of an argument to a system turn that
#: demanded five beats of it, so every kind produced the same five-beat essay
#: at a different length. A kind is a *job*: the answer's first sentence, what
#: carries it, what would overturn it, and whether it ends in a decision at
#: all. Those differ between an analysis, a citation and a memo, and nothing
#: else in the brief can say so.
#:
#: They still may not legislate what the register owns -- headings, a labelled
#: call, sentence ceilings -- *except* where the amendment moved that rule onto
#: the kind on purpose (``grounded`` carries no headings in any register, and a
#: ``memo``'s headings are the register's list). The pairing is checked:
#: ``memo`` x ``desk_chat`` is no longer a legal row (inventory §C), which is
#: what makes "use the headings your register allows" a sentence with an
#: answer.
_KIND_SHAPE = {
    "analysis": (
        "answer the question and stop. In this order: (1) the first sentence "
        "gives your reading of the situation and the number that forces it, in "
        "one breath -- a judgement with a figure in it, never the quantity's "
        "name restated; (2) one sentence on the "
        "mechanism that produces that number, using a figure from the pack; "
        "(3) one constraint or assumption that would change it; (4) a call "
        "consistent with (1)-(3) -- if they conflict, say so and prefer the "
        "number. Do not restate the inputs, do not tour the pack, and do not "
        "write the words 'binding constraint' unless your register is "
        "risk_committee and you mean a limit"
    ),
    "memo": (
        "write a scannable document, not a longer analysis. Use the headings "
        "your register allows: ic_memo takes Finding / Evidence / Call, "
        "risk_committee takes Exposure / Assumption / Limits / Breach case. "
        "The finding is one sentence; the evidence is the decomposition; the "
        "call is one sentence and may not contradict the finding. Paste no "
        "formulas. Where the pack states a reconciling residual, that figure "
        "is the reconciliation -- quote it rather than adding the pieces up "
        "yourself. Only when the pack's own residual is material does the "
        "finding become that the figures do not reconcile, with abstention as "
        "the call; never invent a reason they do"
    ),
    "grounded": (
        "cite, do not argue twice. Open by naming the quantity the question "
        "asks for exactly as the question names it, with its figure and unit, "
        "and judge nothing in that sentence -- an analysis of this pack opens "
        "on a reading, and this row must not. Then at most three more "
        "figures, each tied to one phrase of the question, and one caveat. Fold "
        "the points you must engage into the sentences that carry their "
        "figures -- name the mechanism beside the number it produces -- rather "
        "than arguing them separately. No call unless the question asked for "
        "one, and no section labels of any kind"
    ),
    "critique": (
        "name one defect in the draft you are given: what is wrong, why it is "
        "wrong, and what would have to be true for it to hold. Do not rewrite "
        "the passage and do not list every flaw you can see -- one, judged"
    ),
    "abstention": (
        "say precisely what the fact pack does not contain, and stop. Name the "
        "missing quantity, not a nearby one; offer no substitute figure, no "
        "estimate and no answer to a question that was not asked"
    ),
}


def kind_shape(kind: str, register: str = "") -> str:
    """The job :data:`_KIND_SHAPE` asks of *kind*, with its caps stated.

    The caps ride here rather than in a field of their own because they are
    part of the job: "answer the question and stop" and "at most 160 words" are
    the same instruction said twice, and a teacher that reads them in two
    places obeys the one it read last.
    """
    shape = _KIND_SHAPE.get(kind, "")
    if not shape:
        return ""
    low, high = word_budget(kind, register)
    clauses = [shape, f"keep it between {low} and {high} words"]
    cap = KIND_SENTENCE_CAPS.get(kind)
    if cap:
        clauses.append(f"and to at most {cap} sentences")
    return ", ".join(clauses[:2]) + ("" if not cap else " " + clauses[2])


#: What each work type's own arithmetic makes wrong, appended to the user turn
#: and never added to the system one (§B.4). Every line here is a mistake a
#: live row actually made: the desk note that called a 4.86% clip a pacing
#: problem, the attribution memo that explained a sign flip with a story about
#: k, the risk paper that read a negative daily mean as ten million dollars of
#: expected gain, and the valuation note that recommended owning "near this EV".
_WORK_TYPE_RULES = {
    "execution.tca.arrival": (
        "Participation is shares over ADV; compare it with the cap in the "
        "pack. Below the cap the impact is the cost and the schedule is not "
        "the risk -- this is an impact bill, not a pacing problem. The "
        "half-spread is not the whole cost. Say which benchmark the shortfall "
        "is measured against: arrival is this pack's, and a decision price is "
        "not in it -- name that absence rather than pricing against one, "
        "because a shortfall quoted without its benchmark is a number without "
        "a meaning."
    ),
    "portfolio.attribution.brinson_carino": (
        "Report allocation, selection and interaction against the active "
        "return. The pack carries reconciling_residual_bps: at or near zero it "
        "says the pieces do add to the active number, and you may say so "
        "without adding them yourself. If that residual is material, abstain "
        "on which effect to act on rather than explaining the gap. Do not "
        "account for a sign with a "
        "story about the Carino factor unless k is in the pack and the signed "
        "effect matches the sign of the raw (wp - wb)(rb - Rb). The effect "
        "worth acting on is the largest absolute total that is also a decision "
        "-- weights against names -- and if they are all small, say so."
    ),
    "risk.market.var_es": (
        "A negative daily mean is a negative drift: it adds to the expected "
        "loss. It is never a gain. VaR is a quantile; expected shortfall is "
        "the mean of the tail beyond it. Do not recommend a trade."
    ),
    "valuation.equity.dcf": (
        "Enterprise value is not a share price. If the price or the share "
        "count is missing, the call is that there is no ownership call -- not "
        "that the name is worth owning near this EV."
    ),
    "valuation.equity.multiples": (
        "Enterprise value is not a share price. If the price or the share "
        "count is missing, the call is that there is no ownership call -- not "
        "that the name is worth owning near this EV."
    ),
}


def work_type_rules(work_type: str) -> str:
    """The addendum for *work_type*, or ``""`` where none is written.

    Empty rather than generic: a work type with nothing specific to say gets
    nothing, because a filler sentence in this slot is a sentence the teacher
    reasons about for no gain.
    """
    return _WORK_TYPE_RULES.get(str(work_type or ""), "")


#: The coverage rule, stated once and in the user turn (§B.1 keeps it out of
#: the system prompt). It has to be stated *somewhere*, and for four of the
#: five kinds it briefly was not: the amendment deleted "cover every
#: must_mention" from the system turn and only the `grounded` brief said it
#: again, while the gate went on refusing every kind for it. Measured on the
#: think-off arm of the §B bake-off: 20 of 20 first-pass drafts missed at least
#: one anchor, which is the most-failed rule in the corpus and was the one rule
#: the writer could not read.
POINTS_POLICY = (
    "the points listed in must_mention are the ones this answer has to engage "
    "-- make each of them in your own words, at the place in the argument "
    "where it belongs, using every content word of the point at least once "
    "(the check is mechanical: 'participation against ADV' needs both "
    "'participation' and 'ADV' to appear). They are concepts to engage, not "
    "phrases to quote, and a point you cannot make from the pack's figures is "
    "a point to say you cannot make. Assert nothing in forbidden_claims; "
    "warning against one is not asserting it"
)


def number_policy(pack: dict) -> str:
    """How figures are spelled, with this pack's own display forms quoted.

    §A.3. ``display`` and ``desk_figures`` both existed before the amendment
    and neither reached the teacher as an instruction, so live rows wrote
    ``359667.31`` for a dollar cost and ``0.048555`` for a participation the
    pack spells ``4.86%``. The gate already refuses those (``overprecise_numbers``,
    ``integer_format_offenders``); stating the rule in the brief is what makes
    the refusal a standard rather than a trap.
    """
    display = pack.get("display") or {}
    spellings = "; ".join(
        f"{key} is written {value}" for key, value in sorted(display.items())
    )
    lines = [
        "every numeric token in your answer must be one the fact pack "
        "contains, possibly scaled by 100 (a fraction written as a percent) "
        "or divided by 100; no other number in any form. The conversion "
        "constants 100 and 10000 may be written plainly",
        "do no arithmetic: no totals, no differences, no ratios, no sum "
        "checks. Where a decomposition is given, quote the pieces and say "
        "what they mean -- adding them produces a number the pack does not "
        "contain",
        "write each figure at the precision the pack gives it, never deeper, "
        "and never re-round it to a whole number the question happens to print",
        "a count is written as a count (430,567 or 430567, never 430567.0); "
        "money to whole units unless the pack says otherwise; a rate already "
        "in percent or basis points stays in them (4.86%, 28.16 bp -- never "
        "0.048555)",
    ]
    if spellings:
        lines.append(f"use this pack's own spellings where it gives them: {spellings}")
    return "; ".join(lines)


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

    Four instructions compose the user turn, and they are four because they
    answer four different questions (§B.2):

    * **kind** -- the job. What this row is for, in what order, and where it
      stops. :data:`_KIND_SHAPE`.
    * **register** -- the shape, in the gate's own words
      (:func:`verification.register.register_shape`), so the brief and the
      refusal cannot disagree.
    * **work type** -- what this arithmetic makes wrong. :data:`_WORK_TYPE_RULES`.
    * **number policy** -- how a figure is spelled, quoting this pack's own
      display forms.

    Before the amendment there was one: the kind selected a word budget whose
    bands overlapped, and everything else came from a system turn that asked
    every row for the same five beats. Two record types differing by a literal
    string and thirty words of budget are one record type with two names, and
    the live sample read like it.

    The pack's own ``must_mention`` / ``forbidden_claims`` / ``allowed_numbers``
    ride in the user turn as data, not as prose the model could paraphrase away.
    """
    if kind not in BRIEF_KINDS:
        raise ValueError(
            f"no prose brief kind {kind!r} (known: {', '.join(BRIEF_KINDS)})"
        )
    # Imported here, not at module scope: ``verification.register`` reads this
    # module's budgets to size the desk_chat ceiling, so the two are mutually
    # dependent by construction -- the budget is the prompt's to state and the
    # ceiling is the gate's to enforce.
    from ..verification.register import register_shape

    register = pack.get("register") or ""
    low, high = word_budget(kind, register)
    contract = {
        "fact_pack": desk_figures(pack),
        "task": kind,
        "register": register,
        # The register *gate*, in its own words. The one-line voice hints this
        # replaced ("terse desk chat", "headline then evidence") described a
        # sound and refused nothing, so every shape rule was one the model
        # could only discover by failing it and paying for a retry.
        "register_rules": register_shape(register, kind),
        # What this record type does with the pack, and how much of it. Beside
        # the register rules rather than merged into them: the register is the
        # family's and the job is the row's.
        "task_rules": kind_shape(kind, register),
        # What this work type's own numbers make wrong. Selected by work type
        # and never added to the system turn, which is what keeps four
        # arithmetic disciplines from being paid for on every row of all five.
        "work_type_rules": work_type_rules(pack.get("work_type")),
        "word_budget": f"{low}-{high} words",
        "points_policy": POINTS_POLICY,
        "number_policy": number_policy(pack),
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
            "an allowed value, and write a figure the way the pack spells it "
            "where it gives a spelling. Do not name the violations, their tags, "
            "or this instruction in the answer." + keep
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
