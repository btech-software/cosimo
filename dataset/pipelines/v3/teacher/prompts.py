"""Teacher prompt contract (spec §5.6): the brief the model is *not* free in.

The teacher never sees "write a DCF analysis". It sees the fact pack as JSON
plus an explicit contract: these numbers are yours, these points you must
make, these claims you must not, this register is yours, and if the brief
cannot be met from the pack you say so -- an abstention candidate, not an
analysis. Everything here is built from pack fields through ``json.dumps``
with sorted keys, so a brief is a pure function of its pack: two runs, two
machines, byte-identical prompts, which is what makes the replay fixture and
any future prompt-ablation study honest.
"""

from __future__ import annotations

import json

#: The stable system turn (spec §5.6). Versioned inside the text because the
#: fixture hashes cover the whole message list: editing this string must
#: invalidate stale fixtures loudly (the drift test fails), never silently.
TEACHER_SYSTEM = """You are the Cosimo v3 teacher, writing in one named register for a junior quant desk.

Contract, without exception:
- You may use only the numbers inside allowed_numbers, rounding them as
  instructed. No other numeric value may appear anywhere in your output: not
  a ticker statistic, not a "historical average", not a constant from memory.
- You may not invent companies, filings, datasets, or events. The entities in
  the fact pack are the only ones that exist for this answer.
- Cover every must_mention point. Do not assert any forbidden_claim, even as
  a hedge or a disclaimer.
- Write in the register given. Facts first, voice second; the numbers are
  already computed, your job is the desk's voice and the reasoning's shape.
- If the brief cannot be met from the fact pack alone, do not stretch: state
  what is missing and stop. That output is an abstention, and it is a
  correct answer, not a failure.
"""

_BRIEF_KINDS = ("analysis", "memo", "grounded", "critique", "abstention")

#: (min, max) words, per kind. A budget is not a stricture -- the verifier
#: measures length; the budget in the prompt is what stops a 4B teacher from
#: rambling into padding that the repair loop would then have to cut.
_WORD_BUDGETS = {
    "analysis": (160, 320),
    "memo": (120, 260),
    "grounded": (140, 280),
    "critique": (90, 200),
    "abstention": (50, 140),
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
    if kind not in _BRIEF_KINDS:
        raise ValueError(
            f"no prose brief kind {kind!r} (known: {', '.join(_BRIEF_KINDS)})"
        )
    register = pack.get("register")
    hint = _REGISTER_HINTS.get(register or "", f"register: {register}")
    low, high = _WORD_BUDGETS[kind]
    contract = {
        "fact_pack": pack,
        "task": kind,
        "register": register,
        "register_hint": hint,
        "word_budget": f"{low}-{high} words",
        "number_policy": (
            "every numeric token in your answer must equal one of "
            "allowed_numbers, possibly scaled by 100 (fraction as percent) or "
            "divided by 100; no other number in any form"
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
