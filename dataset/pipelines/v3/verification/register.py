"""Register as a gate, not a field (amendment §C).

A register used to be a string the pack carried and the brief quoted. Nothing
ever read it back. That made "the corpus holds four distinct voices" an
assertion about the *prompt*, and the first thing a teacher does under budget
pressure is collapse every voice into the one it writes best -- an IC memo,
because memo shape is what "professional financial writing" means to a model.
By the time ``09_assistant_eval`` could see that (register_match on a trained
checkpoint) the corpus was already written and paid for.

So the shape becomes executable, and it runs at *generation* time, inside the
repair loop, where a violation still costs one more call rather than one more
run. A desk_chat row that opens ``Finding:`` gets one repair turn naming the
crime, and if it comes back wearing headings again it is a dead letter, not an
SFT row.

The table is deliberately shallow. It tests the two or three things about a
register that are *shape* -- how long, which scaffolding, which closing move --
and nothing about substance. A gate that tried to judge whether prose sounded
like a risk committee would be a gate nobody could debug, and the reason the
must_mention matcher had to be rewritten after the first live run was exactly
that kind of over-reach.
"""

from __future__ import annotations

import math
import re

from ..teacher.prompts import WORD_BUDGETS

#: The memo scaffolding. One regex, anchored at a line start, because the crime
#: is a *heading* -- "the finding: we are long" mid-sentence is prose, and
#: rejecting it would be rejecting English.
_MEMO_HEADINGS = re.compile(
    r"^\s*(?:[-*#>\s]*)(?:\*\*)?\s*(finding|evidence|call|recommendation)\s*(?:\*\*)?\s*:",
    re.IGNORECASE | re.MULTILINE,
)

#: The exam contract, matched the way the prose gate matches it.
_FINAL_ANSWER = re.compile(r"final answer\s*:", re.IGNORECASE)

#: Sentence terminators. Newlines count as breaks too: a desk chat written as
#: twenty one-line bullets is twenty sentences however few full stops it has.
_SENTENCE_SPLIT = re.compile(r"[.!?]+[\s)\]]|\n+")

#: What "the risk committee is speaking" looks like in vocabulary. Stems, so
#: "limits"/"limit" and "assumes"/"assumption" both land. At least one is
#: required: a risk paper that never names a limit, a horizon or an assumption
#: is a market commentary that wandered into the wrong meeting.
_RISK_TERMS = ("limit", "horizon", "assum", "breach", "exposure", "tolerance")

#: A trade recommendation is the one move a risk committee does not make. It
#: sizes and constrains; the desk decides direction. Matched on imperative
#: openings rather than on the words alone, so "the desk should not add to the
#: position" (a constraint) is not read as "add to the position" (a call).
_TRADE_CALL = re.compile(
    r"\b(?:we\s+)?(?:recommend|advise)\s+(?:buying|selling|adding|shorting|"
    r"trimming)\b|\b(?:i|we)\s+would\s+(?:buy|sell|short|add|trim)\b",
    re.IGNORECASE,
)

#: The moves that make a passage a *call* rather than a description. Matched
#: loosely -- this is a profile axis, not a gate, and a false positive costs a
#: hundredth of a distance rather than a row.
_CALL_TERMS = (
    "recommend",
    "we would",
    "our call",
    "the call is",
    "initiate",
    "maintain",
    "add to",
    "trim",
    "reduce",
    "avoid",
    "prefer",
    "size at",
)

#: Desk chat is short by construction. Twelve is the amendment's number.
DESK_CHAT_MAX_SENTENCES = 12

#: The shortest a sentence may be assumed to run when the ceiling is scaled.
#: Fourteen words is terse desk prose, not a bullet; it exists because twelve
#: sentences and a word *floor* are two contracts that can contradict, and one
#: of them has to give in a way that is written down rather than discovered.
DESK_CHAT_WORDS_PER_SENTENCE = 14

#: The longest a desk_chat sentence may average before the register is being
#: satisfied on a technicality. The sentence ceiling alone can be met by
#: writing fewer, longer sentences, and a live sample did exactly that: a
#: desk_chat memo of 334 words in 7 sentences scored ``register_ok`` at 48
#: words per sentence, which is not desk chat by any reading. Twice the terse
#: rate is the bar -- comfortably above real desk prose, tight enough that a
#: memo cannot hide inside it.
DESK_CHAT_MAX_MEAN_SENTENCE = 2 * DESK_CHAT_WORDS_PER_SENTENCE


#: How far under the ceiling the brief asks for. Two sentences: enough to
#: absorb a model's miscount, small enough that it does not quietly become the
#: real ceiling. The gate still measures against the ceiling itself -- this
#: only moves what the brief *asks* for, never what the board accepts.
_CEILING_MARGIN = 2


def desk_chat_ceiling(kind: str) -> int:
    """The sentence ceiling for a desk_chat row of record type *kind*.

    Twelve, except where twelve is unreachable. A register is a property of the
    *family* (``work_types.yaml``) and a word budget is a property of the
    *record type*, and one fact pack serves all eight record types -- so
    ``execution.tca.arrival``, whose families all speak desk chat, has to write
    a ``memo`` at a 200-word floor in the desk's voice. At twelve sentences
    that is a 17-word average, and at the ``memo`` band's top it would be 46:
    the two contracts do not both close, and a gate that demanded they did
    would dead-letter every memo of that work type forever.

    So the ceiling is the amendment's twelve or what the lane's floor actually
    permits, whichever is larger, and the rule it still enforces is the one
    worth enforcing: a desk_chat row may not answer in a long list of short
    lines. Raising ``variants_per_family`` will not change that; giving the
    work type an ``ic_memo`` family would, and that is a plan edit, made on
    purpose, not a gate quietly widened.
    """
    low, high = WORD_BUDGETS.get(kind, (0, 0))
    return max(
        DESK_CHAT_MAX_SENTENCES,
        math.ceil(low / DESK_CHAT_WORDS_PER_SENTENCE),
        # ...and enough sentences that the *top* of the band is reachable
        # without breaching the mean. Without this the ceiling and the mean
        # rule had no joint solution for four of the five kinds, and a row
        # obeying one necessarily broke the other: a 341-word memo came back
        # at 16 sentences against a ceiling of 15, having broken its sentences
        # up exactly as instructed. Both bounds come off the same band, so
        # they cannot disagree.
        math.ceil(high / DESK_CHAT_MAX_MEAN_SENTENCE),
    )


#: What each register's gate actually refuses, in the second person, short
#: enough to ride in the brief. Keyed to the ``_desk_chat`` / ``_ic_memo`` /
#: ``_risk_committee`` rules below and meant to be read beside them: every
#: clause here corresponds to one violation string there.
#:
#: This exists because the two had drifted into a silent asymmetry. The brief
#: told a desk_chat row "short sentences, no preamble, no sign-off"; the gate
#: refused it for memo headings, for a labelled ``Call:``, and for running over
#: a sentence ceiling it was never shown. Five of the first nine live rows of a
#: think-off render dead-lettered on exactly the rules nobody had told the
#: model about. A gate the writer cannot see is not a standard, it is a trap,
#: and the cost of one is paid per row, forever, in teacher tokens.
_REGISTER_SHAPE = {
    "desk_chat": (
        "answer in flat prose -- no section headings (no 'Finding:', "
        "'Evidence:', 'Recommendation:'), and do not label your call: reach "
        "the decision inside the sentence you are already writing"
    ),
    "ic_memo": (
        "end on an explicit call -- write 'Call:' or 'Recommendation:' and say "
        "what you would do; a memo that surveys and stops is not a memo. Never "
        "write 'FINAL ANSWER:'"
    ),
    "risk_committee": (
        "name the limit, horizon or assumption you are constraining, and do "
        "not recommend a trade -- the committee sets the constraint, the desk "
        "takes the position"
    ),
}


def register_shape(register: str, kind: str = "") -> str:
    """The shape contract for *register*, in the words the gate would use.

    One string, assembled from the same constants the gate measures against,
    so the brief and the refusal cannot disagree. Returns ``""`` for a
    register with no shape rules (``auditor``, ``code_review``) rather than
    inventing one -- see :func:`register_violations`.
    """
    clauses = []
    shape = _REGISTER_SHAPE.get(register)
    if shape:
        clauses.append(shape)
    if register == "desk_chat":
        ceiling = desk_chat_ceiling(kind)
        # The ceiling *and* a target under it. A language model does not count
        # its own sentences reliably, and one briefed at exactly the ceiling
        # lands on it or just over: the first row of the corrected render came
        # back at 13 against 12, three attempts running. Asking for the target
        # spends the margin in prose rather than in retries.
        clauses.append(
            f"hard ceiling: at most {ceiling} sentences, and aim for "
            f"{max(1, ceiling - _CEILING_MARGIN)} -- count them before you "
            f"answer. Keep sentences under {DESK_CHAT_MAX_MEAN_SENTENCE} words "
            "on average; meeting the ceiling by writing longer sentences is "
            "not meeting it"
        )
    return ". ".join(clauses)


#: The marker the length violation carries so other stages can recognise it
#: without matching on prose. ``render_repair`` needs to know whether a fault
#: is "too long", because the advice it appends contradicts that one fault and
#: no other; matching on the sentence text would make the repair loop depend on
#: the gate's wording, which is the sort of coupling that breaks in silence the
#: next time somebody improves a message.
_TOO_LONG_MARKER = "sentences, over the"


def is_length_violation(violation: str) -> bool:
    """Is *violation* the desk_chat sentence-ceiling refusal?"""
    return _TOO_LONG_MARKER in str(violation or "")


def sentence_count(text: str) -> int:
    """Sentences in *text*, counting a hard line break as a break."""
    parts = [p for p in _SENTENCE_SPLIT.split(str(text or "")) if p.strip()]
    return len(parts)


def register_violations(register: str, text: str, *, kind: str = "") -> list[str]:
    """Every way *text* breaks the shape ``register`` promises, as sentences.

    Empty list means the voice is the one the pack declared. ``kind`` is the
    record type: an ``exam`` row is judged by the exam row of the §C table
    whatever register its pack carries, because an exam item's shape is its
    grading contract and outranks its voice.

    Registers the table does not describe (``auditor``, ``code_review``) return
    clean rather than raising. They are declared in ``VALID_REGISTERS`` and no
    family emits them yet; inventing shape rules for a voice nobody has written
    would be pinning down prose that has never been read.
    """
    text = str(text or "")
    if kind == "exam":
        return _exam(text)
    if register == "desk_chat":
        return _desk_chat(text, kind)
    if register == "ic_memo":
        return _ic_memo(text)
    if register == "risk_committee":
        return _risk_committee(text)
    return []


#: The memo's labelled-call device, *wherever it appears*. Distinct from
#: ``_MEMO_HEADINGS`` on purpose: that one is anchored at a line start because
#: it answers "is this scaffolded like a memo", and a mid-sentence "the finding:
#: we are long" is prose. This answers a different question -- "did the writer
#: reach for the memo's speech act" -- and the answer does not depend on where
#: the label sits.
#:
#: Anchoring cost two live desk_chat rows their register. Both ended
#: ``... Call: execute under current participation, but cap the schedule`` on
#: the same line as the prose before it, both scored ``register_ok: true``, and
#: the register field was decoration for exactly the rows it existed to judge.
_CALL_LABEL = re.compile(r"\b(?:call|recommendation)\s*:", re.IGNORECASE)


def makes_a_call(text: str) -> bool:
    """Whether this passage decides something, in either form a desk uses.

    Two forms, and missing the second cost three of four live ic_memo rows
    their place in the corpus. The model writes its decision under a heading --
    ``Call: use 21.98 as the central case, but flag the dispersion`` -- which
    is not merely *a* way to make a call, it is the form §C's own table
    licenses for this register ("Finding + Evidence + Call allowed").

    The gate already recognised that heading; it just recognised it in the
    wrong place. ``_MEMO_HEADINGS`` matched ``Call:`` to permit memo
    scaffolding while this check, reading only a phrase list, concluded the
    memo had reached no decision. One regex saying yes and one list saying no
    about the same four characters.
    """
    lowered = str(text or "").casefold()
    if any(term in lowered for term in _CALL_TERMS):
        return True
    return bool(_CALL_LABEL.search(str(text or "")))


def _headings(text: str) -> list[str]:
    return sorted({m.group(1).lower() for m in _MEMO_HEADINGS.finditer(text)})


def _desk_chat(text: str, kind: str) -> list[str]:
    out: list[str] = []
    found = _headings(text)
    if found:
        out.append(
            "register desk_chat carries memo headings ("
            + ", ".join(f"{h.title()}:" for h in found)
            + ") -- the desk answers in prose, without section labels"
        )
    # The memo's labelled call, which the desk does not use. A desk note
    # reaches its decision in the sentence it is already writing -- "work it
    # patiently, cap the schedule" -- and labelling it `Call:` is the memo
    # device wearing the desk's name. Checked anywhere in the text, not at a
    # line start: see ``_CALL_LABEL`` for what anchoring cost.
    label = _CALL_LABEL.search(text)
    if label:
        out.append(
            f"register desk_chat labels its call ({label.group(0)!r}) -- the "
            "desk states the call in the sentence it is writing, it does not "
            "announce one; drop the label and keep the decision"
        )
    ceiling = desk_chat_ceiling(kind)
    sentences = sentence_count(text)
    words = len(text.split())
    mean = words / sentences if sentences else 0.0
    if sentences and mean > DESK_CHAT_MAX_MEAN_SENTENCE:
        out.append(
            f"register desk_chat averages {mean:.0f} words a sentence over "
            f"{sentences} sentences, past the {DESK_CHAT_MAX_MEAN_SENTENCE}-word "
            "mark -- the sentence ceiling was met by writing longer sentences, "
            "which is a memo breathing slowly; break the argument up"
        )
    if sentences > ceiling:
        # Say how many to lose, not just that there are too many. The repair
        # turn quotes this string back at the teacher, and "over the ceiling"
        # left it to recount -- which is the operation it had already got
        # wrong. A target is something it can act on in one pass.
        cut = sentences - max(1, ceiling - _CEILING_MARGIN)
        out.append(
            f"register desk_chat runs {sentences} sentences, over the "
            f"{ceiling}-sentence ceiling for a {kind or 'prose'} row -- merge "
            f"or cut at least {cut} of them (a desk note is prose, not a list "
            "of short lines)"
        )
    return out


def _ic_memo(text: str) -> list[str]:
    out: list[str] = []
    # Headings are *permitted* here, not required: the §C table says "Finding +
    # Evidence + Call allowed", and demanding all three would fail a memo that
    # led with its call, which is a memo.
    if _FINAL_ANSWER.search(text):
        out.append(
            "register ic_memo closes on 'FINAL ANSWER:' -- that is the exam "
            "contract, not a memo's call"
        )
    # But it must *make* the call. This is the one positive requirement the §C
    # table does not state and the register cannot do without: an investment
    # committee memo exists to produce a decision, and one that surveys the
    # evidence and stops is a research note with the wrong label on it. It is
    # also the only row-level lever against the collapse axis 16 measures --
    # if ic_memo has a habit desk_chat does not, the two profiles separate.
    #
    # Safe to require because it is already asked for: the teacher's system
    # turn ends "what would move your conclusion, and your call".
    if not makes_a_call(text):
        out.append(
            "register ic_memo states no call -- an investment-committee memo "
            "ends in a decision, not a survey; say what you would do"
        )
    return out


def _risk_committee(text: str) -> list[str]:
    out: list[str] = []
    lowered = text.casefold()
    if not any(term in lowered for term in _RISK_TERMS):
        out.append(
            "register risk_committee names no limit, horizon or assumption -- "
            "the committee's language is what it constrains, not what it thinks "
            "will happen"
        )
    call = _TRADE_CALL.search(text)
    if call:
        out.append(
            f"register risk_committee makes a trade recommendation ({call.group(0)!r}) "
            "-- the committee sets the constraint, the desk takes the position"
        )
    return out


def _exam(text: str) -> list[str]:
    out: list[str] = []
    found = _headings(text)
    if found:
        out.append(
            "an exam answer carries memo headings ("
            + ", ".join(f"{h.title()}:" for h in found)
            + ") -- an item is worked, not written up"
        )
    lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines or not _FINAL_ANSWER.search(lines[-1]):
        out.append(
            "an exam answer must close on its 'FINAL ANSWER:' line and nothing after it"
        )
    return out


# --------------------------------------------------------------------------
# Distinctiveness: a corpus-level question a per-row gate cannot ask
# --------------------------------------------------------------------------
#
# Everything above judges one row against the shape its own register forbids.
# That is the §C table, and it is a *miscategorisation* test: it catches a desk
# note wearing memo headings and a risk paper that constrains nothing. What it
# structurally cannot catch is the failure the corpus is actually at risk of --
# four registers written in one voice. No single row is wrong in that case; the
# slice is, and a row-level gate has no slice to look at.
#
# So the distinctiveness question is asked over a *profile*: three coarse
# numbers per register, compared pairwise. Coarse on purpose. A similarity
# score on embeddings would be more sensitive and less actionable; "ic_memo and
# desk_chat both run 19 words a sentence and neither writes a heading" is a
# sentence somebody can take back to the briefs.

#: How far apart two register profiles must sit to count as distinguishable.
#: The profile axes are normalised to roughly 0-1 each, so this is a distance
#: in a small unit cube -- 0.15 is "they differ on at least one axis by more
#: than a rounding". Tuned to be *insensitive*: the failure being watched for
#: is collapse, not nuance, and an axis that fires on nuance would be switched
#: off within a week.
REGISTER_MIN_SEPARATION = 0.15

#: Words per sentence at which the `sentence_len` axis saturates. A desk note
#: runs ~12 and a memo ~25; 40 is past any register in the table, so dividing
#: by it puts every real corpus inside the unit interval.
_SENTENCE_LEN_SCALE = 40.0


def register_profile(texts: list[str]) -> dict:
    """Three measurable habits of a body of prose, each scaled to 0-1.

    ``sentence_len`` how long the sentences run, ``heading_rate`` how often a
    passage carries memo scaffolding, ``call_rate`` how often it decides
    something rather than describing it. Together they separate the four
    registers the corpus claims to write -- and, more to the point, they fail
    to separate them when the teacher has collapsed into one voice, which is
    the measurement worth having.
    """
    texts = [str(t) for t in texts if str(t).strip()]
    if not texts:
        return {"n": 0, "sentence_len": 0.0, "heading_rate": 0.0, "call_rate": 0.0}
    lengths = []
    for text in texts:
        sentences = max(1, sentence_count(text))
        lengths.append(len(text.split()) / sentences)
    return {
        "n": len(texts),
        "sentence_len": sum(lengths) / len(lengths),
        "heading_rate": sum(1 for t in texts if _headings(t)) / len(texts),
        "call_rate": sum(
            1 for t in texts if any(term in t.casefold() for term in _CALL_TERMS)
        )
        / len(texts),
    }


def profile_distance(left: dict, right: dict) -> float:
    """How far apart two register profiles sit, as a plain Euclidean distance.

    ``sentence_len`` is divided by :data:`_SENTENCE_LEN_SCALE` so all three
    axes carry comparable weight; without it the word count would dominate and
    the two rate axes -- the ones that actually describe a register's *habits*
    -- would never move the number.
    """
    dl = (left["sentence_len"] - right["sentence_len"]) / _SENTENCE_LEN_SCALE
    dh = left["heading_rate"] - right["heading_rate"]
    dc = left["call_rate"] - right["call_rate"]
    return (dl * dl + dh * dh + dc * dc) ** 0.5
