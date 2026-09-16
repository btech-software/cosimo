"""A verifiable reward over the gates the corpus already enforces.

The v3 corpus computes its facts before it writes a word, so every row carries
the contract its answer was held to: the pack's canonical figures, the terms a
grounded answer owes, an exam item's key, an implementation row's hidden tests,
an agentic row's expected calls. Those are *verifiers* -- cheap, low-noise and
hard to game -- which is exactly the signal reinforcement learning wants and
the thing a learned reward model only approximates.

Nothing here is new judgement. Every component delegates to the function the
evaluation already scores with (``assistant``/``grading``), so a reward and a
metric can never disagree about the same answer; this module only decides how
they compose into one number a trainer can rank rollouts by.

Two of those decisions are *gates* rather than terms, and the distinction
matters more than the weights. A fabricated figure and a collapsed register
set the reward to zero however good the rest of the answer is, because a
quant assistant that invents a number is not a good answer with one flaw --
it is the failure the corpus exists to prevent. Averaging would teach that a
fabricated number is survivable if the prose around it is strong.

The gate on register is the aggressive choice and is recorded as such: it is a
shallow shape check by its own author's admission, and RL is good at finding
shallow checks. ``gates_tripped`` exists so that is visible -- a run whose
register gate trips on almost nothing while its text drifts is a run being
gamed, and the counter is what shows it.

Scores are per record type and comparable in [0, 1]. Which prompts get sampled,
and how the types are balanced against each other, is the training script's
decision, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import assistant, grading

#: Record types whose answer is prose held to the pack's contract.
PROSE_KINDS = frozenset({"analysis", "memo", "critique", "grounded", "abstention"})

#: Arithmetic furniture rather than pack facts: the small integers prose counts
#: with, percent denominators, the trading-day convention, the basis-point
#: factor. Mirrors ``dataset/pipelines/v3/config.NUMBER_WHITELIST``, and it is
#: the corpus's list rather than the narrower one ``configs/assistant.yaml``
#: hands the evaluation on purpose: a reward that refused what the generator
#: permitted would train the model away from the corpus it was given.
NUMBER_WHITELIST = (
    *(str(n) for n in range(11)),
    "100",
    "252",
    "10000",
    "10,000",
)


def number_whitelist(record: dict) -> tuple[str, ...]:
    """The tokens this row may write beyond the figures its pack entails.

    The constant above, plus every number spelled in the pack's ``as_of``
    date: a row is allowed to *date* its analysis, and the date is the pack's
    own, so letting it through re-authorises nothing the pack did not state.

    Read with the same reader the answer is read with. ``2026-06-30`` is three
    tokens, and which three depends on whether a hyphen counts as a sign -- so
    a whitelist built by one reader and checked by another authorises spellings
    that never appear and refuses the ones that do.
    """
    pack = record.get("fact_pack") or {}
    as_of = str(pack.get("as_of") or record.get("as_of") or "")
    return tuple(NUMBER_WHITELIST) + tuple(assistant.number_tokens(as_of))


@dataclass
class RewardBreakdown:
    """One scored rollout: the number, and why it is that number.

    ``total`` is what a trainer ranks on. ``components`` and ``gates_tripped``
    are what an operator reads when the number is surprising -- a reward with
    no decomposition is a reward nobody can debug, and the first question of
    any RL run that plateaus is which term stopped moving.
    """

    total: float
    kind: str
    components: dict[str, float] = field(default_factory=dict)
    gates_tripped: list[str] = field(default_factory=list)

    @property
    def gated(self) -> bool:
        return bool(self.gates_tripped)


def allowed_numbers(record: dict) -> list[float]:
    """The figures the pack entails, from ``canonical`` and not the union.

    Two permissions, and both are needed. ``canonical`` holds one value per
    named quantity -- the union carries the question's own roundings, so
    grading against it accepts the drift the corpus repairs. ``conventions``
    holds the named constants the work type declared in advance, which is what
    lets a risk row say the Basel multiplier starts at 3 without inventing it.

    Reading only the first is not a stricter gate but a broken one: the
    evaluation records a live answer, right arithmetic and right reading, that
    scored ``invented_numbers`` 1.000 for citing its own conventions -- so the
    headline number for the corpus's whole purpose was a false positive.
    """
    pack = record.get("fact_pack") or {}
    canonical = pack.get("canonical") or {}
    values = sorted({v for v in _scalars(canonical) if isinstance(v, (int, float))})
    if not values:
        values = [float(x) for x in (pack.get("allowed_numbers") or [])]
    conventions = _scalars(pack.get("conventions") or record.get("conventions") or [])
    return sorted(set(values) | set(conventions))


def _scalars(node) -> list[float]:
    out: list[float] = []
    if isinstance(node, bool):
        return out
    if isinstance(node, (int, float)):
        return [float(node)]
    if isinstance(node, dict):
        for value in node.values():
            out.extend(_scalars(value))
    elif isinstance(node, (list, tuple)):
        for value in node:
            out.extend(_scalars(value))
    return out


def _gates(record: dict, text: str, kind: str) -> list[str]:
    """The refusals no other term can outweigh.

    ``exam_shape`` is here for every kind but ``exam``: the grading liturgy is
    an exam item's contract and a leak of it anywhere else is the register
    collapse the corpus was rebuilt to prevent, not a stylistic slip.
    """
    tripped: list[str] = []
    allowed = allowed_numbers(record)
    if allowed:
        invented = assistant.invented_numbers(text, allowed, number_whitelist(record))
        if invented:
            tripped.append(f"invented_numbers({len(invented)})")
    if kind != "exam" and assistant.has_exam_shape(text):
        tripped.append("exam_shape")
    register = record.get("register")
    if register:
        violations = assistant.register_violations(str(register), text, kind=kind)
        if violations:
            tripped.append(f"register({len(violations)})")
    return tripped


def _prose_score(record: dict, text: str) -> dict[str, float]:
    """What a prose answer earns once the gates have let it through.

    Coverage only. The figures were already judged -- by a gate, not a term --
    and a second, softer opinion about them here would be the corpus's own
    mistake repeated: one regex saying yes and one list saying no about the
    same number.
    """
    required = assistant.mention_points(
        record.get("fact_pack") or {}, str(record.get("record_type") or "")
    )
    if not required:
        # A row that owes no terms is not a row that scores zero for owing
        # none. It passed the gates; that is the whole of its contract.
        return {"must_mention": 1.0}
    hit, _ = assistant.must_mention_hits(text, required)
    return {"must_mention": len(hit) / len(required)}


def _exam_score(record: dict, text: str) -> dict[str, float]:
    """Binary, against the key the computer wrote.

    No partial credit: an exam item has one right answer, and its distractors
    are named wrong models rather than near misses -- landing on one is not
    most of the way there, it is the specific error the item was built to
    catch.

    The gold the grader needs is the *option*, not the row's ``answer`` field:
    that field holds the worked item, tag and all, and grading it against
    itself scores every row half-right for the reading rather than the answer.
    ``options`` carries the labelled choices and ``answer_value`` the figure
    they point at, which is what the student is actually being asked for.
    """
    options = record.get("options") or []
    letter = next(
        (
            str(o.get("label"))
            for o in options
            if o.get("value") == record.get("answer_value")
        ),
        None,
    )
    gold = letter or record.get("answer_value")
    if gold is None:
        return {"answer": 0.0}
    graded = dict(record)
    graded["answer"] = str(gold)
    graded["question_type"] = "MCQ" if letter else ""
    grade = grading.grade_cosimo(graded, text)
    return {
        "answer": 1.0 if grade.correct else 0.0,
        "format": 1.0 if grade.format_ok else 0.0,
    }


def _implementation_score(record: dict, text: str, run_tests) -> dict[str, float]:
    """The share of hidden tests the code passes.

    ``run_tests`` is injected rather than imported: executing model-written
    code is the caller's decision, made once with a sandbox it trusts, and a
    reward module that shelled out on its own would make every unit test of
    this file a code-execution risk.
    """
    hidden = record.get("hidden_tests") or []
    if not hidden or run_tests is None:
        # Unscoreable, not failed. Returning 0.0 would teach the policy that
        # every implementation rollout is worthless whenever the caller
        # declined to run a sandbox, which is a training signal invented by
        # the absence of a tool rather than by the answer.
        return {}
    passed, total = run_tests(text, hidden)
    return {"hidden_tests": (passed / total) if total else 0.0}


def _agentic_score(record: dict, calls, text: str) -> dict[str, float]:
    """The trajectory gate the eval already applies, read as a number."""
    scenario = record.get("scenario") or record
    result = assistant.grade_trajectory(scenario, list(calls or []), text)
    return {"trajectory": 1.0 if result.get("correct") else 0.0}


def verifiable_reward(
    record: dict,
    completion: str,
    *,
    calls=None,
    run_tests=None,
) -> RewardBreakdown:
    """Score one rollout of *record* in [0, 1].

    The gates are checked first and short-circuit to zero. What follows is the
    record type's own contract, and only rows of that type are ever compared
    against each other -- balancing the types is the sampler's problem, which
    is why nothing here is normalised across them.
    """
    text = str(completion or "")
    kind = str(record.get("record_type") or "")
    tripped = _gates(record, text, kind)
    if tripped:
        return RewardBreakdown(0.0, kind, {}, tripped)

    if kind == "exam":
        components = _exam_score(record, text)
    elif kind == "agentic":
        components = _agentic_score(record, calls, text)
    elif kind == "implementation":
        components = _implementation_score(record, text, run_tests)
    elif kind in PROSE_KINDS:
        components = _prose_score(record, text)
    else:
        # An unknown record type scores nothing rather than a default: a
        # silent 1.0 here would reward a row nobody wrote a contract for.
        return RewardBreakdown(0.0, kind, {}, ["unknown_record_type"])

    if not components:
        return RewardBreakdown(0.0, kind, {}, ["unscoreable"])
    total = sum(components.values()) / len(components)
    return RewardBreakdown(total, kind, components, [])
