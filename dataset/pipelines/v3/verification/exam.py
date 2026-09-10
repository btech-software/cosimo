"""The exam contract, executable (analysis spec §5.10; arch spec §6, axes 5-8).

The exam row is the one record type the teacher never writes: the item, its
worked solution and its distractors are *composed* from the fact pack by this
module, deterministically, so an exam row is verifiable by recomposition --
the board rebuilds the item from ``(work_type, family, variant)`` and compares
bytes, exactly as axes 2-5 re-derive prose's authority from the pack. One
implementation, two call sites (the renderer builds through it, the board
re-audits through it), the same discipline that keeps prose's gate and the
board from disagreeing about what "clean" means.

Three §5.10 promises live here, and where each is enforced matters:

* **trace shape is sampled**, four shapes over the item's own seed -- prose,
  back-of-envelope, short table, and the v1 liturgy (``ASSUMPTIONS:`` +
  ``Step N.``) kept as the fourth, capped at ``config.LITURGY_CAP`` of the
  slice. The cap is not trusted to the sampler's luck: the liturgy is a
  residue class of the variant (``config.EXAM_LITURGY_*``), so no
  *pre-thinning* slice can exceed it; the board still measures the shipped
  corpus, and a measured overshoot then means the pack guard thinned the
  slice unevenly -- itself a finding worth a red axis.
* **distractors are generated from named pitfalls, then checked != correct**
  (see :mod:`verification.exam_pitfalls`). A derivation whose printed form
  collides with a value already kept is dropped, not patched.
* **``FINAL ANSWER:`` lives on the exam side of the wall** (axis 5). Its
  absence from an exam answer is as unforgivable as its presence in a memo.

The numbers the *student* may print are the numbers the pack authorised: the
correct value is ``computed[answer_key]`` verbatim and every trace figure is a
named component of ``computed`` (flattened by ``assemble_numbers`` into
``allowed_numbers``). The options list -- including the wrong ones -- is
presented in the *question*, where the examiner's own figures live: an
examiner's distractors are data of the item, not claims of the student, and
axis 3 grades the answer, as it always has. The board re-derives the options
and compares bytes, so a hand-edited item is caught regardless.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # verification; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verification.gates import FINAL_ANSWER_TAG  # noqa: E402

from .. import config  # noqa: E402
from ..seed import render_seed, rng_for  # noqa: E402
from .exam_pitfalls import PITFALL_DERIVATIONS  # noqa: E402
from .invented_numbers import invented_numbers  # noqa: E402
from .prose import forbidden_hits, whitelist_for  # noqa: E402

#: The record type this contract governs, named once.
EXAM_KIND = "exam"

#: The four trace shapes (§5.10). The fourth is the v1 liturgy, kept: it is
#: the shape 12% of v1 *was*, and the cap is what separates a house style
#: from a collapse. The first three are indexed by the variant's residue
#: when the draw is not liturgical.
TRACE_STYLES = ("prose", "back_of_envelope", "short_table", "liturgy")
LITURGICAL_STYLE = "liturgy"

#: Violation tags: the board routes each prefix to the axis that owns it,
#: exactly as the agentic gate's REPLAY/GROUNDING tags do.
TAG_SHAPE = "exam shape: "
TAG_TAMPER = "exam recomposition: "
TAG_FINAL = "exam final answer: "
TAG_INVENTED = "exam invented number: "
TAG_FORBIDDEN = "forbidden claim asserted: "

#: What each work type's item resolves and how its slice prints. ``trace``
#: lists the ``computed`` keys the worked solution may cite (all already in
#: ``allowed_numbers`` by the pack's own flattening); ``print_spec`` is the
#: house style for figures of this work type, matching the pack's own
#: question typography so a value keeps one spelling corpus-wide.
EXAM_SPECS: dict[str, dict] = {
    "valuation.equity.dcf": {
        "answer_key": "enterprise_value_m",
        "answer_label": "the enterprise value",
        "unit": "$M",
        "print_spec": "{:,.2f}",
        "trace": (
            ("pv_explicit_m", "the explicit strip"),
            ("pv_terminal_m", "the discounted terminal value"),
        ),
    },
    "valuation.equity.multiples": {
        "answer_key": "implied_value_per_share",
        "answer_label": "the implied value per share",
        "unit": "$/share",
        "print_spec": "{:,.2f}",
        "trace": (
            ("implied_ev_m", "the implied enterprise value"),
            ("implied_equity_m", "the bridged equity"),
        ),
    },
    "risk.market.var_es": {
        "answer_key": "var95_h_m",
        "answer_label": "the horizon 95% VaR",
        "unit": "$M",
        "print_spec": "{:,.2f}",
        "trace": (
            ("var95_1d_m", "the one-day quantile"),
            ("horizon_scale_sqrt", "scaled by the square root of the horizon"),
        ),
    },
    "portfolio.attribution.brinson_carino": {
        "answer_key": "active_bps",
        "answer_label": "the active return",
        "unit": "bp",
        "print_spec": "{:,.1f}",
        "trace": (
            ("allocation_total_bps", "allocation"),
            ("selection_total_bps", "selection"),
            ("interaction_total_bps", "interaction"),
        ),
    },
    "execution.tca.arrival": {
        "answer_key": "cost_bps_vs_arrival",
        "answer_label": "the implementation shortfall",
        "unit": "bp",
        "print_spec": "{:,.2f}",
        "trace": (
            ("spread_cost_bps", "the half-spread"),
            ("impact_bps", "the square-root impact"),
        ),
    },
}

#: The markers axis 6 counts, defined once: the board scans the *shipped
#: text* for these (marker detection, not trust in the render field), so a
#: row that claims ``liturgy: false`` while printing ``Step 1.`` is still
#: counted. The compositor emits exactly these spellings; widening either
#: side silently would desynchronise the measurement.
LITURGY_MARKERS = ("ASSUMPTIONS:", "Step 1.")

#: The ordinals the trace shapes themselves print ("Step 1.", "Step 2.",
#: "Step 3." in the liturgy) are typography of the *method*, not figures of
#: the scenario -- the same species as the tiny whitelist's "the two of a
#: two-way bridge" (§5.6). Enumerated from the shapes actually composed, so
#: the mercy extends exactly as far as the compositor writes.
EXAM_SHAPE_TOKENS = frozenset(
    str(step)
    for step in range(1, 1 + max(len(s["trace"]) for s in EXAM_SPECS.values()))
)


def _family_of(pack: dict) -> str:
    """The scenario family, recovered from the pack's ``scenario_id``."""
    return pack["scenario_id"][len(pack["work_type"]) + 1 :]


def _print(value: float, print_spec: str) -> str:
    return print_spec.format(float(value))


def _trace_lines(style: str, spec: dict, pack: dict, printed_answer: str) -> list:
    """The worked solution, in the drawn shape. Figures are pack components."""
    computed = pack.get("computed") or {}
    parts = []
    for key, label in spec["trace"]:
        value = _num_or_none(computed.get(key))
        if value is None:
            raise ValueError(
                f"pack {pack.get('scenario_id')}: trace key {key!r} absent from "
                "computed -- the computer and the exam contract disagree"
            )
        parts.append((label, _print(value, spec["print_spec"])))
    if style == "prose":
        climb = "; ".join(f"{label} {value}" for label, value in parts)
        return [
            f"Work the instrument through: {climb}.",
            f"So {spec['answer_label']} is {printed_answer} {spec['unit']}.",
        ]
    if style == "back_of_envelope":
        figures = ", ".join(value for _, value in parts)
        return [
            f"{figures} -- add them up the way the formula says.",
            f"Roughly {printed_answer}.",
        ]
    if style == "short_table":
        rows = [f"  {label:>34}  {value}" for label, value in parts]
        rows.append(f"  {spec['answer_label']:>34}  {printed_answer}")
        # No leading blank line. This used to open with "", which put a newline
        # at the very start of the assistant turn -- and the student's chat
        # template already ends the role header with one. The two merge into a
        # single "\n\n" token, so `train_on_responses_only` stops finding its
        # "<|im_start|>assistant\n" marker, masks every label to -100, and the
        # harness drops the row: 45 of 163 exam rows, silently, blamed on
        # truncation. A supervised target may not begin with whitespace.
        return ["\n".join(rows).strip()]
    if style == "liturgy":
        lines = ["ASSUMPTIONS: the pack's own, taken as stated."]
        for step, (label, value) in enumerate(parts, start=1):
            lines.append(f"Step {step}. {label}: {value}.")
        return lines
    raise ValueError(f"unknown trace style {style!r}")


def _num_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def compose_exam_item(pack: dict) -> dict:
    """The deterministic item a pack line entails -- pure, no disk, no model.

        The renderer writes its row from this; the board rebuilds it and compares
        bytes. Every draw is from :func:`seed.rng_for` on
    :func:`seed.render_seed` of the pack's own coordinates, so two runs (or a
        laptop and a Spark executor) compose the identical item.
    """
    work_type = pack["work_type"]
    spec = EXAM_SPECS.get(work_type)
    if spec is None:
        raise ValueError(
            f"no exam contract for work type {work_type!r} (known: "
            + ", ".join(sorted(EXAM_SPECS))
            + ")"
        )
    computed = pack.get("computed") or {}
    inputs = pack.get("inputs") or {}
    variant = int(pack["variant"])
    answer_value = _num_or_none(computed.get(spec["answer_key"]))
    if answer_value is None:
        raise ValueError(
            f"pack {pack.get('scenario_id')!r} variant {variant} carries no "
            f"{spec['answer_key']!r}: the computer and the exam contract disagree"
        )
    print_spec = spec["print_spec"]
    printed_answer = _print(answer_value, print_spec)

    rng = rng_for(render_seed(work_type, _family_of(pack), EXAM_KIND, variant))
    if variant % config.EXAM_LITURGY_MODULUS == config.EXAM_LITURGY_RESIDUE:
        style = LITURGICAL_STYLE
    else:
        style = TRACE_STYLES[variant % (len(TRACE_STYLES) - 1)]

    # Distractors: sample the named wrong models, drop what does not print
    # distinctly (§5.10: drop the distractor, do not inflate the item).
    registry = PITFALL_DERIVATIONS.get(work_type) or ()
    candidates = list(registry)
    rng.shuffle(candidates)
    kept: list[dict] = []
    seen_prints = {printed_answer}
    for name, derive in candidates:
        if len(kept) >= config.EXAM_MAX_DISTRACTORS:
            break
        try:
            value = derive(inputs, computed)
        except Exception:  # a wrong model may fail arithmetically on a draw
            value = None  # (sqrt of a negative, log of a zero): that is a miss
        value = _num_or_none(value)
        if value is None:
            continue
        printed = _print(value, print_spec)
        if printed in seen_prints:
            continue
        seen_prints.add(printed)
        kept.append({"value": value, "text": printed, "pitfall": name})
    if len(kept) < config.EXAM_MIN_DISTRACTORS:
        raise ValueError(
            f"exam slice for {pack.get('scenario_id')!r} variant {variant} "
            f"survives with only {len(kept)} distinct distractors "
            f"(need {config.EXAM_MIN_DISTRACTORS}); the item is not an item"
        )

    # Seat the correct answer among the distractors and label A, B, C...
    body = [{"value": answer_value, "text": printed_answer, "pitfall": None}]
    body.extend(kept)
    rng.shuffle(body)
    options = [
        {"label": chr(ord("A") + position), **entry}
        for position, entry in enumerate(body)
    ]
    letters = [o["label"] for o in options]
    if len(letters) != len(set(letters)):
        raise AssertionError("option labels collided -- a compositor bug")

    trace = _trace_lines(style, spec, pack, printed_answer)
    # The final line names the letter and the figure: the contract of axis 5.
    correct_label = next(
        option["label"]
        for option in options
        if option["pitfall"] is None and option["text"] == printed_answer
    )
    answer_text = "\n".join(
        [
            *trace,
            f"{FINAL_ANSWER_TAG} {correct_label} -- {printed_answer} {spec['unit']}",
        ]
    )
    question_text = (
        pack["question"]
        + "\n\nOptions:\n"
        + "\n".join(
            f"({option['label']}) {option['text']} {spec['unit']}" for option in options
        )
    )
    return {
        "style": style,
        "liturgy": style == LITURGICAL_STYLE,
        "answer_key": spec["answer_key"],
        "answer_value": answer_value,
        "unit": spec["unit"],
        "options": options,
        "distractor_values": [d["value"] for d in kept],
        "distractor_names": [d["pitfall"] for d in kept],
        "question_text": question_text,
        "answer_text": answer_text,
    }


def exam_gate_violations(pack: dict, row: dict) -> list[str]:
    """Every way a composed exam row fails its own contract, tagged by owner.

    Runs on the *stored* row against the *recomputed* item: a row that
    survives this is byte-identical to what the pack entails, so the shard
    can be audited without trusting the file it came from.
    """
    violations: list[str] = []
    try:
        item = compose_exam_item(pack)
    except ValueError as exc:
        return [f"{TAG_SHAPE} the pack no longer composes an item: {exc}"]
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        violations.append(f"{TAG_SHAPE} answer is empty")
        return violations
    if FINAL_ANSWER_TAG not in answer:
        violations.append(
            f"{TAG_FINAL} {FINAL_ANSWER_TAG!r} missing -- an exam row must close "
            "its working where a memo never opens one"
        )
    if answer != item["answer_text"]:
        violations.append(
            f"{TAG_TAMPER} the answer on the shard is not the item the pack "
            "entails (composited bytes differ)"
        )
    messages = row.get("messages") or []
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        violations.append(f"{TAG_SHAPE} message roles are not the exam triad")
    elif messages[-1].get("content") != answer:
        violations.append(
            f"{TAG_SHAPE} answer and the assistant turn have drifted apart"
        )
    elif messages[1].get("content") != item["question_text"]:
        violations.append(
            f"{TAG_TAMPER} the question on the shard is not the item the pack "
            "entails (options tampered?)"
        )
    if row.get("options") != item["options"]:
        violations.append(
            f"{TAG_TAMPER} the options on the shard are not the item the pack entails"
        )
    if row.get("answer_value") != item["answer_value"]:
        violations.append(f"{TAG_TAMPER} answer_value disagrees with the pack")
    render = (row.get("verification") or {}).get("render") or {}
    if render.get("style") != item["style"]:
        violations.append(f"{TAG_TAMPER} the trace style field lies about the text")
    # Axis 3, the exam reading: the student's figures are the pack's own;
    # distractors live in the question (the examiner's side of the wall).
    for token in invented_numbers(
        answer,
        pack.get("allowed_numbers") or [],
        whitelist_for(pack) | EXAM_SHAPE_TOKENS,
    ):
        violations.append(f"{TAG_INVENTED} {token!r} is not a number of the pack")
    for claim in forbidden_hits(pack, answer):
        violations.append(f"{TAG_FORBIDDEN} {claim!r}")
    return violations


__all__ = [
    "EXAM_KIND",
    "EXAM_SHAPE_TOKENS",
    "EXAM_SPECS",
    "LITURGICAL_STYLE",
    "LITURGY_MARKERS",
    "TAG_FINAL",
    "TAG_FORBIDDEN",
    "TAG_INVENTED",
    "TAG_SHAPE",
    "TAG_TAMPER",
    "TRACE_STYLES",
    "compose_exam_item",
    "exam_gate_violations",
]
