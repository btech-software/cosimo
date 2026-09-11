"""The implementation contract, executable (analysis spec §5.9; arch spec §6 axis 11).

An implementation record is a specification, a test suite, a dirty fixture, a
reference implementation and a statement of limitations. §5.9 fixes the
doctrine; this module is that doctrine with units:

* the **reference is static per work type** (:mod:`impl_references`) and the
  **tests are generated per pack**, pinning the *pack's* computed figures --
  the oracle is the pack, never the reference. ``assert value > 0`` is not a
  test and cannot be emitted from that table: every generated assertion
  compares a value the pack itself states;
* the **dirty fixture** is the pack's inputs with the seed's measure of dirt
  drawn through, and the hidden suite checks the instrument rejects it
  (``ValueError``, per the spec's policy) instead of returning a plausible
  figure for an impossible input;
* **execution is measurement**: :func:`run_sandboxed` runs code in an
  isolated interpreter (``-I -S``, no site, no environment, wall-clock
  timeout) and counts failures. The renderer drops the row when a hidden
  test does not pass; the board re-executes the stored bytes -- and runs
  them *only* after they have agreed byte-for-byte with the recomposition,
  because the board never executes code the corpus did not author.

The pins and the recompute axis read the same ``computed``: a hidden test and
axis 2 cannot be satisfied by two different stories about one pack.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # verification; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ..seed import render_seed, rng_for  # noqa: E402
from .impl_references import IMPL_SPECS  # noqa: E402
from .invented_numbers import invented_numbers  # noqa: E402
from .prose import forbidden_hits, whitelist_for  # noqa: E402

#: The record type this contract governs, named once.
IMPL_KIND = "implementation"

#: Violation tags: the board routes each prefix to the axis that owns it.
TAG_SHAPE = "impl shape: "
TAG_TAMPER = "impl recomposition: "
TAG_SANDBOX = "impl sandbox: "
TAG_FORBIDDEN = "forbidden claim asserted: "

#: One tolerance for every numeric pin. The computers round their figures to
#: two or three places, so a correct re-derivation reproduces them exactly and
#: no finer claim is sound; the tolerance admits the print, never the slop.
TOLERANCE_ABS = 0.011

#: The dirt drawn through: a NaN spelling, a missing reading, an overflowed
#: one. The policy is in the spec; the hidden suite checks the instrument
#: obeys it.
_DIRT_SPELLINGS = ("NaN", None, "1e999")

#: The system turn of an implementation row: the protocol the student is
#: trained to answer in. The answer it trains toward is the table's
#: instrument -- the row says so and the suite proves it.
IMPL_PROTOCOL = (
    "You are completing a Cosimo implementation record. Provide a correct, "
    "self-contained Python implementation of the instrument described: read "
    "every figure from the ``inputs`` mapping, reject every non-finite "
    "reading and every undefined instrument with ``ValueError``, and return "
    "each pinned figure under its own key."
)

#: The one thing asked of the teacher, said once. Short on purpose: the
#: narrower the ask, the fewer ways the answer can draw a gate violation.
LIMITATIONS_ASK = (
    "State, in one short passage of your own words, the limitations of the "
    "instrument specified above: what its inputs are assumed to be, what it "
    "does not model. Do not restate any figure, do not introduce any number, "
    "and make no forward-looking or warranty claim. Answer with the passage "
    "alone."
)


def impl_brief(pack: dict, item: dict) -> list[dict]:
    """The messages the limitations author is asked in -- the contract's own
    words, so the gate, the stage and the fixture harness all read *one*
    definition of what the user turn must say. The replay keys are hashes of
    these bytes; two copies of a wording could not stay byte-equal by
    luck alone, and the board compares the stored turn against this."""
    return [
        {"role": "system", "content": IMPL_PROTOCOL},
        {
            "role": "user",
            "content": (
                f"{item['question_text']}\n\nInstrument: {item['spec']}\n\n"
                + LIMITATIONS_ASK
            ),
        },
    ]


def _family_of(pack: dict) -> str:
    """The scenario family, recovered from the pack's ``scenario_id``."""
    return pack["scenario_id"][len(pack["work_type"]) + 1 :]


def _canonical(value):
    """Order-canonical *dict keys only, at every depth.

    The same inputs reach us insertion-ordered from the computer and
    sort-ordered back from disk, and the dirty test's source bakes the
    fixture's repr -- a repr whose order depended on where the pack had been
    would not be a record of the pack. Lists keep their order: for a peer
    set it is data, not spelling.
    """
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _pin_test(key: str, want) -> str:
    """One generated assertion: the instrument's figure for *key*, pinned to
    the pack's own, compared within ``TOLERANCE_ABS``."""
    return (
        f"got = solve(inputs)[{key!r}]\n"
        f"want = {float(want)!r}\n"
        f"assert abs(got - want) <= max({TOLERANCE_ABS}, abs(want) * 1e-9), "
        f"({key!r}, got, want)\n"
    )


def _edge_test(label: str, mutation: dict) -> str:
    """One generated rejection: the named mis-set dial must raise, not shrug."""
    return (
        "bad = dict(inputs)\n"
        f"bad.update({mutation!r})\n"
        "try:\n"
        "    solve(bad)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        f"    raise AssertionError({label!r} + ': no ValueError raised')\n"
    )


def _dirty_test(label: str, fixture: dict) -> str:
    """The rejection of the whole dirty fixture -- §5.9's audit sentence,
    executable: the instrument must not return a plausible figure for an
    impossible input."""
    return (
        f"bad = {fixture!r}\n"
        "try:\n"
        "    solve(bad)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        f"    raise AssertionError({label!r} + ': the dirty fixture was "
        "solved, not refused')\n"
    )


def compose_impl_item(pack: dict) -> dict:
    """The deterministic record a pack entails -- pure, no disk, no model.

    The renderer writes its row from this (the ``limitations`` field aside,
    the one authored part); the board rebuilds it and compares bytes. The
    only randomness is the seed's choice of which reading to dirty and which
    spelling to dirty it with, drawn from the record type's own render seed
    -- never the wall clock, never the walk order.
    """
    work_type = pack["work_type"]
    spec = IMPL_SPECS.get(work_type)
    if spec is None:
        raise ValueError(
            f"no implementation contract for work type {work_type!r} (known: "
            + ", ".join(sorted(IMPL_SPECS))
            + ")"
        )
    computed = pack.get("computed") or {}
    missing = [key for key in spec["pins"] if computed.get(key) is None]
    if missing:
        raise ValueError(
            f"pack {pack.get('scenario_id')!r} carries no {missing[0]!r}: the "
            "computer and the implementation contract disagree"
        )
    public = [_pin_test(key, computed[key]) for key in spec["public"]]
    rng = rng_for(
        render_seed(work_type, _family_of(pack), IMPL_KIND, int(pack["variant"]))
    )
    # Canonical key order at every depth (see _canonical): the stored bytes
    # must record the pack's content, not the pack's provenance.
    fixture = _canonical(pack.get("inputs") or {})
    keys = [key for key in spec["reads"] if key in fixture]
    if sorted(keys) != sorted(spec["reads"]):
        raise ValueError(
            f"pack {pack.get('scenario_id')!r} lacks the inputs the reference "
            f"consumes (wanted {sorted(spec['reads'])}, has {sorted(fixture)})"
        )
    fixture[keys[rng.randrange(len(keys))]] = _DIRT_SPELLINGS[
        rng.randrange(len(_DIRT_SPELLINGS))
    ]
    hidden = [
        _pin_test(key, computed[key])
        for key in spec["pins"]
        if key not in spec["public"]
    ]
    hidden += [_edge_test(label, mutation) for label, mutation in spec["edges"]]
    hidden.append(_dirty_test("the pack's dirty fixture", fixture))
    statement = (
        f"{pack['question']}\n\n{spec['ask']} {spec['measure']} Reject every "
        "non-finite reading and every undefined instrument with ValueError; "
        "return each pinned figure under its own key.\n\n"
        f"Inputs (clean):\n{json.dumps(pack.get('inputs') or {}, sort_keys=True)}"
        "\n\nPublic tests:\n" + "\n".join(public)
    )
    return {
        "spec": f"{spec['ask']} {spec['measure']} Reject non-finite readings "
        "and undefined instruments with ValueError; return every pinned "
        "figure under its own key.",
        "question_text": statement,
        "reference_code": spec["reference"],
        "public_tests": public,
        "hidden_tests": hidden,
        "dirty_fixture": fixture,
    }


def _runner(reference_code: str, tests: list[str], inputs: dict) -> str:
    """The program the sandboxed interpreter receives.

    Built by concatenation, not ``format``, because the body is full of the
    literal braces of dict displays that a template would eat. The literals
    travel as ``repr`` of their JSON encoding: the only way a corpus string
    becomes Python source without becoming source-trusting.
    """
    return (
        "import json, math\n"
        f"INPUTS = json.loads({json.dumps(json.dumps(inputs, sort_keys=True))})\n"
        f"TESTS = json.loads({json.dumps(json.dumps(list(tests)))})\n"
        "FAILURES = []\n"
        "try:\n"
        + "".join(f"    {line}\n" for line in reference_code.splitlines())
        + "except Exception as _exc:\n"
        "    FAILURES.append('reference: ' + repr(_exc))\n"
        "else:\n"
        "    for _index, _source in enumerate(TESTS):\n"
        "        _namespace = {'inputs': dict(INPUTS), 'solve': solve, "
        "'math': math}\n"
        "        try:\n"
        "            exec(compile(_source, '<test-' + str(_index) + '>', "
        "'exec'), _namespace)\n"
        "        except Exception as _exc:\n"
        "            FAILURES.append(str(_index) + ': ' + repr(_exc))\n"
        "print(json.dumps({'failures': FAILURES}))\n"
    )


def run_sandboxed(reference_code: str, tests: list[str], inputs: dict) -> tuple:
    """Execute reference + suite in an isolated interpreter; count failures.

    ``-I`` isolates (environment and user paths ignored), ``-S`` skips site,
    the timeout bounds what a hung instrument can cost. Returns
    ``(passed, log)`` and never raises for a *content* failure: a failing
    test is data, exactly as the gate's other verdicts are.
    """
    from .. import config  # late: the sandbox constants live with the rest

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _runner(reference_code, tests, inputs)],
            capture_output=True,
            text=True,
            timeout=config.SANDBOX_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, f"sandbox exceeded {config.SANDBOX_TIMEOUT_S}s"
    except OSError as exc:
        return False, f"sandbox could not start: {exc}"
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "no output")[:2000]
    try:
        verdict = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return False, f"sandbox spoke nonsense: {completed.stdout[:500]!r}"
    failures = verdict.get("failures") if isinstance(verdict, dict) else None
    if not isinstance(failures, list):
        return False, f"sandbox reported no verdict: {verdict!r}"
    return not failures, "; ".join(str(failure) for failure in failures[:6])


def limitations_violations(pack: dict, limitations) -> list[str]:
    """The fact-lock on the one authored field (spec §5.6, §5.9)."""
    from .. import config  # late, with the other control-plane reads

    if not isinstance(limitations, str) or not limitations.strip():
        return [f"{TAG_SHAPE} limitations missing"]
    words = limitations.split()
    if len(words) < config.MIN_LIMITATION_WORDS:
        return [
            f"{TAG_SHAPE} limitations is a shrug, not a statement "
            f"({len(words)} words; need {config.MIN_LIMITATION_WORDS})"
        ]
    violations = [
        f"{TAG_SHAPE} {token!r} in limitations is not a number of the pack"
        for token in invented_numbers(
            limitations, pack.get("allowed_numbers") or [], whitelist_for(pack)
        )
    ]
    violations += [
        f"{TAG_FORBIDDEN} {claim!r}" for claim in forbidden_hits(pack, limitations)
    ]
    return violations


def impl_gate_violations(pack: dict, row: dict) -> list[str]:
    """Every way a stored implementation row fails its contract, tagged.

    Recomposition first, execution never reached here: the board compares the
    stored bytes against :func:`compose_impl_item` and runs the suite only on
    bytes that agreed -- tampered code is reported, not executed.
    """
    violations: list[str] = []
    try:
        item = compose_impl_item(pack)
    except ValueError as exc:
        # Tamper-filed, not shape-filed: a pack that composes nothing cannot
        # be byte-compared, and the board runs only bytes it has compared.
        return [f"{TAG_TAMPER} the pack no longer composes a record: {exc}"]
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return [f"{TAG_SHAPE} answer (the reference implementation) is empty"]
    for field in (
        "reference_code",
        "spec",
        "public_tests",
        "hidden_tests",
        "dirty_fixture",
    ):
        if row.get(field) != item[field]:
            violations.append(
                f"{TAG_TAMPER} {field} on the shard is not what the pack entails"
            )
    messages = row.get("messages") or []
    # Two turns, not three, and the user turn is the *spec* rather than
    # ``impl_brief``'s wrapper around it. The brief asks the teacher for a
    # statement of limitations -- it is the factory talking, and §A keeps it
    # off every trainable surface. What the record's prompt has always been is
    # the instrument's own spec, which is what the pack entails and what the
    # recomposition below re-derives.
    if [m.get("role") for m in messages] != ["user", "assistant"]:
        violations.append(f"{TAG_SHAPE} message roles are not the impl pair")
    else:
        if messages[-1].get("content") != answer:
            violations.append(
                f"{TAG_SHAPE} answer and the assistant turn have drifted apart"
            )
        if messages[0].get("content") != item["spec"]:
            violations.append(
                f"{TAG_TAMPER} the question on the shard is not what the pack entails"
            )
    if answer != item["reference_code"]:
        violations.append(
            f"{TAG_TAMPER} the shipped implementation is not the reference the "
            "pack entails"
        )
    violations += limitations_violations(pack, row.get("limitations"))
    return violations


__all__ = [
    "IMPL_KIND",
    "IMPL_PROTOCOL",
    "IMPL_SPECS",
    "LIMITATIONS_ASK",
    "TOLERANCE_ABS",
    "TAG_FORBIDDEN",
    "TAG_SANDBOX",
    "TAG_SHAPE",
    "TAG_TAMPER",
    "compose_impl_item",
    "impl_brief",
    "impl_gate_violations",
    "limitations_violations",
    "run_sandboxed",
]
