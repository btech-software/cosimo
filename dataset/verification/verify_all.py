"""Full-scan regression runner. Exits non-zero on any failure.

Composes every gate over the whole corpus:

  1. structure          required fields per record_type, valid defects, id uniqueness
  2. numeric            exam answers recompute from (template, seed); traces match
  3. format             `FINAL ANSWER:` appears only on exam records
  4. implementation     the generated Python parses, executes, and its tests pass
  5. agentic            tool calls valid, results present, answer uses them
  6. preference         chosen != rejected, disjoint id space, mode tagged
  7. terminology        eponym/concept collocations (verification/terms.py)
  8. length             mixed-set p95 floor (verification/length_gate.py)
  9. suite overlap      no near-duplicate of a held-out eval prompt

The first version of this file checked four numeric axes and **skipped every
non-exam record** (`if not v.get('template'): skipped`), which is why a corpus
that was 99.96% exam could pass it at 100%.

Run:
    python3 verification/verify_all.py
    python3 verification/verify_all.py --quick     # skip gates 4 and 9
"""
import ast
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import length_gate  # noqa: E402
import suite_overlap  # noqa: E402
import terms  # noqa: E402
from gates import (  # noqa: E402
    EXAM_ONLY_MARKERS, REQUIRED_FIELDS, Result, VALID_DEFECTS, by_type,
    load_records, record_type,
)
from nums import nums  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines import core  # noqa: E402

PROG2MOD = {
    "CFA_Level_I": "cfa_l1", "CFA_Level_II": "cfa_l2", "CFA_Level_III": "cfa_l3",
    "FRM_Part_1": "frm1", "FRM_Part_2": "frm2",
}


# ---------------------------------------------------------------------------
# 1. structure
# ---------------------------------------------------------------------------

def gate_structure(records):
    result = Result("1. structure")
    seen = {}
    for rec in records:
        result.checked += 1
        rid = rec.get("id")
        rtype = record_type(rec)
        if not rid:
            result.fail("?", "record has no id")
            continue
        if rid in seen:
            result.fail(rid, f"duplicate id (also in {seen[rid]})")
        seen[rid] = rec.get("_source", "?")
        if rtype not in REQUIRED_FIELDS:
            result.fail(rid, f"unknown record_type {rtype!r}")
            continue
        for field in REQUIRED_FIELDS[rtype]:
            if not rec.get(field):
                result.fail(rid, f"{rtype} record is missing {field!r}")
        if rtype == "abstention":
            defect = (rec.get("metadata") or {}).get("defect")
            if defect not in VALID_DEFECTS:
                result.fail(rid, f"metadata.defect is {defect!r}, "
                                 f"expected one of {VALID_DEFECTS}")
        if not (rec.get("metadata") or {}).get("round"):
            result.fail(rid, "metadata.round is missing; cross-round attribution "
                             "and round-based holdout both need it")
    return result


# ---------------------------------------------------------------------------
# 2. numeric recomputation (exam)
# ---------------------------------------------------------------------------

def gate_numeric(records):
    result = Result("2. numeric recompute")
    mods = {}
    for rec in records:
        if record_type(rec) != "exam":
            continue
        ver = rec.get("verification") or {}
        template, seed = ver.get("template"), ver.get("seed")
        if not template or seed is None:
            continue
        program = rec.get("program")
        if program not in PROG2MOD:
            continue
        if program not in mods:
            mods[program] = importlib.import_module(
                "pipelines.templates." + PROG2MOD[program]
            )
        fn = mods[program].TEMPLATES.get(template)
        if fn is None:
            result.fail(rec["id"], f"template {template!r} no longer exists")
            continue
        result.checked += 1
        try:
            rich = fn(core.RNG(seed), 0)
        except Exception as exc:
            result.fail(rec["id"], f"recomputation raised {type(exc).__name__}: {exc}")
            continue
        if nums(rich["answer"]) != nums(rec["answer"]):
            result.fail(rec["id"], "answer does not match recomputation")
        if rich["reasoning_trace"] != rec["reasoning_trace"]:
            result.fail(rec["id"], "reasoning trace does not match recomputation")
        answer_n = nums(rec["answer"])
        for distractor in rec.get("distractors") or []:
            if nums(distractor) and nums(distractor) == answer_n:
                result.fail(rec["id"], "a distractor equals the correct answer")
                break
    return result


# ---------------------------------------------------------------------------
# 3. format -- FINAL ANSWER: is an exam grading contract, not a house style
# ---------------------------------------------------------------------------

def gate_format(records):
    result = Result("3. format")
    for rec in records:
        rtype = record_type(rec)
        if rtype == "exam":
            continue
        result.checked += 1
        blob = " ".join(
            str(rec.get(f) or "")
            for f in ("answer", "reasoning_trace", "chosen", "rejected")
        )
        for marker in EXAM_ONLY_MARKERS:
            if marker in blob:
                result.fail(
                    rec.get("id", "?"),
                    f"{rtype} record carries {marker!r}; attaching the exam "
                    f"grading contract to non-exam records is what taught the "
                    f"model to answer everything as an exam item",
                )
    return result


# ---------------------------------------------------------------------------
# 4. implementation -- the generated Python must actually run
# ---------------------------------------------------------------------------

def gate_implementation(records):
    result = Result("4. implementation exec")
    for rec in records:
        if record_type(rec) != "implementation":
            continue
        result.checked += 1
        code = rec.get("code") or ""
        try:
            ast.parse(code)
        except SyntaxError as exc:
            result.fail(rec["id"], f"code does not parse: {exc.msg} (line {exc.lineno})")
            continue
        namespace = {"__name__": "__generated__"}
        try:
            exec(compile(code, f"<{rec['id']}>", "exec"),  # noqa: S102
                 {"__builtins__": __builtins__, **_SAFE_IMPORTS}, namespace)
        except Exception as exc:
            result.fail(rec["id"], f"code raised on import: {type(exc).__name__}: {exc}")
    return result


_SAFE_IMPORTS = {}
for _name in ("math", "statistics", "json", "random", "itertools", "functools"):
    _SAFE_IMPORTS[_name] = importlib.import_module(_name)


# ---------------------------------------------------------------------------
# 5. agentic structure
# ---------------------------------------------------------------------------

def gate_agentic(records):
    result = Result("5. agentic structure")
    for rec in records:
        if record_type(rec) != "agentic":
            continue
        result.checked += 1
        rid = rec["id"]
        conversation = rec.get("conversation") or []
        schemas = rec.get("tool_schemas") or []
        offered = {
            (s.get("function") or {}).get("name")
            for s in schemas
            if isinstance(s, dict)
        }
        roles = [t.get("role") for t in conversation]
        if "tool_result" in roles:
            result.fail(rid, "conversation uses role 'tool_result'; the chat "
                             "template only recognises 'tool', and a wrong role "
                             "puts the tool output in the supervised span")
        if not any(r == "assistant" for r in roles):
            result.fail(rid, "conversation has no assistant turn")
        called = []
        for turn in conversation:
            for call in turn.get("tool_calls") or []:
                fn = call.get("function") or call
                name = fn.get("name")
                called.append(name)
                if offered and name not in offered:
                    result.fail(rid, f"calls tool {name!r} which was never offered")
                args = fn.get("arguments")
                if not isinstance(args, (dict, str)):
                    result.fail(rid, f"tool call {name!r} has non-JSON arguments")
        results_present = [t for t in conversation if t.get("role") == "tool"]
        if called and not results_present:
            result.fail(rid, "makes a tool call but no tool result comes back")
        if results_present and roles[-1] != "assistant":
            result.fail(rid, "conversation does not end with an assistant answer")
    return result


# ---------------------------------------------------------------------------
# 6. preference pairs
# ---------------------------------------------------------------------------

def gate_preference(records):
    result = Result("6. preference pairs")
    supervised_ids = {
        rec["id"] for rec in records
        if record_type(rec) != "preference" and rec.get("id")
    }
    modes = {}
    for rec in records:
        if record_type(rec) != "preference":
            continue
        result.checked += 1
        rid = rec["id"]
        if rid in supervised_ids:
            result.fail(rid, "preference id collides with a supervised record; "
                             "this is the overlap that made the first DPO run a "
                             "zero-gradient no-op")
        if not rid.startswith("cosimopref_"):
            result.fail(rid, "preference record is outside the cosimopref_ namespace")
        chosen = (rec.get("chosen") or "").strip()
        rejected = (rec.get("rejected") or "").strip()
        if chosen == rejected:
            result.fail(rid, "chosen == rejected")
        if not rec.get("mode"):
            result.fail(rid, "no failure mode tagged")
        if not rec.get("pitfall"):
            result.fail(rid, "no pitfall described")
        modes[rec.get("mode")] = modes.get(rec.get("mode"), 0) + 1

    # The corpus is only as good as its coverage of failure modes: a preference
    # set that is all one mode teaches one thing.
    if result.checked and len(modes) < 2:
        result.warn(f"only {len(modes)} failure mode(s) present: {modes}")
    if modes:
        result.warn(f"modes: {dict(sorted(modes.items()))}")
    return result


# ---------------------------------------------------------------------------

def main():
    quick = "--quick" in sys.argv
    started = time.time()
    records = load_records()
    grouped = by_type(records)

    print(f"verify_all: {len(records)} records from {core.SHARDS_DIR}")
    print(f"  by type: {dict(sorted((k, len(v)) for k, v in grouped.items()))}\n")

    results = [
        gate_structure(records),
        gate_numeric(records),
        gate_format(records),
    ]
    if not quick:
        results.append(gate_implementation(records))
    results.append(gate_agentic(records))
    results.append(gate_preference(records))
    results.append(terms.run(records))

    length_result, length_stats = length_gate.run(records)
    results.append(length_result)

    if not quick:
        overlap_result, _ = suite_overlap.run(records)
        results.append(overlap_result)

    ok = True
    for result in results:
        ok = result.report() and ok

    print()
    length_gate.print_table(length_stats)

    print(f"\nverify_all: {'PASS' if ok else 'FAIL'} in {time.time() - started:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
