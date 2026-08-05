#!/usr/bin/env python3
"""Phase A verification: one variant per generator, into a scratch directory.

Proves the generation pipeline is whole without a bulk run. Bulk generation to
the 200k target happens separately; nothing here needs more than one record per
generator to fail when something is broken.

    python3 scripts/smoke_generate.py

Checks, in order:
  1. every registered generator produces exactly one record (zero [GEN][FAIL])
  2. all five record_type values are present
  3. each record carries the fields FORMAT.md requires for its type
  4. verify_all's numeric recomputation passes on the exam subset
  5. every agentic record renders through the harness chat template with its
     tool calls and tool responses intact
  6. seeds are process-stable, so ids reproduce across runs
  7. a second run generates nothing new (idempotent)

Exits non-zero on any failure.
"""
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import contextlib
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(BASE, ".smoke", "shards")

# Must be set before pipelines.core is imported: it reads the env var at import
# time to decide where shards land, and the real corpus must never be touched.
os.environ["COSIMO_SHARDS_DIR"] = SCRATCH
sys.path.insert(0, BASE)

from pipelines import core  # noqa: E402
from pipelines import generate as gen  # noqa: E402

# Per FORMAT.md: the fields that make each record type useful. A record missing
# one of these validates as JSON but teaches nothing.
REQUIRED = {
    "exam": ("question", "answer", "reasoning_trace"),
    "analysis": ("question", "answer"),
    "abstention": ("question", "answer"),
    "agentic": ("question", "answer", "tool_schemas", "conversation"),
    "implementation": ("question", "answer", "code"),
}

failures = []
notes = []


def check(condition, message):
    (notes if condition else failures).append(message)
    print(("  ok   " if condition else "  FAIL ") + message)


def registered_generators():
    """Every (record_type, name) the generation driver will call."""
    seen = set()
    for modname in gen.PROGRAMS.values():
        for name in importlib.import_module(modname).TEMPLATES:
            seen.add(("exam", name))
    for type_map in gen.NEW_RECORD_TYPES.values():
        for record_type, modname in type_map.items():
            mod = importlib.import_module(modname)
            for name in getattr(mod, "TEMPLATES", {}):
                seen.add((record_type, name))
    return seen


def load_records():
    records = []
    for prog in sorted(os.listdir(SCRATCH)):
        d = os.path.join(SCRATCH, prog)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(d, fn)) as f:
                    records.extend(json.loads(line) for line in f if line.strip())
    return records


def run_generation():
    """One variant per generator. Returns (produced, fail_lines)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        produced = gen.generate(per_template=1)
    out = buf.getvalue()
    return produced, [ln for ln in out.splitlines() if "[GEN][FAIL]" in ln]


def check_seed_stability():
    """_variant_key must not depend on PYTHONHASHSEED.

    Run in a subprocess with a different hash seed: the builtin hash() this
    replaced was salted, so seeds -- and every record id -- changed per process.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from pipelines.generate import _variant_key;"
        "print(_variant_key(100000, 'CFA_Level_I', 'tvm_annuity_fv', 0))" % BASE
    )
    outs = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        outs.add(
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env=env
            ).stdout.strip()
        )
    return len(outs) == 1, outs


def check_agentic_rendering(records):
    """Agentic records must render through the harness's own chat template.

    The generators emit `role: "tool_result"`; configs/chat_template.jinja tests
    `role == "tool"`. A mismatch renders the tool output as nothing, so the
    supervised target silently loses its tool results -- invisible in the JSONL
    and only detectable by rendering.
    """
    template_path = os.path.join(
        os.path.dirname(BASE), "jobs", "fine-tune", "configs", "chat_template.jinja"
    )
    try:
        import jinja2
    except ImportError:
        return None, "skipped (jinja2 unavailable)"
    if not os.path.exists(template_path):
        return None, f"skipped ({template_path} not found)"

    # Rendered with jinja2 directly rather than through transformers'
    # apply_chat_template: the template is the artifact that decides whether a
    # tool turn survives, and it is the only part of the harness this check needs.
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    # transformers replaces Jinja's `tojson` with plain json.dumps. The stock filter
    # HTML-escapes and returns Markup, which then escapes every string concatenated
    # with it -- so `<tool_call>` would render as `&lt;tool_call&gt;` and this check
    # would fail on correct data. Match the runtime the template is served under.
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False)
    with open(template_path) as f:
        template = env.from_string(f.read())

    agentic = [r for r in records if r.get("record_type") == "agentic"]
    bad = []
    for rec in agentic:
        messages = [{"role": "system", "content": "smoke"}] + rec["conversation"]
        try:
            text = template.render(
                messages=messages,
                tools=rec.get("tool_schemas"),
                add_generation_prompt=False,
                eos_token="<|endoftext|>",
            )
        except Exception as exc:
            bad.append(f"{rec['id']}: {type(exc).__name__}: {exc}")
            continue
        if "<tool_call>" not in text:
            bad.append(f"{rec['id']}: rendered without <tool_call>")
        elif any(t.get("role") == "tool" for t in rec["conversation"]) and (
            "<tool_response>" not in text
        ):
            bad.append(f"{rec['id']}: tool results vanished from the rendering")
        elif rec.get("tool_schemas") and "<|tool|>" not in text:
            bad.append(f"{rec['id']}: tool schemas were dropped")
    return not bad, (bad[:5] if bad else f"{len(agentic)} agentic records render")


def main():
    shutil.rmtree(os.path.dirname(SCRATCH), ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)

    expected = registered_generators()
    print(f"smoke: {len(expected)} registered generators -> {SCRATCH}\n")

    produced, fail_lines = run_generation()
    records = load_records()

    print("1. every generator produced a record")
    for line in fail_lines[:10]:
        print("     " + line)
    check(not fail_lines, f"{len(fail_lines)} generator failures (expected 0)")
    check(
        produced == len(records),
        f"produced {produced} == {len(records)} rows on disk",
    )

    print("\n2. record types present")
    by_type = Counter(r.get("record_type", "exam") for r in records)
    for rtype in sorted(REQUIRED):
        check(by_type.get(rtype, 0) > 0, f"{rtype}: {by_type.get(rtype, 0)} rows")

    print("\n3. required fields per record type")
    missing = Counter()
    for rec in records:
        rtype = rec.get("record_type", "exam")
        for field in REQUIRED.get(rtype, ()):
            if not rec.get(field):
                missing[f"{rtype}.{field}"] += 1
    for key, count in sorted(missing.items()):
        print(f"     {key}: {count} rows missing")
    check(not missing, "no record is missing a required field")
    abstentions = [r for r in records if r.get("record_type") == "abstention"]
    check(
        all(r["metadata"].get("defect") for r in abstentions),
        f"all {len(abstentions)} abstention rows carry metadata.defect",
    )

    print("\n4. numeric recomputation (exam subset)")
    proc = subprocess.run(
        [sys.executable, os.path.join(BASE, "verification", "verify_all.py")],
        capture_output=True, text=True, env={**os.environ, "COSIMO_SHARDS_DIR": SCRATCH},
    )
    for line in proc.stdout.strip().splitlines():
        print("     " + line)
    check(proc.returncode == 0, "verify_all gate passes")

    print("\n5. agentic records render through the harness template")
    ok, detail = check_agentic_rendering(records)
    if ok is None:
        print(f"     {detail}")
        notes.append(f"agentic rendering {detail}")
    else:
        if isinstance(detail, list):
            for line in detail:
                print("     " + line)
        check(ok, "agentic conversations render with calls and responses intact")

    print("\n6. seeds are process-stable")
    stable, outs = check_seed_stability()
    check(stable, f"_variant_key is PYTHONHASHSEED-independent ({len(outs)} distinct)")

    print("\n7. idempotency")
    again, _ = run_generation()
    check(again == 0, f"second run produced {again} new records (expected 0)")

    # Reported, not enforced: Phase A's contract is one variant per generator, and
    # at one variant this is always 1.0. It is measured here because it is the gate
    # on bulk generation -- a generator whose question does not vary with the seed
    # contributes exactly one unique row however many variants are asked for, and
    # re-randomising a fixed stem is the memorisation failure the corpus exists to
    # avoid. Fixing it is Phase B (parameter variety).
    print("\n8. parameter variety across seeds (Phase B gate, reported only)")
    for record_type, modname in sorted(
        {rt: mn for tm in gen.NEW_RECORD_TYPES.values() for rt, mn in tm.items()}.items()
    ):
        mod = importlib.import_module(modname)
        questions, total = set(), 0
        for fn in mod.TEMPLATES.values():
            for v in range(5):
                with contextlib.redirect_stdout(io.StringIO()):
                    d = fn(core.RNG(1000 + v * 7919), 200000 + v * 1000)
                questions.add(d.get("question") or d.get("docstring") or "")
                total += 1
        ratio = len(questions) / total
        flag = "ok  " if ratio > 0.9 else "WARN"
        print(
            f"  {flag} {record_type:15s} {len(questions):4d} distinct questions "
            f"of {total:4d} draws ({ratio:.0%})"
        )
        if ratio <= 0.9:
            notes.append(
                f"{record_type}: only {len(questions)} unique questions across "
                f"{len(mod.TEMPLATES)} generators - blocks bulk generation"
            )

    print(f"\n{'=' * 60}")
    print(f"{len(records)} records from {len(expected)} generators")
    print(f"  by type: {dict(sorted(by_type.items()))}")
    if failures:
        print(f"\nSMOKE: FAIL ({len(failures)} checks)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
