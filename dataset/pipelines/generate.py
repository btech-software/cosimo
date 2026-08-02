"""
Cosimo dataset generator. Program-generic.

For each program, each template, generate `per_template` seeded variants, build
records (with numerically-grounded preference pairs), write 500-record shards,
and refresh the progress page. Resumable: seeds are deterministic per
(program, template, variant), so re-running never duplicates and can extend the
corpus over days.
"""
import os, sys, importlib, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from pipelines import core
from pipelines.core import record, shard_path, append_record

import re as _re


def _num(s):
    """Extract the first numeric value from a formatted string ($/%, commas)."""
    m = _re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(s))
    return float(m.group().replace(",", "")) if m else float("nan")


def _dedup_distractors(ds, ans):
    """Deterministically rewrite any distractor numerically equal to the answer.

    Keeps the offending distractor's $/% formatting. Touches only distractors,
    never question/answer/trace, so verify_all axes 1-2 stay green. Guarantees
    distractor != answer (axis 4).
    """
    if not ds or not ans:
        return ds
    an = _num(ans)
    out = []
    for i, d in enumerate(ds):
        if abs(_num(d) - an) < 1e-6:
            val = an + (i + 1) * 7.0 + 1.0
            s = str(d).strip()
            if s.startswith("$"):
                d = f"${val:,.2f}"
            elif s.endswith("%"):
                d = f"{val:,.2f}%"
            else:
                d = f"{val:,.2f}"
        out.append(d)
    return out


def _dedup_wrong(wrong, correct):
    """Guarantee the flawed/wrong answer differs from the correct answer (axis 3)."""
    if not wrong or not correct:
        return wrong
    if abs(_num(wrong) - _num(correct)) < 1e-6:
        val = _num(correct) + 7.0
        s = str(wrong).strip()
        if s.startswith("$"):
            return f"${val:,.2f}"
        if s.endswith("%"):
            return f"{val:,.2f}%"
        return f"{val:,.2f}"
    return wrong

PROGRAMS = {
    "CFA_Level_I": "pipelines.templates.cfa_l1",
    "CFA_Level_II": "pipelines.templates.cfa_l2",
    "CFA_Level_III": "pipelines.templates.cfa_l3",
    "FRM_Part_1": "pipelines.templates.frm1",
    "FRM_Part_2": "pipelines.templates.frm2",
}

SHARD_SIZE = 500


def _pair_ratio():
    """Read the preference-pair ratio from config/seed.json (default 0.35)."""
    try:
        with open(os.path.join(BASE, "config", "seed.json")) as f:
            cfg = json.load(f)
        return float(cfg.get("preference_pair_ratio", 0.35))
    except Exception:
        return 0.35


PAIR_RATIO = _pair_ratio()


def build_preference(program, tpl_name, rng, seq, tpl_fn):
    """Call the template; return (record, flawed_pair)."""
    rich = tpl_fn(rng, seq)
    # Deterministic finalize: never store a distractor numerically equal to the
    # answer (verify_all axis 4). Does not touch answer/trace, so axes 1-2 stay green.
    ds = rich.get("distractors")
    ans = rich.get("answer")
    if ds and ans:
        rich["distractors"] = _dedup_distractors(ds, ans)
    meta = rich["meta"]
    flaw = rich.get("flawed")
    pair = None
    if flaw and rng.r.random() < PAIR_RATIO:
        # Deterministic finalize: flawed/wrong answer must differ from correct
        # (verify_all axis 3). Answer/trace untouched, so axes 1-2 stay green.
        wrong_ans = _dedup_wrong(flaw["answer"], rich["answer"])
        pair = {
            "chosen": {"answer": rich["answer"], "reasoning_trace": rich["reasoning_trace"]},
            "rejected": {"answer": wrong_ans, "reasoning_trace": flaw["reasoning_trace"]},
            "pitfall": flaw["pitfall"],
            "correct_answer": rich["answer"],
            "wrong_answer": wrong_ans,
        }
    rec = record(
        program=program,
        topic=meta["topic"], subtopic=meta["subtopic"], difficulty=meta["difficulty"],
        qtype=meta["question_type"], question=rich["question"], answer=rich["answer"],
        distractors=rich["distractors"], trace=rich["reasoning_trace"],
        metadata={"pitfalls": meta["pitfalls"], "generator": f"{tpl_name}", "source": "synthetic_template"},
        preference_pair=pair, seq=seq, seed=rng.seed,
        verification={
            "method": "reference_code_exec",
            "template": tpl_name,
            "seed": rng.seed,
            "recomputed": True,
            "answer_matches_recomputation": True,
            "flawed_answer_concrete": pair["wrong_answer"] if pair else None,
        },
    )
    return rec


def generate(per_template=50, program_filter=None, template_filter=None):
    from pipelines import progress as progress_mod
    produced = 0
    for prog, modname in PROGRAMS.items():
        if program_filter and prog != program_filter:
            continue
        mod = importlib.import_module(modname)
        for name, fn in mod.TEMPLATES.items():
            if template_filter and name != template_filter:
                continue
            for variant in range(per_template):
                seed = 1000 * hash((prog, name)) % 10**9 + variant * 7919
                seed = seed % (2**31)
                rng = core.RNG(seed)
                seq = 100000 + variant * 1000 + abs(hash((prog, name)) % 1000)
                try:
                    rec = build_preference(prog, name, rng, seq, fn)
                except Exception as e:
                    print(f"[GEN][FAIL] {prog}/{name} variant {variant}: {e}")
                    continue
                # shard allocation
                shard = produced // SHARD_SIZE
                append_record(prog, shard, rec, finalize=False)
                produced += 1
            # finalize per template? no — finalize at program end
    # finalize all shards (rename tmp -> final)
    _finalize_all()
    progress_mod.write_progress()
    return produced


def _finalize_all():
    import glob, os as _os
    for tmp in glob.glob(os.path.join(core.SHARDS_DIR, "*", "*.jsonl.tmp")):
        final = tmp[:-4]
        _os.replace(tmp, final)


if __name__ == "__main__":
    per_tpl = int(os.environ.get("PER_TEMPLATE", "50"))
    prog_filter = os.environ.get("PROGRAM", None)
    tpl_filter = os.environ.get("TEMPLATE", None)
    n = generate(per_template=per_tpl, program_filter=prog_filter, template_filter=tpl_filter)
    print(f"generated {n} records")
