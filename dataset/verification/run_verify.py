"""
Independent verification harness (reference_code_exec).

For every record in a shard, re-execute the generating template from its stored
seed + template name, recompute the answer and the flawed answer, and compare to
what was persisted. Confirms every stored numerical answer is reproducible and
internally consistent. Also validates preference-pair concreteness. Writes stats
and refreshes the progress page.
"""
import os, sys, json, glob, importlib, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from pipelines import core
from pipelines.core import RNG, SHARDS_DIR

PROGRAMS = {
    "CFA_Level_I": "pipelines.templates.cfa_l1",
    "CFA_Level_II": "pipelines.templates.cfa_l2",
    "CFA_Level_III": "pipelines.templates.cfa_l3",
    "FRM_Part_1": "pipelines.templates.frm1",
    "FRM_Part_2": "pipelines.templates.frm2",
}

def _norm(x):
    return re.findall(r"[-+]?\d*\.?\d+", str(x))


def _num(x):
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(x))
    return float(m.group().replace(",", "")) if m else float("nan")


def _finalize_wrong(raw_flaw, correct):
    """Reproduce the generator's deterministic finalize: ensure the stored
    wrong answer differs numerically from the correct answer."""
    if abs(_num(raw_flaw) - _num(correct)) < 1e-6:
        val = _num(correct) + 7.0
        s = str(raw_flaw).strip()
        if s.startswith("$"):
            return f"${val:,.2f}"
        if s.endswith("%"):
            return f"{val:,.2f}%"
        return f"{val:,.2f}"
    return raw_flaw

def verify_shard(path, prog):
    mod = importlib.import_module(PROGRAMS[prog])
    ok = fail = 0
    errors = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            v = rec["verification"]
            tpl = v.get("template")
            seed = v.get("seed")
            fn = mod.TEMPLATES.get(tpl)
            if not fn:
                fail += 1; errors.append(f"{rec['id']}: unknown template {tpl}"); continue
            rng = RNG(seed)
            rich = fn(rng, 0)
            if _norm(rich["answer"]) != _norm(rec["answer"]):
                fail += 1; errors.append(f"{rec['id']}: answer mismatch ({rec['answer']} vs {rich['answer']})"); continue
            flaw = rich.get("flawed")
            pp = rec.get("preference_pair")
            if flaw and pp:
                final = _finalize_wrong(flaw["answer"], rec["answer"])
                if _norm(final) != _norm(pp["wrong_answer"]):
                    fail += 1; errors.append(f"{rec['id']}: flawed answer mismatch"); continue
                if _norm(final) == _norm(rec["answer"]):
                    fail += 1; errors.append(f"{rec['id']}: flawed == correct (bad pair)"); continue
            ok += 1
    return ok, fail, errors

def verify_all():
    from pipelines import progress as progress_mod
    summary = {}
    all_ok = all_fail = 0
    all_errs = []
    for prog in sorted(os.listdir(SHARDS_DIR)):
        d = os.path.join(SHARDS_DIR, prog)
        if not os.path.isdir(d):
            continue
        summary[prog] = {"ok": 0, "fail": 0}
        for fn in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            ok, fail, errs = verify_shard(fn, prog)
            summary[prog]["ok"] += ok
            summary[prog]["fail"] += fail
            all_ok += ok; all_fail += fail
            all_errs.extend(errs[:5])
    print(f"VERIFY: ok={all_ok} fail={all_fail}")
    for e in all_errs[:20]:
        print("  ERR:", e)
    progress_mod.write_progress(verify=summary, errors=all_errs[:20])
    return all_ok, all_fail

if __name__ == "__main__":
    verify_all()
