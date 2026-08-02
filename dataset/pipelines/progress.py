"""
Writes live progress page: progress.md + progress.html.
Shows totals per program, per-topic coverage heat-map, verification pass/fail,
preference-pair coverage, and freshness timestamp.
"""
import os, sys, json, datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from pipelines.core import SHARDS_DIR, PROGRESS_DIR, shard_counts

PROGRAMS = {
    "CFA_Level_I": "Level I", "CFA_Level_II": "Level II", "CFA_Level_III": "Level III",
    "FRM_Part_1": "FRM Part 1", "FRM_Part_2": "FRM Part 2",
}

def _coverage():
    """topic -> {subtopic: count} across all shards, plus pair/total counts."""
    cov = {}
    pairs = 0
    total = 0
    gens = set()
    for prog in os.listdir(SHARDS_DIR):
        d = os.path.join(SHARDS_DIR, prog)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(d, fn)) as f:
                for line in f:
                    r = json.loads(line)
                    total += 1
                    gens.add(r["metadata"]["generator"])
                    if r.get("preference_pair"):
                        pairs += 1
                    t = r["metadata"]["topic"]
                    st = r["metadata"]["subtopic"]
                    cov.setdefault(t, {}).setdefault(st, 0)
                    cov[t][st] += 1
    return cov, pairs, total, gens

def _stats():
    counts, total = shard_counts()
    return counts, total

def write_progress(verify=None, errors=None):
    cov, pairs, total, gens = _coverage()
    counts, total2 = _stats()
    total = total2 or total
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_shards = sum(len(v) for v in counts.values()) if counts else 0
    md = []
    md.append("# Cosimo Synthetic Dataset — Live Progress\n")
    md.append(f"*Updated*: {now}\n")
    md.append(f"**Total verified records**: {total}  \n")
    pair_pct = 100*pairs/total if total else 0
    md.append(f"**Preference pairs**: {pairs} ({pair_pct:.1f}% of records)  \n")
    md.append(f"**Shards**: {n_shards} (500 records each)  \n")
    md.append(f"**Templates**: {len(gens)} across {len(os.listdir(SHARDS_DIR))} programs\n")
    if verify:
        md.append("\n## Verification (reference_code_exec)\n")
        all_ok = sum(v["ok"] for v in verify.values())
        all_fail = sum(v["fail"] for v in verify.values())
        md.append(f"**PASS**: {all_ok}  **FAIL**: {all_fail}\n")
        if errors:
            md.append("Sample failures:\n")
            for e in errors[:10]:
                md.append(f"- `{e}`\n")
    md.append("\n## Records per program\n")
    md.append("| Program | Records | Shards |\n|---|---|---|\n")
    for prog, label in PROGRAMS.items():
        c = counts.get(prog, {})
        recs = sum(c.values())
        nsh = len(c)
        md.append(f"| {label} | {recs} | {nsh} |\n")
    md.append("\n## Coverage heat-map (topic × subtopic)\n")
    md.append("| Topic | Subtopic | Records |\n|---|---|---|\n")
    for t, subs in sorted(cov.items()):
        for st, n in sorted(subs.items()):
            md.append(f"| {t} | {st} | {n} |\n")
    md.append("\n## Gap analysis (target 50k)\n")
    md.append(f"Progress vs 50k target: **{100*total/50000:.1f}%**\n")
    md.append(f"Remaining: {max(50000-total,0)} records. Run `PER_TEMPLATE=N python3 -m pipelines.generate` to extend.\n")
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    with open(os.path.join(PROGRESS_DIR, "progress.md"), "w") as f:
        f.write("".join(md))
    _write_html(counts, total, pairs, cov, verify, now)
    return total

def _write_html(counts, total, pairs, cov, verify, now):
    rows = []
    rows.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Cosimo Progress</title>")
    rows.append("<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:900px}table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:4px 10px;text-align:left}.ok{color:#0a0}.bad{color:#c00}.bar{background:#0a0;height:12px}</style></head><body>")
    rows.append(f"<h1>Cosimo Synthetic Dataset — Live Progress</h1>")
    rows.append(f"<p><b>Updated</b>: {now}</p>")
    pair_pct2 = 100*pairs/total if total else 0
    rows.append(f"<p><b>Total verified records</b>: {total} &nbsp; <b>Preference pairs</b>: {pairs} ({pair_pct2:.1f}%)</p>")
    rows.append(f"<div class='bar' style='width:{100*total/50000:.1f}%'></div>")
    rows.append(f"<p>Progress vs 50k target: <b>{100*total/50000:.1f}%</b> ({total}/50000)</p>")
    rows.append("<h2>Records per program</h2><table><tr><th>Program</th><th>Records</th><th>Shards</th></tr>")
    for prog, label in PROGRAMS.items():
        c = counts.get(prog, {})
        rows.append(f"<tr><td>{label}</td><td>{sum(c.values())}</td><td>{len(c)}</td></tr>")
    rows.append("</table>")
    if verify:
        all_ok = sum(v["ok"] for v in verify.values())
        all_fail = sum(v["fail"] for v in verify.values())
        rows.append(f"<h2>Verification (reference_code_exec)</h2><p class='ok'>PASS: {all_ok}</p><p class='bad'>FAIL: {all_fail}</p>")
    rows.append("<h2>Coverage heat-map</h2><table><tr><th>Topic</th><th>Subtopic</th><th>Records</th></tr>")
    for t, subs in sorted(cov.items()):
        for st, n in sorted(subs.items()):
            rows.append(f"<tr><td>{t}</td><td>{st}</td><td>{n}</td></tr>")
    rows.append("</table></body></html>")
    with open(os.path.join(PROGRESS_DIR, "progress.html"), "w") as f:
        f.write("".join(rows))

def main():
    write_progress()

if __name__ == "__main__":
    main()
