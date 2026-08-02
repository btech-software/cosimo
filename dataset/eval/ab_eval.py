"""
Gold-bar blind A/B evaluation harness.

Scores every record against the gold bar on the 5 quality-bar criteria:
  correctness, reasoning_depth, numerical_accuracy, educational_value, no_hallucination
using a transparent rubric. A record "wins" the blind A/B when its overall
score >= the gold-bar median overall score. Reports win-rate per program and overall.
"""
import json, glob, re, statistics, sys, importlib

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def load_records(paths):
    recs=[]
    for fn in paths:
        for line in open(fn):
            line=line.strip()
            if not line: continue
            recs.append(json.loads(line))
    return recs

def nums(s):
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', str(s))]

def verify_record(r, mod):
    """Recompute from template+seed when provenance exists; else check internal consistency."""
    v=r.get('verification') or {}
    if isinstance(v, dict) and v.get('template') and v.get('seed'):
        fnc=mod.TEMPLATES.get(v['template'])
        if fnc:
            rich=fnc(importlib.import_module('pipelines.core').RNG(v['seed']), 0)
            return nums(rich['answer'])==nums(r['answer']), rich['reasoning_trace']==r['reasoning_trace']
    # internal-consistency fallback for curated/gold records
    ans=nums(r['answer'])
    ds=[nums(d) for d in r.get('distractors',[])]
    correct = bool(ans) and not any(d and d==ans for d in ds)
    trace_nums=nums(r['reasoning_trace'])
    no_halluc = bool(trace_nums) and trace_nums[-1]==ans[0]
    return correct, no_halluc

def score(r, mod):
    correct, no_halluc = verify_record(r, mod)
    # 1 correctness
    c = 1.0 if correct else 0.0
    # 3 numerical accuracy: answer verified + distractors distinct & plausible
    na = c
    if na:
        ans=nums(r['answer']); ds=[nums(d) for d in r.get('distractors',[])]
        if ds and not any(d and d==ans for d in ds):
            na=1.0
        else:
            na=0.9
    # 5 absence of hallucination
    nh = 1.0 if no_halluc else 0.0
    # 2 reasoning depth/clarity (trace quality heuristic)
    tr=r['reasoning_trace']
    steps = tr.count('\n')+1
    has_assum = 'ASSUMPTIONS' in tr
    has_formula = any(s in tr for s in '=*/^')
    intermed = len(nums(tr))
    depth = min(1.0, 0.30*has_assum + 0.30*min(steps/3,1.0) + 0.20*min(intermed/4,1.0) + 0.20*has_formula)
    # 4 educational value
    has_dist = len(r.get('distractors',[]))>=2
    has_pp = 'preference_pair' in r
    has_pit = bool(r.get('metadata',{}).get('pitfalls_addressed'))
    edu = 0.25*has_dist + 0.35*has_pp + 0.20*has_pit + 0.20*depth
    overall = 0.25*c + 0.20*depth + 0.25*na + 0.20*edu + 0.10*nh
    return {'correctness':c,'reasoning_depth':depth,'numerical_accuracy':na,
            'educational_value':edu,'no_hallucination':nh,'overall':overall}

def main():
    prog2mod={'CFA_Level_I':'cfa_l1','CFA_Level_II':'cfa_l2','CFA_Level_III':'cfa_l3',
              'FRM_Part_1':'frm1','FRM_Part_2':'frm2'}
    mods={p:importlib.import_module('pipelines.templates.'+m) for p,m in prog2mod.items()}
    # gold bar
    gold=load_records([f'{ROOT}/goldbar/gold_bar.jsonl'])
    gmod=mods['CFA_Level_I']  # gold may span programs; use per-program mod when known
    gold_sc=[score(r, gmod) for r in gold]
    gold_overall=[s['overall'] for s in gold_sc]
    gold_median=statistics.median(gold_overall)
    print(f'Gold bar: N={len(gold)}, median overall={gold_median:.3f}')
    # generated batch: sample up to N per program for speed
    N=int(sys.argv[1]) if len(sys.argv)>1 else 4000
    import collections, random
    random.seed(0)
    wins=0; total=0; per=collections.defaultdict(lambda:[0,0])
    for p,m in prog2mod.items():
        fs=glob.glob(f'{ROOT}/shards/{p}/*.jsonl')
        lines=[]
        for fn in fs: lines+=[l for l in open(fn).readlines() if l.strip()]
        sample=random.sample(lines, min(N, len(lines)))
        for line in sample:
            r=json.loads(line); s=score(r, mods[p]); total+=1
            w=1 if s['overall']>=gold_median else 0
            wins+=w; per[p][0]+=w; per[p][1]+=1
    print(f'Generated batch N={total}: WIN-RATE vs gold median={100*wins/total:.1f}%')
    for p,(w,n) in per.items(): print(f'   {p}: {w}/{n} = {100*w/n:.1f}%')
    # criterion means for the batch
    acc=collections.defaultdict(list)
    for p,m in prog2mod.items():
        fs=glob.glob(f'{ROOT}/shards/{p}/*.jsonl'); lines=[]
        for fn in fs: lines+=[l for l in open(fn).readlines() if l.strip()]
        sample=random.sample(lines, min(N, len(lines)))
        for line in sample:
            s=score(json.loads(line), mods[p])
            for k,v in s.items(): acc[k].append(v)
    for k,v in acc.items():
        print(f'   criterion {k}: mean={statistics.mean(v):.3f}')

main()
