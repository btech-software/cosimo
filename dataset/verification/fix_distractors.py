"""Deterministic in-place fix: any distractor numerically equal to the correct answer is
perturbed to a distinct, plausible wrong value (preserving formatting). Answer + trace
are untouched, so verification integrity is preserved."""
import json, glob, re, importlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines import core
ROOT="."
def nums(s): return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', str(s))]
prog2mod={'CFA_Level_I':'cfa_l1','CFA_Level_II':'cfa_l2','CFA_Level_III':'cfa_l3','FRM_Part_1':'frm1','FRM_Part_2':'frm2'}
mods={p:importlib.import_module('pipelines.templates.'+m) for p,m in prog2mod.items()}

def fmt(ans_str, val):
    if ans_str.lstrip().startswith('$'): return f'${val:,.2f}'
    if ans_str.rstrip().endswith('%'): return f'{val:.2f}%'
    if '.' in ans_str: return f'{val:.2f}'
    return f'{val:.0f}'

def perturb(ans_val, seed, idx, others):
    for mult in (1.05, 1.08, 1.03, 0.95, 0.97, 1.10):
        p=round(ans_val*mult, 2)
        if p!=ans_val and all(abs(p-o)>1e-6 for o in others): return p
    # fallback additive
    for d in (1, 2, 3, 0.5, 0.25, 5):
        p=round(ans_val+d, 2)
        if p!=ans_val and all(abs(p-o)>1e-6 for o in others): return p
    return None

fixed=0; scanned=0; still_bad=0
for p,m in prog2mod.items():
    mod=mods[p]
    for fn in glob.glob(f'{ROOT}/shards/{p}/*.jsonl'):
        out=[]
        for line in open(fn):
            r=json.loads(line); scanned+=1
            v=r['verification']
            if not isinstance(v,dict) or not v.get('template'):
                out.append(line.rstrip('\n')); continue
            fnc=mod.TEMPLATES[v['template']]
            rich=fnc(core.RNG(v['seed']),0)
            if nums(rich['answer'])!=nums(r['answer']):
                out.append(line.rstrip('\n')); continue
            ans=nums(r['answer']); ans_val=ans[0]
            ds=[nums(d) for d in r.get('distractors',[])]
            bad=[i for i,d in enumerate(ds) if d and abs(d[0]-ans_val)<1e-6]
            if not bad:
                out.append(line.rstrip('\n')); continue
            others=[d[0] for i,d in enumerate(ds) if i not in bad and d]
            newp=perturb(ans_val, v['seed'], bad[0], others)
            if newp is None:
                still_bad+=1; out.append(line.rstrip('\n')); continue
            for i in bad:
                r['distractors'][i]=fmt(r['answer'], newp)
                r['metadata']['distractor_fixed']=True
            fixed+=1
            out.append(json.dumps(r, ensure_ascii=False))
        with open(fn,'w') as f: f.write('\n'.join(out)+'\n')
print(f'scanned={scanned} fixed_records={fixed} still_bad={still_bad}')
