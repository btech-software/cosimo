"""Full verification of the synthetic dataset (51k records).
Checks per record: (1) final answer recomputable from stored template+seed,
(2) reasoning trace byte-identical to deterministic recomputation (no hallucinated
intermediates), (3) preference-pair wrong answer is concrete and != correct,
(4) no distractor numerically equals the correct answer.
"""
import json, glob, re, importlib, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines import core
from nums import nums
prog2mod={'CFA_Level_I':'cfa_l1','CFA_Level_II':'cfa_l2','CFA_Level_III':'cfa_l3','FRM_Part_1':'frm1','FRM_Part_2':'frm2'}
mods={p:importlib.import_module('pipelines.templates.'+m) for p,m in prog2mod.items()}
def main():
    t0=time.time(); total=0
    bad_ans=bad_trace=bad_pair=bad_dist=0
    root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p,m in prog2mod.items():
        mod=mods[p]
        for fn in glob.glob(os.path.join(root, 'shards', p, '*.jsonl')):
            for line in open(fn):
                r=json.loads(line); total+=1
                v=r['verification']
                if not isinstance(v,dict) or not v.get('template'):
                    bad_ans+=1; continue
                fnc=mod.TEMPLATES.get(v['template'])
                if not fnc: bad_ans+=1; continue
                rich=fnc(core.RNG(v['seed']),0)
                if nums(rich['answer'])!=nums(r['answer']): bad_ans+=1
                if rich['reasoning_trace']!=r['reasoning_trace']: bad_trace+=1
                pp=r.get('preference_pair')
                if pp:
                    w=pp.get('wrong_answer'); c=pp.get('correct_answer')
                    if not w or w==c: bad_pair+=1
                ans=nums(r['answer']); ds=[nums(d) for d in r.get('distractors',[])]
                for d in ds:
                    if d and d==ans: bad_dist+=1; break
    print(f'verify_all: N={total} in {time.time()-t0:.1f}s')
    print(f'  answer_reproducible: {(total-bad_ans)/total*100:.2f}%  failures={bad_ans}')
    print(f'  trace_consistency:   {(total-bad_trace)/total*100:.2f}%  failures={bad_trace}')
    print(f'  dpo_pair_concrete:   {(total-bad_pair)/total*100:.2f}%  failures={bad_pair}')
    print(f'  distractor_clean:    {(total-bad_dist)/total*100:.2f}%  failures={bad_dist}')
    ok = (bad_ans+bad_trace+bad_pair+bad_dist)==0
    print('  GATE:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1
main()
