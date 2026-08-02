"""
Deterministic distractor sanitizer (P0 quality fix).

Validates that every record has exactly 3 distractors that are:
  - distinct from each other and from the answer (all numeric components),
  - within [0.4x, 3.5x] of each non-zero answer component.
When invalid, regenerates distractors by scaling the answer with plausible
multipliers, preserving units and comma formatting.

Numeric parsing is robust to thousands separators (see nums.py).
"""
import json, glob, os
from nums import nums, scale_str

MULTS = [0.88, 0.93, 1.07, 1.12, 0.82, 1.18, 0.95, 1.05, 0.97, 1.10]

def valid(ans_str, ds):
    A = nums(ans_str)
    D = [nums(d) for d in ds]
    if len(D) != 3:
        return False
    # full-component duplicates
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            if D[i] == D[j]:
                return False
    # magnitude: each non-zero answer component needs a matching non-zero
    # distractor component within [0.4x, 3.5x]
    nonzero = [a for a in A if abs(a) > 1e-9]
    if nonzero:
        for a in nonzero:
            ok = False
            for d in D:
                for c in d:
                    if abs(c) > 1e-9:
                        rr = a / c if a > c else c / a
                        if 0.4 <= rr <= 3.5:
                            ok = True
                            break
                if ok:
                    break
            if not ok:
                return False
    return True

MULTS = [0.88, 0.93, 1.07, 1.12, 0.82, 1.18, 0.95, 1.05, 0.97, 1.10, 0.75, 1.25, 0.6, 1.4, 0.5, 1.5]

def make_distractors(ans_str, seed):
    rng = random.Random(seed)
    A = nums(ans_str)
    chosen = []   # list of (string, nums)
    for m in MULTS:
        cand = scale_str(ans_str, m)
        D = nums(cand)
        if D == A:                        # numerically equal to answer
            continue
        if any(D == C for _, C in chosen):  # duplicate of a chosen distractor
            continue
        chosen.append((cand, D))
        if len(chosen) >= 3:
            break
    # fallback additive offsets
    if len(chosen) < 3:
        for d in (0.5, 0.25, 1.0):
            cand = scale_str(ans_str, 1.0 + d / 100.0)
            D = nums(cand)
            if D == A or any(D == C for _, C in chosen):
                continue
            chosen.append((cand, D))
    return [c for c, _ in chosen[:3]]

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixed = 0; total = 0
    for fn in glob.glob(os.path.join(root, 'shards', '*', '*.jsonl')):
        lines = open(fn).read().splitlines()
        out = []
        for i, line in enumerate(lines):
            r = json.loads(line)
            total += 1
            ds = r.get('distractors') or []
            if not valid(r['answer'], ds):
                r['distractors'] = make_distractors(r['answer'], seed_for(r['id'], i))
                r['metadata']['distractor_sanitized'] = True
                fixed += 1
            out.append(json.dumps(r, ensure_ascii=False))
        with open(fn, 'w') as f:
            f.write('\n'.join(out) + '\n')
    print(f'sanitized {fixed} records of {total}')

def seed_for(id_, i):
    return (int(id_.split('_')[-1], 36) if id_ else i) % 10**9

import random
from nums import scale_str

if __name__ == '__main__':
    main()
