"""Diversity/novelty report for the synthetic dataset (batch-level axis of the quality bar)."""
import json, glob, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main():
    tpl=collections.Counter(); qtype=collections.Counter(); cells=collections.Counter()
    total=0
    for p in glob.glob('shards/*'):
        prog=p.split('/')[-1]
        for fn in glob.glob(p+'/*.jsonl'):
            for line in open(fn):
                r=json.loads(line); total+=1
                tpl[(prog,r['verification']['template'])]+=1
                qtype[(prog,r['metadata']['question_type'])]+=1
                cells[(prog,r['metadata']['topic'],r['metadata']['subtopic'])]+=1
    print(f'total records: {total}')
    print(f'distinct question stems (templates): {len(tpl)}')
    print(f'records per stem: min={min(tpl.values())} max={max(tpl.values())} avg={total//len(tpl)}')
    print(f'question-type distribution:')
    for (p,qt),n in sorted(qtype.items()): print(f'   {p}/{qt}: {n}')
    print(f'distinct topic-subtopic cells: {len(cells)}')
    print(f'NOTE: structural novelty is bounded by {len(tpl)} stems; within-stem records differ only in numbers.')
main()
