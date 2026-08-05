#!/usr/bin/env python3
"""PublishCosimoV2 dataset - assemble shards, compute stats, write dataset card.

Run from repo root:
    python3 scripts/publish_dataset.py
    python3 scripts/publish_dataset.py --output dataset_v2.json
"""
import json, glob, os, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_records(shard_dir):
    """Yield all records from a shard directory."""
    for fn in sorted(glob.glob(os.path.join(shard_dir, '*.jsonl'))):
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def compute_stats(records):
    """Compute dataset statistics."""
    stats = {
        'total_records': len(records),
        'by_program': Counter(),
        'by_record_type': Counter(),
        'by_question_type': Counter(),
        'by_topic': Counter(),
        'by_difficulty': Counter(),
        'length_stats': defaultdict(list),
        'by_subtopic': Counter(),
        'by_generator': Counter(),
        'has_preference_pair': 0,
        'has_code': 0,
        'has_trace': 0,
        'verified': 0,
    }

    for r in records:
        prog = r.get('program', 'unknown')
        rtype = r.get('record_type', 'exam')
        qtype = r.get('question_type', 'unknown')
        topic = r.get('topic', 'unknown')
        subtopic = r.get('subtopic', 'unknown')
        difficulty = r.get('difficulty', 'unknown')
        generator = r.get('metadata', {}).get('generator', 'unknown')

        stats['by_program'][prog] += 1
        stats['by_record_type'][rtype] += 1
        stats['by_question_type'][qtype] += 1
        stats['by_topic'][topic] += 1
        stats['by_difficulty'][difficulty] += 1
        stats['by_subtopic'][subtopic] += 1
        stats['by_generator'][generator] += 1

        # Token/character length of answer and question
        question_len = len(r.get('question', ''))
        answer_len = len(r.get('answer', ''))
        stats['length_stats']['question_chars'].append(question_len)
        stats['length_stats']['answer_chars'].append(answer_len)

        if 'reasoning_trace' in r and r['reasoning_trace']:
            trace_len = len(r['reasoning_trace'])
            stats['length_stats']['trace_chars'].append(trace_len)
            stats['has_trace'] += 1
        
        if r.get('preference_pair'):
            stats['has_preference_pair'] += 1
        
        if r.get('code'):
            stats['has_code'] += 1
        
        if r.get('verified'):
            stats['verified'] += 1

    # Compute length percentiles
    def percentile_stats(key, n=100):
        vals = sorted(stats['length_stats'][key])
        if not vals:
            return {}
        result = {
            'min': vals[0],
            'max': vals[-1],
            'mean': round(sum(vals) / len(vals), 2),
            'median': vals[len(vals) // 2],
        }
        for p in [10, 25, 50, 75, 90, 95, 99]:
            idx = int(len(vals) * p / 100)
            result[f'p{p}'] = vals[min(idx, len(vals) - 1)]
        return result

    stats['length_percentiles'] = {
        'question_chars': percentile_stats('question_chars'),
        'answer_chars': percentile_stats('answer_chars'),
        'trace_chars': percentile_stats('trace_chars'),
    }

    return stats


def write_dataset_card(stats, shard_count, output_path=None):
    """Write dataset card JSON."""
    card = {
        'name': 'cosimo_dapt_v2',
        'description': 'Cosimo fine-tuning dataset v2 - diverse record types across finance curricula',
        'version': '2.0.0',
        'stats': stats,
        'shard_count': shard_count,
        'record_type_summary': {
            rt: {
                'count': int(count),
                'percent': round(count / stats['total_records'] * 100, 1)
            }
            for rt, count in stats['by_record_type'].items()
        },
        'diversity': {
            'unique_subtopics': len(stats['by_subtopic']),
            'unique_programs': len(stats['by_program']),
            'unique_generators': len(stats['by_generator']),
            'unique_topics': len(stats['by_topic']),
        },
        'length_summary': stats['length_percentiles'],
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(card, f, indent=2)
        print(f"Dataset card written to {output_path}")
    
    return card


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    start = time.time()
    
    # Gather all records from all programs
    all_records = []
    shard_count = 0
    for fn in sorted(glob.glob('shards/**/*.jsonl', recursive=True)):
        shard_dir = os.path.dirname(fn)
        prog = os.path.basename(shard_dir)
        for r in get_records(shard_dir):
            r['_source_shard'] = fn
            all_records.append(r)
            shard_count += 1
    
    if not all_records:
        print("No records found in shards/. Run generate.py first.")
        sys.exit(1)
    
    programs_found = set()
    for r in all_records:
        shard_path = r.get('_source_shard', '')
        # _source_shard is the full path like 'shards/CFA_Level_I/shard_0000.jsonl'
        parts = shard_path.split('/')
        if len(parts) >= 2:
            programs_found.add(parts[1])
    
    print(f"Found {shard_count} shards across {len(programs_found)} programs")
    print(f"Assembled {len(all_records)} total records")

    # Compute statistics
    stats = compute_stats(all_records)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COSIMO DAPT V2 — Dataset Summary")
    print(f"{'='*60}")
    print(f"\nTotal records: {stats['total_records']}")
    print(f"\nBy program:")
    for prog, count in sorted(stats['by_program'].items(), key=lambda x: -x[1]):
        print(f"  {prog:15s} {count:6d} ({count/stats['total_records']*100:.1f}%)")
    
    print(f"\nBy record type:")
    for rtype, count in sorted(stats['by_record_type'].items(), key=lambda x: -x[1]):
        print(f"  {rtype:15s} {count:6d} ({count/stats['total_records']*100:.1f}%)")
    
    print(f"\nBy question type:")
    for qtype, count in sorted(stats['by_question_type'].items(), key=lambda x: -x[1]):
        print(f"  {qtype:25s} {count:6d}")
    
    print(f"\nDiversity:")
    print(f"  Subtopics:    {len(stats['by_subtopic']):>6d}")
    print(f"  Topics:       {len(stats['by_topic']):>6d}")
    print(f"  Generators:   {len(stats['by_generator']):>6d}")
    print(f"  Difficulty:   {', '.join(f'{k}={v}' for k,v in sorted(stats['by_difficulty'].items()))}")
    
    print(f"\nContent features:")
    print(f"  With preference pairs:  {stats['has_preference_pair']}")
    print(f"  With code:              {stats['has_code']}")
    print(f"  With trace:              {stats['has_trace']}")
    print(f"  Verified:                {stats['verified']}")
    
    # Length stats
    lp = stats['length_percentiles']
    for name in ['question_chars', 'answer_chars', 'trace_chars']:
        if name in lp and lp[name]:
            stats_str = ', '.join(f'p{p}={v}' for p,v in lp[name].items())
            print(f"  {name:18s} {stats_str}")
    
    output_path = output_path or f'dataset_v2_{int(time.time())}.json'
    card = write_dataset_card(stats, shard_count, output_path)
    
    print(f"\nElapsed: {time.time() - start:.1f}s")
    return card


if __name__ == '__main__':
    main()
