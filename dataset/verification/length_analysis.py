"""
Length distribution analyzer for Cosimo dataset v2.

Checks that generated records have the required length distributions:
- exam: reasoning_trace should vary in length (not all collapsed to ~120 tokens)
- analysis: answer should typically be 800-2000 tokens
- implementation: code section should be 400-1200 tokens

Run:
    python3 verification/length_analysis.py
"""

import json
import os
import sys
import statistics
from collections import defaultdict

def analyze_shard(shard_path):
    """Analyze a single shard file for record types and lengths."""
    types_found = defaultdict(list)
    
    with open(shard_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rt = rec.get("record_type", "exam")
                
                # Count tokens approx by word count
                question_tokens = len(rec.get("question", "").split())
                answer_tokens = len(rec.get("answer", "").split())
                
                if rt == "exam":
                    trace_tokens = len(rec.get("reasoning_trace", "").split())
                    types_found[rt].append({
                        "question_tokens": question_tokens,
                        "answer_tokens": answer_tokens,
                        "trace_tokens": trace_tokens,
                    })
                elif rt == "analysis":
                    types_found[rt].append({
                        "answer_tokens": answer_tokens,
                    })
                elif rt == "abstention":
                    types_found[rt].append({
                        "answer_tokens": answer_tokens,
                    })
                elif rt == "agentic":
                    conv = rec.get("conversation", [])
                    turns = len(conv)
                    types_found[rt].append({
                        "conversation_turns": turns,
                        "answer_tokens": answer_tokens,
                    })
                elif rt == "implementation":
                    code = rec.get("code", "")
                    code_tokens = len(code.split()) if code else 0
                    types_found[rt].append({
                        "answer_tokens": answer_tokens,
                        "code_tokens": code_tokens,
                    })
            except (json.JSONDecodeError, KeyError):
                pass
    
    return dict(types_found)


def stats(values, label="Values"):
    """Compute statistics for a list of numeric values."""
    if not values:
        return {label: "N/A"}
    n = len(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    p95 = sorted(values)[int(n * 0.95)] if n > 1 else values[0]
    p05 = sorted(values)[int(n * 0.05)] if n > 1 else values[0]
    return {
        "count": n,
        "mean": f"{mean:.1f}",
        "median": f"{median:.1f}",
        "p05": f"{p05:.1f}",
        "p95": f"{p95:.1f}",
        "min": f"{min(values):.1f}",
        "max": f"{max(values):.1f}",
    }


def main():
    shard_dir = os.environ.get("SHARD_DIR", "shards")
    
    if not os.path.isdir(shard_dir):
        print(f"[WARN] No shards directory at {shard_dir}")
        print("Run `python3 pipelines/generate.py` first to generate data.")
        return
    
    all_types = defaultdict(list)
    total_records = 0
    
    for program_dir in sorted(os.listdir(shard_dir)):
        program_path = os.path.join(shard_dir, program_dir)
        if not os.path.isdir(program_path):
            continue
        for shard_file in sorted(os.listdir(program_path)):
            if not shard_file.endswith(".jsonl"):
                continue
            shard_path = os.path.join(program_path, shard_file)
            shard_stats = analyze_shard(shard_path)
            
            for rec_type, recs in shard_stats.items():
                all_types[rec_type].extend(recs)
            total_records += sum(len(v) for v in shard_stats.values())
    
    if not all_types:
        print("No records found. Generate data first.")
        return
    
    print("=" * 60)
    print(f"  Cosimo Length Distribution Analysis")
    print(f"  Total records analyzed: {total_records}")
    print(f"  Shard directory: {shard_dir}")
    print("=" * 60)
    
    for rec_type, recs in sorted(all_types.items()):
        n = len(recs)
        print(f"\n--- {rec_type.upper()} ({n} records) ---")
        
        if rec_type == "exam":
            traces = [r["trace_tokens"] for r in recs]
            questions = [r["question_tokens"] for r in recs]
            answers = [r["answer_tokens"] for r in recs]
            
            print(f"  Question tokens : {stats(questions, 'Question')}")
            print(f"  Trace tokens    : {stats(traces, 'Reasoning Trace')}")
            print(f"  Answer tokens   : {stats(answers, 'Answer')}")
            print(f"  > 800 token traces: {sum(1 for t in traces if t > 800)}")
            print(f"  > 400 token traces: {sum(1 for t in traces if t > 400)}")
            
        elif rec_type == "analysis":
            answers = [r["answer_tokens"] for r in recs]
            print(f"  Answer tokens   : {stats(answers, 'Answer')}")
            print(f"  > 800 tokens    : {sum(1 for a in answers if a > 800)}")
            print(f"  < 800 tokens    : {sum(1 for a in answers if a < 800)}")
            
        elif rec_type == "abstention":
            answers = [r["answer_tokens"] for r in recs]
            print(f"  Answer tokens   : {stats(answers, 'Answer')}")
            print(f"  Target: 50-300 tokens (calibration)")
            
        elif rec_type == "agentic":
            turns = [r["conversation_turns"] for r in recs]
            answers = [r["answer_tokens"] for r in recs]
            print(f"  Conversation turns : {stats(turns, 'Turns')}")
            print(f"  Answer tokens      : {stats(answers, 'Answer')}")
            
        elif rec_type == "implementation":
            answers = [r["answer_tokens"] for r in recs]
            codes = [r["code_tokens"] for r in recs]
            print(f"  Answer tokens     : {stats(answers, 'Answer')}")
            print(f"  Code tokens       : {stats(codes, 'Code')}")


if __name__ == "__main__":
    main()
