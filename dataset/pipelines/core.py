"""
Cosimo dataset pipeline core.

Shared helpers for building, verifying, and serializing synthetic financial
reasoning examples. Every record's numerical answer is COMPUTED by code, never
hallucinated; the reasoning trace is written to reference those computed
intermediates, so traces are numerically consistent by construction. Preference
pairs are numerically-grounded: the rejected trace uses a wrong formula variant
and therefore lands on a concrete (wrong) number, internally consistent with its
own arithmetic.

Schema (JSONL, one record per line) is defined by FORMAT.md.
"""
import hashlib, json, math, os, random, re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARDS_DIR = os.path.join(BASE_DIR, "shards")
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")


def sha_of(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]


def make_id(program, seq, payload):
    return f"cosimo_{program}_{seq:06d}_{sha_of(payload)}"


def fmt(x, dp=2):
    return f"{x:,.{dp}f}"


def pct(x, dp=2):
    return f"{x*100:,.{dp}f}%"


def load_taxonomy():
    with open(os.path.join(BASE_DIR, "taxonomy", "taxonomy.json")) as f:
        return json.load(f)


class RNG:
    """Deterministic RNG so a (program, template, seed) tuple is reproducible."""
    def __init__(self, seed):
        self.seed = seed
        self.r = random.Random(seed)

    def uniform(self, lo, hi, dp=4):
        return round(self.r.uniform(lo, hi), dp)

    def choice(self, seq):
        return self.r.choice(seq)

    def choices(self, seq, k):
        return self.r.choices(seq, k=k)

    def randint(self, a, b):
        return self.r.randint(a, b)

    def sample(self, seq, k):
        return self.r.sample(seq, k)

    def shuffle(self, seq):
        self.r.shuffle(seq)


def record(program, topic, subtopic, difficulty, qtype, question, answer,
           distractors, trace, verified=True, verification=None, metadata=None,
           preference_pair=None, seq=0, seed=None):
    payload = [program, topic, subtopic, question, answer]
    rid = make_id(program, seq, payload)
    meta = {
        "topic": topic, "subtopic": subtopic, "difficulty": difficulty,
        "question_type": qtype,
        "pitfalls_addressed": metadata.get("pitfalls", []) if metadata else [],
        "source": metadata.get("source", "synthetic_template") if metadata else "synthetic_template",
        "seed": seed,
        "generator": metadata.get("generator", "cosimo_template") if metadata else "cosimo_template",
        "generator_version": "1.0.0",
    }
    rec = {
        "id": rid, "program": program, "topic": topic, "subtopic": subtopic,
        "difficulty": difficulty, "question_type": qtype,
        "question": question, "answer": answer, "distractors": distractors,
        "reasoning_trace": trace,
        "verified": verified,
        "verification": verification or {},
        "metadata": meta,
    }
    if preference_pair:
        rec["preference_pair"] = preference_pair
    return rec


def shard_path(program, shard_idx):
    prog_dir = os.path.join(SHARDS_DIR, program)
    os.makedirs(prog_dir, exist_ok=True)
    return os.path.join(prog_dir, f"{program}_shard_{shard_idx:04d}.jsonl")


def append_record(program, shard_idx, rec, finalize=False):
    """Append one record to the current shard file (temp, then rename on finalize)."""
    prog_dir = os.path.join(SHARDS_DIR, program)
    os.makedirs(prog_dir, exist_ok=True)
    fname = f"{program}_shard_{shard_idx:04d}.jsonl"
    tmp = os.path.join(prog_dir, fname + ".tmp")
    final = os.path.join(prog_dir, fname)
    with open(tmp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    if finalize:
        os.replace(tmp, final)
    return final


def shard_counts(program=None):
    """Return {program: {shard: count}} and total from finalized shard files."""
    counts = {}
    total = 0
    base = SHARDS_DIR
    if not os.path.isdir(base):
        return counts, total
    prog_dirs = [program] if program else sorted(os.listdir(base))
    for prog in prog_dirs:
        d = os.path.join(base, prog)
        if not os.path.isdir(d):
            continue
        c = {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".jsonl"):
                n = sum(1 for _ in open(os.path.join(d, fn)))
                c[fn] = n
                total += n
        if c:
            counts[prog] = c
    return counts, total
