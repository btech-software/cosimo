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
import hashlib, json, math, os, random, re, shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# COSIMO_SHARDS_DIR lets the smoke run write to a scratch directory instead of the
# real corpus. Without it a verification run would append to shards/.
SHARDS_DIR = os.environ.get("COSIMO_SHARDS_DIR") or os.path.join(BASE_DIR, "shards")
# Keep the progress page beside whichever shards it describes, so a smoke run
# cannot overwrite the real one with its own counts.
PROGRESS_DIR = (
    os.path.join(os.path.dirname(SHARDS_DIR), "progress")
    if os.environ.get("COSIMO_SHARDS_DIR")
    else os.path.join(BASE_DIR, "progress")
)


def sha_of(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]


def stable_hash(*parts):
    """Process-stable integer digest of the parts.

    Python's builtin hash() is salted by PYTHONHASHSEED, so a seed derived from
    it changes between runs and every record gets a new id. Generation is only
    deterministic, resumable and idempotent if the (program, template, variant)
    key maps to the same seed in every process.
    """
    return int(sha_of(*parts), 16)


def make_id(program, seq, payload):
    return f"cosimo_{program}_{seq:06d}_{sha_of(payload)}"


def fmt(x, dp=2):
    return f"{x:,.{dp}f}"


def pct(x, dp=2):
    return f"{x*100:,.{dp}f}%"


def load_taxonomy():
    with open(os.path.join(BASE_DIR, "taxonomy", "taxonomy.json")) as f:
        return json.load(f)


def load_seed_config():
    try:
        with open(os.path.join(BASE_DIR, "config", "seed.json")) as f:
            return json.load(f)
    except Exception:
        return {}


# Which generation round produced a record. Stamped on every record so a later
# session can ask whether round-N data moved round-N metrics, and so a round can
# be held out as a unit. Bump it in config/seed.json before each new round.
ROUND = int(load_seed_config().get("round", 1))

# Asking-context pools, shared by the record types whose question text is
# otherwise a fixed literal. A generator that hardcodes its question contributes
# exactly one unique row however many variants are requested, which is the
# memorisation failure the corpus exists to avoid -- 77,500 abstention rows once
# shipped carrying 310 distinct questions.
DESKS = (
    "a UK DB pension scheme", "a multi-family office", "a university endowment",
    "a UCITS long-only fund", "an insurance general account", "a sovereign wealth sleeve",
    "a fund-of-funds mandate", "a corporate treasury", "a discretionary wealth book",
    "a systematic macro sleeve", "a credit opportunities fund", "a listed infrastructure fund",
    "a charity investment committee", "a family trust", "a private credit vehicle",
    "an emerging-markets debt mandate", "a liability-matching portfolio",
    "a market-neutral equity book", "a defined-contribution default fund",
    "a multi-asset growth mandate",
)

ASK_OPENERS = (
    "Quick one before the IC meeting", "The trustees have asked", "Client just called",
    "Prepping for the quarterly review", "Risk flagged this", "The board wants a view",
    "Following up from yesterday", "Need this for the pack", "Portfolio manager asked",
    "Compliance raised a query", "The consultant is pushing back", "Ahead of the rebalance",
    "For the manager selection paper", "Auditors have queried this",
    "The CIO wants this by Friday", "New mandate onboarding",
)

ASK_CLOSERS = (
    "Can you take a look?", "What's your read?", "How would you approach it?",
    "Where would you start?", "Thoughts?", "Can you work this up?",
    "What do you need from me?", "How should I frame it?", "Give me your view.",
)


_BOUND_RNG = None


def bind_rng(fn):
    """Wrap a generator so its module's record helper can reach the seeded RNG.

    The helpers that build a record (`_pref`, `_rec`, `_impl`) do not receive the
    RNG, and threading it would mean editing every generator. Bound per call and
    restored afterwards, so nested generators cannot read each other's RNG.
    """
    def wrapped(rng, seq):
        global _BOUND_RNG
        previous = _BOUND_RNG
        _BOUND_RNG = rng
        try:
            return fn(rng, seq)
        finally:
            _BOUND_RNG = previous

    wrapped.__name__ = getattr(fn, "__name__", "generator")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


def bound_rng():
    """The RNG of the generator currently executing, or None outside one."""
    return _BOUND_RNG


def scenario_clause(rng, base):
    """Wrap a fixed question in a drawn asking context.

    Roughly 20 x 16 x 9 x 4 shapes before the base text, which is enough that a
    generator with a hardcoded question still yields distinct rows at the variant
    counts bulk generation uses. The base text -- and therefore whatever the
    record is meant to teach -- is untouched.
    """
    base = str(base).strip()
    desk = rng.choice(list(DESKS))
    opener = rng.choice(list(ASK_OPENERS))
    closer = rng.choice(list(ASK_CLOSERS))
    size = rng.randint(8, 940)
    shape = rng.randint(0, 3)
    if shape == 0:
        return f"{opener}: {base} This is for {desk}. {closer}"
    if shape == 1:
        return f"{opener} — we run {desk} (about ${size}m). {base}"
    if shape == 2:
        return f"{base} Context: {desk}, roughly ${size}m. {closer}"
    return f"[{desk}] {opener}. {base} {closer}"


TRACE_STYLES = ("assumptions_steps", "prose", "table", "backward")


def render_trace(rng, assumptions, steps, conclusion="", style=None):
    """Render a computed reasoning trace in one of several shapes.

    `steps` is a list of (label, text); the numbers in them are already computed
    by the caller, so only the presentation varies here and the recomputation gate
    is unaffected. The style is drawn from the variant's seeded RNG, which makes
    it reproducible -- verify_all re-derives the identical string from the seed.

    The first corpus emitted `ASSUMPTIONS:` + `Step N.` on every single record, and
    the model learned that being Cosimo means answering in four steps. A model
    cannot learn that structure is a *choice* if it only ever sees one structure.
    """
    style = style or rng.choice(list(TRACE_STYLES))
    assumptions = list(assumptions or [])
    steps = list(steps or [])

    if style == "assumptions_steps":
        head = f"ASSUMPTIONS: {'; '.join(assumptions)}.\n" if assumptions else ""
        body = "\n".join(f"Step {i}. {text}" for i, (_, text) in enumerate(steps, 1))
        return head + body + (f"\n{conclusion}" if conclusion else "")

    if style == "prose":
        head = (
            "Taking " + ", ".join(assumptions) + " as given, " if assumptions else ""
        )
        body = " ".join(
            f"{text[0].lower()}{text[1:]}" if i and text else text
            for i, (_, text) in enumerate(steps)
        )
        return (head + body).strip() + (f" {conclusion}" if conclusion else "")

    if style == "table":
        head = f"Assumptions: {'; '.join(assumptions)}.\n\n" if assumptions else ""
        width = max((len(label) for label, _ in steps), default=8)
        rows = "\n".join(f"{label:<{width}} | {text}" for label, text in steps)
        return head + rows + (f"\n\n{conclusion}" if conclusion else "")

    # backward: state the result, then justify it in reverse
    lead = conclusion or (steps[-1][1] if steps else "")
    rest = "\n".join(
        f"- {text}" for _, text in reversed(steps[:-1] if conclusion == "" else steps)
    )
    tail = f"\nThis holds under {'; '.join(assumptions)}." if assumptions else ""
    return f"{lead}\n\nWorking backwards:\n{rest}{tail}"


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
           preference_pair=None, seq=0, seed=None, record_type=None, **extra_fields):
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
        "round": ROUND,
    }
    # Carry through generator-specific metadata (defect, call_depth, language,
    # source_template, ...). The canonical keys above win; nothing is overwritten,
    # so a generator cannot silently redefine topic/subtopic/generator.
    for k, v in (metadata or {}).items():
        if k not in meta and k != "pitfalls":
            meta[k] = v
    rec = {
        "id": rid, "program": program, "topic": topic, "subtopic": subtopic,
        "difficulty": difficulty, "question_type": qtype,
        "question": question, "answer": answer, "distractors": distractors,
        "reasoning_trace": trace,
        "verified": verified,
        "verification": verification or {},
        "metadata": meta,
    }
    if record_type:
        rec["record_type"] = record_type
    for k, v in extra_fields.items():
        if v is not None:
            rec[k] = v
    if preference_pair:
        rec["preference_pair"] = preference_pair
    return rec


def shard_path(program, shard_idx):
    prog_dir = os.path.join(SHARDS_DIR, program)
    os.makedirs(prog_dir, exist_ok=True)
    return os.path.join(prog_dir, f"{program}_shard_{shard_idx:04d}.jsonl")


def append_record(program, shard_idx, rec, finalize=False):
    """Append one record to the current shard file (temp, then rename on finalize).

    When the shard already has a finalized file and no write is in flight, the temp
    is seeded from it first. Without that, finalizing a resumed run renames a temp
    holding only the newly generated rows over a full shard and destroys it.
    """
    prog_dir = os.path.join(SHARDS_DIR, program)
    os.makedirs(prog_dir, exist_ok=True)
    fname = f"{program}_shard_{shard_idx:04d}.jsonl"
    tmp = os.path.join(prog_dir, fname + ".tmp")
    final = os.path.join(prog_dir, fname)
    if not os.path.exists(tmp) and os.path.exists(final):
        shutil.copyfile(final, tmp)
    with open(tmp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    if finalize:
        os.replace(tmp, final)
    return final


def existing_ids(program=None):
    """Every record id already committed to a finalized shard.

    Generation skips ids in this set, which is what makes a re-run idempotent
    rather than duplicating: seeds are stable per (program, template, variant),
    so a repeated key reproduces a record that is already on disk.
    """
    ids = set()
    if not os.path.isdir(SHARDS_DIR):
        return ids
    for prog in ([program] if program else sorted(os.listdir(SHARDS_DIR))):
        d = os.path.join(SHARDS_DIR, prog)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(d, fn)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.add(json.loads(line)["id"])
    return ids


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
