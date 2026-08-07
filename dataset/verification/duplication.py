"""Blocking gate: distinct-question ratio, measured on the corpus as shipped.

Every other gate is per-record. That is why a 200,742-row corpus in which 52% of
questions were duplicates passed all nine of them: each individual row was
well-formed, reproducible and correctly typed. The defect only exists *between*
rows, so nothing could see it.

77,500 abstention rows carried 310 distinct questions. That is the memorisation
failure the whole corpus exists to avoid, at a worse ratio than v1's 71 stems x
1,000 variants -- and it shipped past a green board.

Two thresholds, because they fail differently:

  * per record type -- catches a whole type collapsing, as abstention did.
  * per generator -- catches one dead generator hiding inside a healthy type.
    A type can sit at 95% while one of its sixty generators emits a single
    question 250 times.

`smoke_generate.py` check 8 measures the same property on *generators*, before
generation. This measures it on *records*, after. Both are needed: the smoke
check cannot see how many variants a generator was actually asked for, and this
one cannot run until a corpus exists.

Run standalone:
    python3 verification/duplication.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates import Result, load_records, record_type  # noqa: E402

# Below this share of distinct questions, a type is re-randomising a fixed stem
# rather than teaching a subject.
TYPE_FLOOR = 0.90

# A single generator is allowed to be a little worse than its type -- some
# subjects have genuinely few distinct framings -- but not by much.
GENERATOR_FLOOR = 0.80

# Generators with fewer rows than this are not measured: a stem with 3 rows and
# 2 distinct questions is at 67% and says nothing.
MIN_ROWS = 25


# A supervised answer may not be mostly text it shares with other answers.
# 8,084 analysis records once carried an identical 819-token block appended to
# 117 tokens of real content -- 88% boilerplate, one distinct block across every
# record. It raised the type's measured length from p50 117 to p50 936 and the
# length gate passed it, because the gate counts tokens and cannot tell padding
# from reasoning. The block was topically wrong too: a risk-budgeting question
# received filler about revenue growth and capital expenditures.
BOILERPLATE_SHARE_CEILING = 0.40
BOILERPLATE_MIN_TOKENS = 40


def question_of(rec):
    return rec.get("question") or rec.get("prompt") or rec.get("docstring") or ""


def answer_of(rec):
    return rec.get("answer") or rec.get("chosen") or ""


def _paragraphs(text):
    return [p.strip() for p in str(text).split("\n\n") if len(p.split()) >= 12]


def boilerplate_share(records):
    """Fraction of the mean answer made of paragraphs shared across records.

    Works on paragraphs rather than whole answers: padding is appended to
    otherwise distinct content, so an exact-duplicate check on the full answer
    sees nothing.
    """
    counts = collections.Counter()
    per_record = []
    for rec in records:
        paras = _paragraphs(answer_of(rec))
        per_record.append(paras)
        counts.update(set(paras))
    if not per_record:
        return 0.0, None
    threshold = max(2, len(records) // 10)
    shared, total, worst = 0, 0, collections.Counter()
    for paras in per_record:
        for p in paras:
            n = len(p.split())
            total += n
            if counts[p] >= threshold and n >= BOILERPLATE_MIN_TOKENS:
                shared += n
                worst[p] += 1
    if not total:
        return 0.0, None
    top = worst.most_common(1)[0] if worst else None
    return shared / total, top


def run(records=None, type_floor=TYPE_FLOOR, generator_floor=GENERATOR_FLOOR):
    result = Result("question duplication")
    records = load_records() if records is None else records
    if not records:
        return result, {}

    by_type = collections.defaultdict(collections.Counter)
    by_generator = collections.defaultdict(collections.Counter)
    for rec in records:
        result.checked += 1
        question = question_of(rec)
        rtype = record_type(rec)
        by_type[rtype][question] += 1
        generator = (rec.get("metadata") or {}).get("generator") or "?"
        by_generator[(rtype, generator)][question] += 1

    stats = {}
    for rtype, questions in sorted(by_type.items()):
        rows = sum(questions.values())
        distinct = len(questions)
        ratio = distinct / rows
        stats[rtype] = {
            "rows": rows,
            "distinct": distinct,
            "ratio": round(ratio, 4),
            "max_repeats": max(questions.values()),
        }
        if ratio < type_floor:
            result.fail(
                rtype,
                f"{distinct} distinct questions across {rows} rows "
                f"({ratio:.1%}, floor {type_floor:.0%}); the most repeated "
                f"question appears {max(questions.values())} times. This is "
                f"re-randomising a fixed stem, not coverage.",
            )

    # Shared-paragraph padding, per type. A high token count means nothing if the
    # tokens are the same tokens in every record.
    by_type_records = collections.defaultdict(list)
    for rec in records:
        by_type_records[record_type(rec)].append(rec)
    for rtype, recs in sorted(by_type_records.items()):
        share, top = boilerplate_share(recs)
        stats.setdefault(rtype, {})["boilerplate_share"] = round(share, 4)
        if share > BOILERPLATE_SHARE_CEILING:
            sample = (top[0][:70] + "...") if top else "?"
            result.fail(
                rtype,
                f"{share:.0%} of the mean answer is boilerplate shared across "
                f"records (ceiling {BOILERPLATE_SHARE_CEILING:.0%}); the most "
                f"repeated block appears in {top[1] if top else 0} records: "
                f"{sample!r}. Padding inflates the length gate without adding "
                f"reasoning.",
            )

    offenders = []
    for (rtype, generator), questions in by_generator.items():
        rows = sum(questions.values())
        if rows < MIN_ROWS:
            continue
        ratio = len(questions) / rows
        if ratio < generator_floor:
            offenders.append((ratio, rtype, generator, len(questions), rows))
    for ratio, rtype, generator, distinct, rows in sorted(offenders)[:20]:
        result.fail(
            f"{rtype}/{generator}",
            f"{distinct} distinct questions across {rows} rows ({ratio:.0%}, "
            f"floor {generator_floor:.0%})",
        )
    if len(offenders) > 20:
        result.warn(f"{len(offenders) - 20} further generators below the floor")

    stats["_generators_below_floor"] = len(offenders)
    return result, stats


def print_table(stats):
    print(f"  {'record_type':<16}{'rows':>9}{'distinct':>10}{'ratio':>8}{'max rep':>9}")
    for rtype, s in sorted(stats.items()):
        if rtype.startswith("_"):
            continue
        print(f"  {rtype:<16}{s['rows']:>9}{s['distinct']:>10}"
              f"{s['ratio']:>7.1%}{s['max_repeats']:>9}")


def main():
    print("=== Gate: question duplication ===")
    result, stats = run()
    if stats:
        print_table(stats)
    ok = result.report()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
