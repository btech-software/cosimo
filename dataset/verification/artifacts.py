"""Blocking gate: no template artifact may reach a supervised target.

The failure this exists to catch shipped in 4,524 rows across 26 generators. A
trace read, verbatim:

    Step 1. | This is {risk_type}: {explanation}.\\n
    Step 2. | Distinguishing categories:\\n

Two separate defects, both from a string that was never formatted:

  * `{risk_type}` -- an f-string placeholder in a literal that had no `f` prefix,
    so the braces survived into the training data.
  * a literal backslash-n -- the two characters, not a newline.

Neither is caught by any other gate. Every one of those rows passed structure,
recomputation, format, terminology and length: the record is well-formed, the
number is reproducible, and the text is simply wrong. It is pure poison -- the
model is being trained to emit `{risk_type}` and `\\n` as literal output.

Also catches the residue of partially-applied string formatting: a stray `%s`
or `%0.3f` that no argument replaced.

Run standalone:
    python3 verification/artifacts.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates import Result, load_records, supervised_text  # noqa: E402

# {name} or {name:spec} or {name!r} -- an unrendered f-string placeholder.
# Requires a lower-case identifier start so JSON, LaTeX and set notation in
# legitimate prose are not swept up.
PLACEHOLDER = re.compile(r"\{[a-z_][a-z0-9_]*(?:[.\[][^{}]*)?(?:![rsa])?(?::[^{}]*)?\}")

# The two characters backslash-n, not a newline.
LITERAL_ESCAPE = re.compile(r"\\[nt]")

# printf-style conversions left unsubstituted by a missing % operand.
# The space flag is deliberately NOT accepted: "% g" is a legitimate printf form
# but in this corpus it is always prose -- "a 5% growth rate" -- and allowing it
# flagged thousands of valid records.
PERCENT_LEFTOVER = re.compile(r"%[-+#0]*\d*(?:\.\d+)?[sdifgeEG](?![a-zA-Z])")

FIELDS = ("question", "answer", "reasoning_trace", "prompt", "chosen", "rejected",
          "code", "docstring")

CHECKS = (
    ("unrendered placeholder", PLACEHOLDER),
    ("literal escape sequence", LITERAL_ESCAPE),
    ("unsubstituted %-conversion", PERCENT_LEFTOVER),
)


def scan_record(rec):
    """Every artifact found in one record, as (kind, sample) pairs."""
    found = []
    blob = " ".join(str(rec.get(f) or "") for f in FIELDS)
    blob += " " + supervised_text(rec)
    for kind, pattern in CHECKS:
        match = pattern.search(blob)
        if match:
            found.append((kind, match.group(0)))
    return found


def run(records=None):
    result = Result("template artifacts")
    records = load_records() if records is None else records
    for rec in records:
        result.checked += 1
        for kind, sample in scan_record(rec):
            result.fail(
                rec.get("id", "?"),
                f"{kind} {sample!r} reached a supervised target "
                f"(generator {(rec.get('metadata') or {}).get('generator')!r})",
            )
    return result


def main():
    print("=== Gate: template artifacts ===")
    result = run()
    ok = result.report()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
