"""Gate 2: terminology, checked at the level of collocations.

The failure this exists to catch is "Durbin-Watson duration", which the first
tuned checkpoint invented when asked about hedging a convexity mismatch. Note
what makes it hard: **the eponym is real and the concept is real; it is the
pairing that is fabricated.** A token-level vocabulary check passes it, because
every token is in the vocabulary.

So the gate is built the other way round. For each eponym it knows, it records
the concept nouns that eponym may legitimately attach to. `Durbin-Watson` may be
followed by `statistic` or `test`; it may not be followed by `duration`.

Two deliberate asymmetries:

* A **known eponym with a disallowed concept blocks the record.** That is a
  fabrication with high confidence.
* An **unknown eponym is reported, never blocked.** The vocabulary is incomplete,
  and a real term the list has not heard of is indistinguishable from an invented
  one -- the same reasoning that makes `09_assistant_eval.py`'s `unknown_terms` a
  triage aid rather than a threshold.

Records flagged `contains_intentional_fabrication` are exempt: the `invented_term`
preference pairs carry a fabricated collocation on their `rejected` side on
purpose, and blocking them would delete the signal that teaches the model not to
do it.

Run standalone:
    python3 verification/terms.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates import Result, load_records, supervised_text  # noqa: E402

# eponym -> the concept nouns it may legitimately attach to (lower-cased).
EPONYM_COLLOCATIONS = {
    "durbin-watson": {"statistic", "test", "d", "d-statistic", "bounds"},
    "macaulay": {"duration"},
    "black-scholes": {"model", "formula", "equation", "price", "value", "pricing", "framework", "assumptions", "call", "put", "delta", "merton", "volatility", "vega", "gamma", "theta", "rho", "greeks", "implied"},
    "black-litterman": {"model", "framework", "approach", "posterior", "weights", "prior", "returns", "views", "equilibrium"},
    "sharpe": {"ratio", "measure"},
    "sortino": {"ratio"},
    "treynor": {"ratio", "measure", "index", "black"},
    "jensen": {"alpha", "measure"},
    "calmar": {"ratio"},
    "vasicek": {"model", "process", "formula", "correlation", "distribution"},
    "merton": {"model", "framework", "approach", "distance", "structural"},
    "hull-white": {"model", "process"},
    "cox-ingersoll-ross": {"model", "process"},
    "gordon": {"growth", "model", "formula"},
    "modigliani-miller": {"theorem", "proposition", "propositions"},
    "fama-french": {"model", "factors", "factor", "three-factor", "five-factor", "loading", "loadings", "regression", "alpha", "premium", "exposure", "exposures"},
    "carhart": {"model", "four-factor", "momentum", "factor", "factors", "loading", "loadings", "alpha", "regression"},
    "brinson": {"attribution", "model", "method", "decomposition", "fachler", "hood"},
    "carino": {"linking", "smoothing", "algorithm", "factor", "adjustment", "coefficient"},
    "hodrick-prescott": {"filter"},
    "engle-granger": {"test", "procedure", "cointegration"},
    "dickey-fuller": {"test", "statistic"},
    "jarque-bera": {"test", "statistic"},
    "breusch-pagan": {"test", "statistic"},
    "newey-west": {"standard", "estimator", "correction", "errors", "covariance", "adjustment"},
    "granger": {"causality", "test"},
    "kalman": {"filter", "gain", "smoothing", "smoother"},
    "girsanov": {"theorem"},
    "feynman-kac": {"theorem", "formula"},
    "markowitz": {"model", "framework", "optimization", "frontier", "portfolio", "variance", "efficient", "mean-variance"},
    "tobin": {"separation", "theorem", "q"},
    "shapley": {"value", "values"},
    "nash": {"equilibrium", "equilibria", "bargaining", "solution"},
    "wilson": {"score", "interval"},
    "mcnemar": {"test", "statistic"},
    "altman": {"z-score", "z", "score", "model"},
    "kupiec": {"test", "pof"},
    "christoffersen": {"test"},
    "cornish-fisher": {"expansion", "approximation"},
    # Compound eponyms that are themselves fabrications when welded together.
    "sharpe-sortino": set(),
    "durbin-sharpe": set(),
}

_STOPWORDS = {"the", "a", "an", "of", "for", "is", "was", "and", "or", "to", "in",
              "on", "with", "that", "this", "at", "by", "as", "its", "it", "are",
              "be", "been", "which", "from", "we", "you", "i"}

# The words that make a collocation a *claim* rather than prose continuing.
# Explicit on purpose: flagging every word after an eponym would drown the signal.
CONCEPT_NOUNS = {
    "duration", "convexity", "ratio", "statistic", "test", "model", "formula",
    "equation", "theorem", "filter", "expansion", "process", "alpha", "beta",
    "gamma", "delta", "vega", "theta", "rho", "premium", "spread", "curve",
    "frontier", "equilibrium", "value", "score", "index", "attribution",
    "invariance", "coefficient", "estimator", "approximation", "measure",
    "volatility", "variance", "covariance", "correlation", "hedge", "parity",
    "arbitrage", "yield", "return", "risk", "exposure", "sensitivity", "effect",
    "adjustment", "factor", "decomposition", "identity", "bound", "limit",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")

# Fields that carry model-facing prose on any record type.
_TEXT_FIELDS = ("question", "prompt", "answer", "reasoning_trace")


def _following_concept(lowered, eponym, start):
    """The first substantive word after an eponym occurrence."""
    tail = lowered[start + len(eponym):]
    for match in _WORD.finditer(tail[:80]):
        word = match.group().lower().strip("-'")
        if word and word not in _STOPWORDS:
            return word
    return None


def scan_text(text):
    """Known eponyms attached to a disallowed concept noun."""
    violations = []
    lowered = str(text).lower()
    for eponym, allowed in EPONYM_COLLOCATIONS.items():
        for match in re.finditer(re.escape(eponym), lowered):
            concept = _following_concept(lowered, eponym, match.start())
            if not allowed:
                # An empty allowed set means the compound eponym is itself
                # fabricated -- there is no legitimate continuation, so the
                # following word does not matter.
                violations.append((eponym, concept or "<any>"))
                continue
            if concept is None:
                continue
            if concept in CONCEPT_NOUNS and concept not in allowed:
                violations.append((eponym, concept))
    return violations


def record_text(rec):
    blob = " ".join(str(rec.get(f) or "") for f in _TEXT_FIELDS)
    return blob + " " + supervised_text(rec)


def run(records=None):
    result = Result("terminology")
    records = load_records() if records is None else records
    for rec in records:
        if rec.get("contains_intentional_fabrication"):
            continue
        result.checked += 1
        seen = set()
        for eponym, concept in scan_text(record_text(rec)):
            if (eponym, concept) in seen:
                continue
            seen.add((eponym, concept))
            result.fail(
                rec.get("id", "?"),
                f"fabricated collocation {eponym!r} + {concept!r} "
                f"(allowed: {sorted(EPONYM_COLLOCATIONS[eponym]) or 'none'})",
            )
    return result


def main():
    print("=== Gate 2: terminology (collocation) ===")
    result = run()
    ok = result.report()
    print(f"  {len(EPONYM_COLLOCATIONS)} eponyms with declared collocations, "
          f"{len(CONCEPT_NOUNS)} concept nouns")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
