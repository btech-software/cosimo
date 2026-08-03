"""Comparison statistics and markdown rendering.

Pure python: the Wilson interval and the exact McNemar test are implemented from
their definitions so no scipy/numpy is needed at report time.
"""

from __future__ import annotations

import math

_METRIC_ROWS = (
    ("format compliance", "format_a", "format_b"),
    ("distractor rate", "distractor_a", "distractor_b"),
)

# Fields that must agree for a difference between two runs to be a model effect.
COMPARABILITY_FIELDS = (
    ("config_hash", ()),
    ("model", ("base_id", "revision", "load_in_4bit", "dtype")),
    # A different chat template means a different prompt surface, so the delta
    # would measure prompting rather than the model.
    ("chat_template", ("path", "sha256")),
    ("generation", ("max_new_tokens", "temperature", "top_p", "seed", "batch_size")),
)


def comparability_warnings(
    metrics_a: dict, metrics_b: dict, run_a: str, run_b: str
) -> list[str]:
    """Differences that would make a delta mean something other than the model.

    ``model.adapter_path``/``model.merged_path`` are excluded on purpose: the two
    runs are *supposed* to evaluate different artifacts.
    """
    warnings = []
    for field, subfields in COMPARABILITY_FIELDS:
        if not subfields:
            value_a, value_b = metrics_a.get(field), metrics_b.get(field)
            if value_a != value_b:
                warnings.append(
                    f"{field} differs: {run_a}={value_a!r} vs {run_b}={value_b!r}"
                )
            continue
        block_a = metrics_a.get(field) or {}
        block_b = metrics_b.get(field) or {}
        for subfield in subfields:
            value_a, value_b = block_a.get(subfield), block_b.get(subfield)
            if value_a != value_b:
                warnings.append(
                    f"{field}.{subfield} differs: "
                    f"{run_a}={value_a!r} vs {run_b}={value_b!r}"
                )
    return warnings


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (not the normal approximation)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denominator
    half_width = (z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial sign test on the discordant pairs).

    ``b`` = correct in A only, ``c`` = correct in B only. No discordant pairs
    means no evidence of a difference, so p = 1.0.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def _correct_by_id(rows: list[dict]) -> dict[str, bool]:
    return {str(row["id"]): bool(row.get("correct")) for row in rows}


def paired_delta(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Paired accuracy comparison over the ids present in both runs."""
    a = _correct_by_id(rows_a)
    b = _correct_by_id(rows_b)
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if n == 0:
        return {
            "n_paired": 0,
            "acc_a": 0.0,
            "acc_b": 0.0,
            "delta": 0.0,
            "b": 0,
            "c": 0,
            "p_value": 1.0,
        }
    k_a = sum(1 for i in shared if a[i])
    k_b = sum(1 for i in shared if b[i])
    only_a = sum(1 for i in shared if a[i] and not b[i])
    only_b = sum(1 for i in shared if b[i] and not a[i])
    return {
        "n_paired": n,
        "acc_a": k_a / n,
        "acc_b": k_b / n,
        "delta": (k_b - k_a) / n,
        "b": only_a,
        "c": only_b,
        "p_value": mcnemar_exact(only_a, only_b),
    }


def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _ci(value) -> str:
    if not value:
        return "n/a"
    return f"[{100.0 * float(value[0]):.1f}, {100.0 * float(value[1]):.1f}]"


def _settings_table(comparison: dict, run_a: str, run_b: str) -> list[str]:
    """The decoding and model settings each run actually used."""
    metrics_a = comparison.get("metrics_a") or {}
    metrics_b = comparison.get("metrics_b") or {}
    if not metrics_a and not metrics_b:
        return []
    lines = [
        f"| setting | {run_a} | {run_b} |",
        "| --- | --- | --- |",
        f"| config_hash | {metrics_a.get('config_hash')} | "
        f"{metrics_b.get('config_hash')} |",
    ]
    for block in ("model", "generation"):
        block_a = metrics_a.get(block) or {}
        block_b = metrics_b.get(block) or {}
        for key in sorted(set(block_a) | set(block_b)):
            lines.append(f"| {block}.{key} | {block_a.get(key)} | {block_b.get(key)} |")
    lines.append("")
    return lines


def render_markdown(comparison: dict) -> str:
    """Render the comparison produced by ``06_compare.py``.

    Expected shape::

        {"run_a": str, "run_b": str, "created_at": str,
         "metrics_a": {"config_hash": str, "model": {...}, "generation": {...}},
         "metrics_b": {...},
         "suites": {name: {**paired_delta(...), "ci_a": [lo, hi], "ci_b": [lo, hi],
                           "format_a": float, "format_b": float,
                           "distractor_a": float, "distractor_b": float,
                           "only_in_a": int, "only_in_b": int}},
         "comparability_warnings": [str], "warnings": [str]}
    """
    run_a = comparison.get("run_a", "A")
    run_b = comparison.get("run_b", "B")
    lines = [
        f"# {run_a} vs {run_b}",
        "",
        f"Generated: {comparison.get('created_at', 'unknown')}",
        "",
    ]
    comparability = comparison.get("comparability_warnings") or []
    if comparability:
        lines.append(
            "> NOT COMPARABLE: the runs were not measured under the same conditions, "
            "so the difference below is not a model effect alone."
        )
        for warning in comparability:
            lines.append(f"> - {warning}")
        lines.append("")
    for warning in comparison.get("warnings") or []:
        lines.append(f"> WARNING: {warning}")
    if comparison.get("warnings"):
        lines.append("")

    lines += _settings_table(comparison, run_a, run_b)
    lines += [
        f"| suite | n paired | {run_a} acc | {run_b} acc | delta | {run_a} only | "
        f"{run_b} only | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suite, stats in (comparison.get("suites") or {}).items():
        delta = 100.0 * float(stats.get("delta", 0.0))
        p_value = float(stats.get("p_value", 1.0))
        lines.append(
            f"| {suite} | {stats.get('n_paired', 0)} | {_pct(stats.get('acc_a'))} | "
            f"{_pct(stats.get('acc_b'))} | {delta:+.1f} pp | "
            f"{stats.get('b', 0)} | {stats.get('c', 0)} | {p_value:.4g} |"
        )
    lines.append("")
    lines += [
        f"| suite | metric | {run_a} | {run_b} |",
        "| --- | --- | ---: | ---: |",
    ]
    for suite, stats in (comparison.get("suites") or {}).items():
        lines.append(
            f"| {suite} | accuracy 95% CI | {_ci(stats.get('ci_a'))} | "
            f"{_ci(stats.get('ci_b'))} |"
        )
        for label, key_a, key_b in _METRIC_ROWS:
            lines.append(
                f"| {suite} | {label} | {_pct(stats.get(key_a))} | "
                f"{_pct(stats.get(key_b))} |"
            )
    lines.append("")
    lines.append(
        "Accuracy intervals are Wilson score intervals; the p-value is a two-sided "
        "exact McNemar test over the paired items (discordant pairs only)."
    )
    lines.append("")
    return "\n".join(lines)
