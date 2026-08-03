#!/usr/bin/env python3
"""Compare evaluation runs item by item and write a markdown/JSON report.

Only items evaluated by both runs are compared; when the item sets differ the
difference is reported as a warning instead of being silently averaged away.

Example:
    ./scripts/07_compare.py --runs baseline sft dpo \
        --suite cosimo_test cosimo_unseen_stems
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from cosimo_ft import evalrun, report, runlog  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs", nargs="+", required=True, help="two or more run names under runs/"
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        default=None,
        dest="suites",
        help="suites to compare (default: every suite both runs measured)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the runs were not measured under the same conditions",
    )
    config_mod.add_config_args(parser)
    return parser


def _rate(rows: list[dict], ids: set[str], key: str) -> float:
    selected = [row for row in rows if str(row["id"]) in ids]
    if not selected:
        return 0.0
    return sum(1 for row in selected if row.get(key)) / len(selected)


def _settings(metrics: dict) -> dict:
    return {
        "config_hash": metrics.get("config_hash"),
        "model": metrics.get("model") or {},
        "generation": metrics.get("generation") or {},
    }


def compare_runs(
    runs_dir: str, run_a: str, run_b: str, suites: list[str] | None = None
) -> dict:
    """Build the comparison payload consumed by ``report.render_markdown``.

    Suites come from each run's ``metrics.json``, not from a directory listing:
    the metrics file is the record of what was actually measured, and it is what
    carries the decoding settings the comparison must be checked against.
    """
    metrics_a = evalrun.load_metrics(runs_dir, run_a)
    metrics_b = evalrun.load_metrics(runs_dir, run_b)
    payload = {
        "run_a": run_a,
        "run_b": run_b,
        "created_at": runlog.utc_now(),
        "runs_dir": str(config_mod.harness_path(runs_dir)),
        "metrics_a": _settings(metrics_a),
        "metrics_b": _settings(metrics_b),
        "comparability_warnings": report.comparability_warnings(
            metrics_a, metrics_b, run_a, run_b
        ),
        "suites": {},
        "warnings": [],
    }
    if suites is None:
        suites_a = set(metrics_a.get("suites") or {})
        suites_b = set(metrics_b.get("suites") or {})
        suites = sorted(suites_a & suites_b)
        for missing in sorted(suites_a ^ suites_b):
            payload["warnings"].append(
                f"suite {missing}: measured by only one of the two runs; skipped"
            )
    for suite in suites:
        rows_a = evalrun.load_generations(runs_dir, run_a, suite)
        rows_b = evalrun.load_generations(runs_dir, run_b, suite)
        if not rows_a or not rows_b:
            payload["warnings"].append(
                f"suite {suite}: missing generations for "
                f"{run_a if not rows_a else run_b}; skipped"
            )
            continue
        ids_a = {str(row["id"]) for row in rows_a}
        ids_b = {str(row["id"]) for row in rows_b}
        shared = ids_a & ids_b
        if not shared:
            payload["warnings"].append(
                f"suite {suite}: the two runs share no items; skipped"
            )
            continue
        if ids_a != ids_b:
            payload["warnings"].append(
                f"suite {suite}: item sets differ ({len(ids_a - ids_b)} only in "
                f"{run_a}, {len(ids_b - ids_a)} only in {run_b}); comparing the "
                f"{len(shared)} shared items"
            )
        stats = report.paired_delta(rows_a, rows_b)
        n = stats["n_paired"]
        stats["ci_a"] = list(report.wilson_ci(round(stats["acc_a"] * n), n))
        stats["ci_b"] = list(report.wilson_ci(round(stats["acc_b"] * n), n))
        stats["format_a"] = _rate(rows_a, shared, "format_ok")
        stats["format_b"] = _rate(rows_b, shared, "format_ok")
        stats["distractor_a"] = _rate(rows_a, shared, "matched_distractor")
        stats["distractor_b"] = _rate(rows_b, shared, "matched_distractor")
        stats["only_in_a"] = len(ids_a - ids_b)
        stats["only_in_b"] = len(ids_b - ids_a)
        payload["suites"][suite] = stats
    return payload


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    if len(args.runs) < 2:
        raise SystemExit("--runs needs at least two run names")
    try:
        cfg = config_mod.load_config(
            stage="eval", extra=args.config, overrides=args.set
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"config error: {exc}") from exc
    runs_dir = config_mod.get(cfg, "paths.runs_dir", "runs")

    out_dir = config_mod.harness_path(runs_dir) / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    not_comparable = []
    for run_a, run_b in itertools.combinations(args.runs, 2):
        try:
            payload = compare_runs(runs_dir, run_a, run_b, args.suites)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{exc} A run is only comparable once 06_evaluate.py has written its "
                "metrics.json."
            ) from exc
        if not payload["suites"]:
            print(f"{run_a} vs {run_b}: no shared suites, skipped")
            continue
        markdown = report.render_markdown(payload)
        md_path = out_dir / f"{run_a}_vs_{run_b}.md"
        md_path.write_text(markdown, encoding="utf-8")
        json_path = runlog.write_json(out_dir / f"{run_a}_vs_{run_b}.json", payload)
        print(markdown)
        print(f"report: {md_path}")
        print(f"report: {json_path}")
        if payload["comparability_warnings"]:
            not_comparable.append(f"{run_a} vs {run_b}")

    if not_comparable and args.strict:
        raise SystemExit(
            "--strict: these run pairs were not measured under the same conditions: "
            + ", ".join(not_comparable)
        )


if __name__ == "__main__":
    main()
