#!/usr/bin/env python3
"""Evaluate the untuned base model and store it as the reference measurement.

Every later comparison is made against runs/baseline, so this script refuses to
overwrite an existing baseline unless --force is given.

Example:
    ./scripts/03_baseline_eval.py --suites cosimo_test cosimo_unseen_stems gsm8k math500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from cosimo_ft import benchmarks, evalrun  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402
from cosimo_ft.runlog import RunDir  # noqa: E402

RUN_NAME = "baseline"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-id", default=None, help="override model.base_id")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=None,
        choices=list(benchmarks.ALL_SUITES),
        help="suites to evaluate (default: eval.suites from the config)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate at most N items per suite"
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip items already generated"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing runs/baseline"
    )
    config_mod.add_config_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    try:
        cfg = config_mod.load_config(
            stage="eval", extra=args.config, overrides=args.set
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"config error: {exc}") from exc

    runs_dir = config_mod.get(cfg, "paths.runs_dir", "runs")
    run = RunDir(runs_dir, RUN_NAME)
    if run.exists() and not (args.force or args.resume):
        raise SystemExit(
            f"{run.root} already exists. The baseline is the reference measurement "
            "for every comparison: pass --resume to continue it, or --force to "
            "overwrite it."
        )

    base_id = args.base_id or config_mod.get(cfg, "model.base_id")
    metrics = evalrun.run_evaluation(
        cfg,
        run_name=RUN_NAME,
        base_id=base_id,
        adapter_path=None,
        merged_path=None,
        suites=args.suites,
        limit=args.limit,
        resume=args.resume,
    )

    for suite, stats in metrics["suites"].items():
        print(
            f"{suite}: n={stats['n']} accuracy={stats['accuracy']:.4f} "
            f"format={stats['format_compliance']:.4f} "
            f"distractor={stats['distractor_rate']:.4f}"
        )
    print(f"metrics: {run.eval_dir / 'metrics.json'}")
    print(f"generations: {run.eval_dir}")


if __name__ == "__main__":
    main()
