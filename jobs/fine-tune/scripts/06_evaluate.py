#!/usr/bin/env python3
"""Evaluate a fine-tuned checkpoint with exactly the same protocol as the baseline.

Example:
    ./scripts/06_evaluate.py --run-name sft --adapter runs/sft/adapter \
        --suites cosimo_test cosimo_unseen_stems gsm8k math500
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-name", required=True, help="evaluation run directory under runs/"
    )
    parser.add_argument(
        "--adapter", default=None, help="LoRA adapter to attach to the base model"
    )
    parser.add_argument(
        "--merged", default=None, help="merged model directory to load directly"
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
    config_mod.add_config_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    if args.adapter and args.merged:
        raise SystemExit("pass either --adapter or --merged, not both")
    try:
        cfg = config_mod.load_config(
            stage="eval", extra=args.config, overrides=args.set
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"config error: {exc}") from exc

    for label, path in (("--adapter", args.adapter), ("--merged", args.merged)):
        if path and not Path(path).exists():
            raise SystemExit(f"{label} path does not exist: {path}")

    base_id = args.base_id or config_mod.get(cfg, "model.base_id")
    metrics = evalrun.run_evaluation(
        cfg,
        run_name=args.run_name,
        base_id=base_id,
        adapter_path=args.adapter,
        merged_path=args.merged,
        suites=args.suites,
        limit=args.limit,
        resume=args.resume,
    )

    run = RunDir(config_mod.get(cfg, "paths.runs_dir", "runs"), args.run_name)
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
