"""The single evaluation implementation shared by baseline and checkpoint runs.

Both ``02_baseline_eval.py`` and ``05_evaluate.py`` call ``run_evaluation`` so
the baseline and every fine-tuned model are measured by identical code: same
prompts, same decoding parameters, same grader.
"""

from __future__ import annotations

import json
import logging
import math
import time

from . import benchmarks, chat, grading, report, runlog
from . import config as config_mod
from .runlog import RunDir

logger = logging.getLogger(__name__)

GENERATION_FIELDS = (
    "id",
    "suite",
    "program",
    "topic",
    "question_type",
    "difficulty",
    "generator",
    "stem_family",
    "prompt",
    "generation",
    "pred",
    "gold",
    "correct",
    "format_ok",
    "matched_distractor",
    "mode",
    "new_tokens",
)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def _group_accuracy(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        groups.setdefault(str(value), []).append(row)
    return {
        name: {
            "n": len(members),
            "accuracy": sum(1 for m in members if m["correct"]) / len(members),
        }
        for name, members in sorted(groups.items())
    }


def summarize_suite(rows: list[dict], max_new_tokens: int) -> dict:
    """Aggregate one suite's generation rows into the metrics block."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "accuracy_ci95": [0.0, 0.0],
            "format_compliance": 0.0,
            "distractor_rate": 0.0,
            "mean_new_tokens": 0.0,
            "p95_new_tokens": 0.0,
            "truncation_rate": 0.0,
            "by_program": {},
            "by_question_type": {},
            "by_topic": {},
            "by_difficulty": {},
        }
    correct = sum(1 for row in rows if row["correct"])
    tokens = [float(row.get("new_tokens") or 0) for row in rows]
    low, high = report.wilson_ci(correct, n)
    return {
        "n": n,
        "accuracy": correct / n,
        "accuracy_ci95": [low, high],
        "format_compliance": sum(1 for row in rows if row["format_ok"]) / n,
        "distractor_rate": sum(1 for row in rows if row.get("matched_distractor")) / n,
        "mean_new_tokens": sum(tokens) / n,
        "p95_new_tokens": _percentile(tokens, 0.95),
        "truncation_rate": sum(1 for t in tokens if t >= max_new_tokens) / n,
        "by_program": _group_accuracy(rows, "program"),
        "by_question_type": _group_accuracy(rows, "question_type"),
        "by_topic": _group_accuracy(rows, "topic"),
        "by_difficulty": _group_accuracy(rows, "difficulty"),
    }


def _grade_item(item: dict, generation: str, tag: str, rel_tol: float) -> grading.Grade:
    if benchmarks.is_math_suite(item["suite"]):
        return grading.grade_math(item["gold"], generation, tag, rel_tol)
    return grading.grade_cosimo(item["meta"], generation, tag, rel_tol)


def _row(
    item: dict, prompt: str, generation: str, new_tokens: int, grade: grading.Grade
) -> dict:
    meta = item.get("meta") or {}
    return {
        "id": item["id"],
        "suite": item["suite"],
        "program": meta.get("program"),
        "topic": meta.get("topic"),
        "question_type": item.get("question_type"),
        "difficulty": meta.get("difficulty"),
        "generator": meta.get("generator"),
        "stem_family": meta.get("stem_family"),
        "prompt": prompt,
        "generation": generation,
        "pred": grade.pred,
        "gold": item["gold"],
        "correct": grade.correct,
        "format_ok": grade.format_ok,
        "matched_distractor": grade.matched_distractor,
        "mode": grade.mode,
        "new_tokens": new_tokens,
    }


def run_evaluation(
    cfg: dict,
    *,
    run_name: str,
    base_id: str,
    adapter_path: str | None = None,
    merged_path: str | None = None,
    suites: list[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> dict:
    """Evaluate one model artifact and write ``runs/<run_name>/eval/``.

    Generations are appended per batch so a crash loses only the batch in
    flight; ``resume=True`` skips ids already present in the suite's JSONL.
    Metrics always cover exactly the items requested by this call, never stale
    rows left by an earlier one.

    Everything is written under ``eval/``, and the run manifest is extended
    rather than replaced, because a run directory is shared with the training
    stage that produced the checkpoint being evaluated.
    """
    from . import generation as generation_mod
    from . import modeling

    started = time.monotonic()
    suite_names = list(suites or config_mod.get(cfg, "eval.suites", []) or [])
    if not suite_names:
        raise ValueError("no evaluation suites requested")

    tag = config_mod.get(cfg, "prompt.final_answer_tag", grading.DEFAULT_TAG)
    rel_tol = float(config_mod.get(cfg, "eval.rel_tol", 1e-3))
    max_new_tokens = int(config_mod.get(cfg, "eval.max_new_tokens", 768))
    batch_size = int(config_mod.get(cfg, "eval.batch_size", 16))
    temperature = float(config_mod.get(cfg, "eval.temperature", 0.0))
    top_p = float(config_mod.get(cfg, "eval.top_p", 1.0))
    seed = int(config_mod.get(cfg, "seed", 3407))
    # Evaluation always uses the FULL identity plus the exam protocol, for every
    # suite: the identity applies everywhere, and GSM8K/MATH-500 are exam-format
    # items too. The short identity is a training-time variation only.
    system = chat.compose_system(cfg, short=False, exam=True)

    # Checked before the model loads, alongside the other fail-fast validation: a
    # checkpoint trained under the harness template but evaluated under the vendor
    # one is prompted differently from its own baseline, which breaks comparability
    # silently and in the direction that flatters the fine-tune. Fatal, exactly as
    # in the training and export scripts. FileNotFoundError propagates as-is.
    if chat.load_chat_template(cfg) is None:
        raise SystemExit(
            "chat.template_path is not set, so evaluation would use the vendor "
            "chat template while trained checkpoints use the harness template. "
            "Base and fine-tuned numbers would not be comparable. Set "
            "chat.template_path in configs/base.yaml."
        )

    # Load every suite before touching the GPU so missing data fails fast.
    loaded = {
        name: benchmarks.load_suite(name, cfg, limit=limit) for name in suite_names
    }
    for name, items in loaded.items():
        # Ids key the resume set and the paired comparison, so they must be unique.
        if len({item["id"] for item in items}) != len(items):
            raise ValueError(f"suite {name} contains duplicate item ids")
        logger.info("suite %s: %d items", name, len(items))

    run = RunDir(config_mod.get(cfg, "paths.runs_dir", "runs"), run_name).create("eval")
    # Eval provenance lives under eval/: writing it to the run root would clobber
    # the training config and environment of the checkpoint being evaluated.
    run.save_config(cfg, dest=run.eval_dir)
    run.save_env(dest=run.eval_dir)
    if not resume:
        # Clear *every* suite, not just the requested ones: a leftover JSONL from an
        # earlier model or an earlier decoding budget would otherwise sit next to
        # fresh results and be read as if it were current.
        for stale in sorted(run.eval_dir.glob("*_generations.jsonl")):
            logger.info("removing stale generations file %s", stale.name)
            stale.unlink()
        (run.eval_dir / "metrics.json").unlink(missing_ok=True)

    model, tokenizer = modeling.load_for_inference(
        base_id,
        adapter_path=adapter_path,
        merged_path=merged_path,
        max_seq_length=int(config_mod.get(cfg, "model.max_seq_length", 2048)),
        load_in_4bit=bool(config_mod.get(cfg, "model.load_in_4bit", False)),
        dtype=config_mod.get(cfg, "model.dtype", "bfloat16"),
        revision=config_mod.get(cfg, "model.revision"),
    )
    # Must happen before any rendering: the vendor template would otherwise inject
    # the Microsoft identity preamble ahead of the Cosimo system message. The
    # null-path case already exited above, before the model was loaded.
    if not chat.apply_chat_template_override(tokenizer, cfg):  # pragma: no cover
        raise SystemExit("chat template override did not apply")

    suite_metrics = {}
    for name, items in loaded.items():
        target = run.eval_dir / f"{name}_generations.jsonl"
        done_ids: set[str] = set()
        if resume:
            done_ids = {str(row["id"]) for row in runlog.read_jsonl(target)}
            if done_ids:
                logger.info(
                    "suite %s: resuming, %d items already done", name, len(done_ids)
                )

        pending = [item for item in items if item["id"] not in done_ids]
        prompts = {
            item["id"]: chat.render_prompt(tokenizer, item["question"], system)
            for item in pending
        }
        # Bucket by prompt length so each batch pads as little as possible; rows
        # are keyed by id, so the write order does not matter.
        pending.sort(key=lambda item: (len(prompts[item["id"]]), item["id"]))

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            outputs = generation_mod.generate(
                model,
                tokenizer,
                [prompts[item["id"]] for item in batch],
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                progress=False,
            )
            rows = []
            for item, output in zip(batch, outputs, strict=True):
                grade = _grade_item(item, output["text"], tag, rel_tol)
                rows.append(
                    _row(
                        item,
                        prompts[item["id"]],
                        output["text"],
                        output["new_tokens"],
                        grade,
                    )
                )
            runlog.append_jsonl(target, rows)
            logger.info(
                "suite %s: %d/%d",
                name,
                min(start + batch_size, len(pending)),
                len(pending),
            )

        # Metrics cover exactly the requested items. A resumed run whose sample
        # size shrank must not report the larger n of the file it resumed from,
        # and stale rows must not be scored against the current max_new_tokens.
        wanted = {item["id"] for item in items}
        rows = runlog.read_jsonl(target)
        scored = [row for row in rows if str(row["id"]) in wanted]
        if len(scored) != len(rows):
            logger.warning(
                "suite %s: %d row(s) in %s are outside the current item set and are "
                "excluded from the metrics",
                name,
                len(rows) - len(scored),
                target.name,
            )
        suite_metrics[name] = summarize_suite(scored, max_new_tokens)

    metrics = {
        "run": run_name,
        "created_at": runlog.utc_now(),
        "config_hash": config_mod.config_hash(cfg),
        "model": modeling.model_fingerprint(
            cfg, base_id=base_id, adapter_path=adapter_path, merged_path=merged_path
        ),
        # The prompt surface the model was measured through. Hashed the same way
        # 01_prepare_data.py hashes it into split_manifest.json, so training data
        # and evaluation runs can be lined up against each other.
        "chat_template": {
            "path": config_mod.get(cfg, "chat.template_path"),
            "sha256": chat.chat_template_hash(cfg),
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "batch_size": batch_size,
            "seed": seed,
        },
        "suites": suite_metrics,
        "wall_clock_seconds": round(time.monotonic() - started, 2),
    }
    runlog.write_json(run.eval_dir / "metrics.json", metrics)
    run.append_manifest_entry(
        "evaluations",
        {
            "created_at": metrics["created_at"],
            "config_hash": metrics["config_hash"],
            "model": metrics["model"],
            "generation": metrics["generation"],
            "suites": {name: metrics["suites"][name]["n"] for name in suite_metrics},
            "artifacts": {
                "metrics": str(run.eval_dir / "metrics.json"),
                "generations": {
                    name: str(run.eval_dir / f"{name}_generations.jsonl")
                    for name in suite_metrics
                },
            },
        },
    )
    return metrics


def load_metrics(runs_dir: str, run_name: str) -> dict:
    """Read ``runs/<run_name>/eval/metrics.json``."""
    path = RunDir(runs_dir, run_name).eval_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no metrics for run {run_name!r}: {path} does not exist"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_generations(runs_dir: str, run_name: str, suite: str) -> list[dict]:
    """Read ``runs/<run_name>/eval/<suite>_generations.jsonl``."""
    return runlog.read_jsonl(
        RunDir(runs_dir, run_name).eval_dir / f"{suite}_generations.jsonl"
    )
