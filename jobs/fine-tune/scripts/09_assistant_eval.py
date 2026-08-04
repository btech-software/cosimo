#!/usr/bin/env python3
"""Measure whether the checkpoint is still an assistant, not just an exam solver.

06_evaluate.py answers "was the number right". This answers the questions the
README's Known limitations said nothing answered:

  * does an exam trace leak into an open-ended answer  (style collapse)
  * does the model ask for what it is missing           (calibration)
  * does it invent terminology                          (triage list)
  * does a multi-step tool conversation complete        (agentic)

Every number is only meaningful as a base-vs-tuned delta, so run it against the
base model first, exactly like 03_baseline_eval.py:

    ./scripts/09_assistant_eval.py --run-name baseline
    ./scripts/09_assistant_eval.py --run-name sft --adapter runs/sft/adapter
    ./scripts/09_assistant_eval.py --run-name sft --merged runs/sft/merged
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from cosimo_ft import assistant, chat, generation, modeling, tools  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402
from cosimo_ft.runlog import RunDir, utc_now, write_json  # noqa: E402

LOGGER = logging.getLogger("assistant_eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-name", required=True, help="run directory under runs/")
    parser.add_argument("--adapter", default=None, help="LoRA adapter to attach")
    parser.add_argument("--merged", default=None, help="merged model directory")
    parser.add_argument("--base-id", default=None, help="override model.base_id")
    parser.add_argument(
        "--suites", nargs="+", default=None, help="suites to run (default: assistant.run)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="at most N items per suite"
    )
    config_mod.add_config_args(parser)
    return parser


def load_suite(cfg: dict, name: str, limit: int | None) -> list[dict]:
    """Read one curated prompt file."""
    configured = config_mod.get(cfg, f"assistant.suites.{name}")
    if not configured:
        raise SystemExit(
            f"unknown suite {name!r}; known: "
            f"{sorted(config_mod.get(cfg, 'assistant.suites', {}))}"
        )
    path = config_mod.harness_path(configured)
    if not path.is_file():
        raise SystemExit(f"suite file not found: {path}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return rows[:limit] if limit else rows


def run_prose_suite(
    model,
    tokenizer,
    cfg: dict,
    rows: list[dict],
    system: str,
    vocabulary: set[str],
) -> list[dict]:
    """Generate one answer per prompt and score its shape, not its content."""
    prompts = [chat.render_prompt(tokenizer, row["prompt"], system) for row in rows]
    outputs = generation.generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=int(config_mod.get(cfg, "assistant.max_new_tokens", 2048)),
        batch_size=int(config_mod.get(cfg, "assistant.batch_size", 8)),
        temperature=float(config_mod.get(cfg, "assistant.temperature", 0.0)),
        top_p=float(config_mod.get(cfg, "assistant.top_p", 1.0)),
        seed=int(config_mod.get(cfg, "seed", 3407)),
    )
    scored = []
    for row, output in zip(rows, outputs):
        text = output["text"]
        markers = assistant.exam_shape_markers(text)
        scored.append(
            {
                **row,
                "generation": text,
                "new_tokens": output["new_tokens"],
                "exam_shape": bool(markers),
                "exam_shape_markers": markers,
                "abstention": assistant.is_abstention(text),
                "unknown_terms": assistant.unknown_terms(text, vocabulary),
            }
        )
    return scored


def run_agentic_suite(
    model, tokenizer, cfg: dict, rows: list[dict], system: str
) -> list[dict]:
    """Drive each tool scenario as a real multi-turn conversation.

    Generated one scenario at a time rather than batched: each turn depends on
    the tool result of the previous one, so the conversations cannot be advanced
    in lockstep. The suite is small by design, which is what makes that
    affordable.
    """
    max_turns = int(config_mod.get(cfg, "assistant.max_tool_turns", 4))
    max_new_tokens = int(config_mod.get(cfg, "assistant.max_new_tokens", 2048))
    seed = int(config_mod.get(cfg, "seed", 3407))

    scored = []
    for row in rows:
        schemas = row.get("tools", [])
        offered = [s["function"]["name"] for s in schemas]
        results = row.get("results", {})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": row["prompt"]},
        ]
        all_calls: list[dict] = []
        final_text = ""
        for _ in range(max_turns):
            prompt = chat.render_conversation(
                tokenizer, messages, schemas, add_generation_prompt=True
            )
            output = generation.generate(
                model,
                tokenizer,
                [prompt],
                max_new_tokens=max_new_tokens,
                batch_size=1,
                temperature=float(config_mod.get(cfg, "assistant.temperature", 0.0)),
                top_p=float(config_mod.get(cfg, "assistant.top_p", 1.0)),
                seed=seed,
                progress=False,
            )[0]
            text = output["text"]
            calls = tools.parse_tool_calls(text)
            if not calls:
                final_text = text
                break
            all_calls.extend(calls)
            messages.append(tools.assistant_tool_call_message(calls))
            for call in calls:
                # An unoffered tool gets an explicit error rather than silence:
                # the interesting behaviour is what the model does after a failed
                # call, and a missing turn would end the conversation instead.
                payload = results.get(
                    call["name"], {"error": f"no such tool: {call['name']}"}
                )
                messages.append(
                    tools.tool_result_message(call["name"], json.dumps(payload))
                )
        else:
            LOGGER.warning(
                "%s: hit the %d-turn budget without a final answer", row["id"], max_turns
            )

        grade = assistant.grade_trajectory(
            {**row, "offered_tools": offered}, all_calls, final_text
        )
        scored.append(
            {
                "id": row["id"],
                "kind": grade["kind"],
                "prompt": row["prompt"],
                "n_expected": len(row.get("expected_calls", [])),
                "calls": all_calls,
                "final": final_text,
                "exam_shape": assistant.has_exam_shape(final_text),
                **grade,
            }
        )
    return scored


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    if args.adapter and args.merged:
        raise SystemExit("pass either --adapter or --merged, not both")
    try:
        cfg = config_mod.load_config(
            stage="assistant", extra=args.config, overrides=args.set
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"config error: {exc}") from exc

    for label, path in (("--adapter", args.adapter), ("--merged", args.merged)):
        if path and not Path(path).exists():
            raise SystemExit(f"{label} path does not exist: {path}")

    if not config_mod.get(cfg, "chat.template_path"):
        raise SystemExit(
            "chat.template_path is not set. The vendor template reinstates the "
            "Microsoft identity preamble, which would make these numbers "
            "incomparable with every other run."
        )

    # An empty vocabulary would report every technical term as unknown, which
    # looks like a catastrophic result rather than a misconfiguration.
    vocabulary = assistant.load_vocabulary(assistant.default_vocabulary_paths(cfg))
    if not vocabulary:
        raise SystemExit(
            "assistant.vocabulary_files resolved to an empty vocabulary; every "
            "term would be reported as unknown. Check the paths in "
            "configs/assistant.yaml."
        )
    LOGGER.info("vocabulary: %d known terms", len(vocabulary))

    # The persona is present because it is present at serving time; the exam
    # protocol is not, because instructing the FINAL ANSWER contract into the
    # prompt would manufacture the very format this script measures.
    system = chat.compose_system(
        cfg,
        short=False,
        exam=bool(config_mod.get(cfg, "assistant.exam_protocol", False)),
    )

    base_id = args.base_id or config_mod.get(cfg, "model.base_id")
    model, tokenizer = modeling.load_for_inference(
        base_id,
        adapter_path=args.adapter,
        merged_path=args.merged,
        max_seq_length=int(config_mod.get(cfg, "model.max_seq_length", 8192)),
        load_in_4bit=bool(config_mod.get(cfg, "model.load_in_4bit", False)),
        dtype=config_mod.get(cfg, "model.dtype", "bfloat16"),
        revision=config_mod.get(cfg, "model.revision"),
    )
    chat.apply_chat_template_override(tokenizer, cfg)

    run = RunDir(config_mod.get(cfg, "paths.runs_dir", "runs"), args.run_name)
    out_dir = run.root / "assistant_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_names = args.suites or config_mod.get(cfg, "assistant.run", [])
    suites: dict[str, dict] = {}
    for name in suite_names:
        rows = load_suite(cfg, name, args.limit)
        LOGGER.info("suite %s: %d prompts", name, len(rows))
        if name == "agentic":
            scored = run_agentic_suite(model, tokenizer, cfg, rows, system)
            suites[name] = assistant.summarize_agentic(scored)
        else:
            scored = run_prose_suite(
                model, tokenizer, cfg, rows, system, vocabulary
            )
            suites[name] = assistant.summarize_open_ended(scored)
        write_json(out_dir / f"{name}_generations.json", scored)

    metrics = {
        "run": args.run_name,
        "created_at": utc_now(),
        "config_hash": config_mod.config_hash(cfg),
        "chat_template_sha256": chat.chat_template_hash(cfg),
        "model": modeling.model_fingerprint(
            cfg,
            base_id=base_id,
            adapter_path=args.adapter,
            merged_path=args.merged,
        ),
        "suites": suites,
    }
    write_json(out_dir / "metrics.json", metrics)
    config_mod.save_config(cfg, out_dir / "resolved_config.yaml")

    for name, stats in suites.items():
        if name == "agentic":
            print(
                f"{name}: n={stats['n']} accuracy={stats['accuracy']:.3f} "
                f"multi_step={stats['multi_step_accuracy']:.3f} "
                f"no_call_precision={stats['no_call_precision']:.3f}"
            )
        else:
            print(
                f"{name}: n={stats['n']} exam_shape={stats['exam_shape_rate']:.3f} "
                f"abstention={stats['abstention_rate']:.3f} "
                f"mean_tokens={stats['mean_new_tokens']:.0f} "
                f"unknown_terms={stats['unknown_term_rate']:.3f}"
            )
    print(f"metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
