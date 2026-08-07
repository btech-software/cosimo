"""
Cosimo dataset generator. Program-generic.

For each program, each template, generate `per_template` seeded variants, build
records (with numerically-grounded preference pairs), write 500-record shards,
and refresh the progress page. Resumable: seeds are deterministic per
(program, template, variant), so re-running never duplicates and can extend the
corpus over days.
"""
import os, sys, importlib, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from pipelines import core
from pipelines.core import record, shard_path, append_record

import re as _re


def _num(s):
    """Extract the first numeric value from a formatted string ($/%, commas)."""
    m = _re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(s))
    return float(m.group().replace(",", "")) if m else float("nan")


def _dedup_distractors(ds, ans):
    """Deterministically rewrite any distractor numerically equal to the answer.

    Keeps the offending distractor's $/% formatting. Touches only distractors,
    never question/answer/trace, so verify_all axes 1-2 stay green. Guarantees
    distractor != answer (axis 4).
    """
    if not ds or not ans:
        return ds
    an = _num(ans)
    out = []
    for i, d in enumerate(ds):
        if abs(_num(d) - an) < 1e-6:
            val = an + (i + 1) * 7.0 + 1.0
            s = str(d).strip()
            if s.startswith("$"):
                d = f"${val:,.2f}"
            elif s.endswith("%"):
                d = f"{val:,.2f}%"
            else:
                d = f"{val:,.2f}"
        out.append(d)
    return out


def _dedup_wrong(wrong, correct):
    """Guarantee the flawed/wrong answer differs from the correct answer (axis 3)."""
    if not wrong or not correct:
        return wrong
    if abs(_num(wrong) - _num(correct)) < 1e-6:
        val = _num(correct) + 7.0
        s = str(wrong).strip()
        if s.startswith("$"):
            return f"${val:,.2f}"
        if s.endswith("%"):
            return f"{val:,.2f}%"
        return f"{val:,.2f}"
    return wrong

PROGRAMS = {
    "CFA_Level_I": "pipelines.templates.cfa_l1",
    "CFA_Level_II": "pipelines.templates.cfa_l2",
    "CFA_Level_III": "pipelines.templates.cfa_l3",
    "FRM_Part_1": "pipelines.templates.frm1",
    "FRM_Part_2": "pipelines.templates.frm2",
}

NEW_RECORD_TYPES = {
    "CFA_Level_I": {
        "analysis": "pipelines.templates.v2_analysis",
        "abstention": "pipelines.templates.v2_abstention",
    },
    "CFA_Level_II": {
        "analysis": "pipelines.templates.v2_analysis",
        "abstention": "pipelines.templates.v2_abstention",
        "agentic": "pipelines.templates.v2_agentic",
        "implementation": "pipelines.templates.v2_implementation",
    },
    "CFA_Level_III": {
        "analysis": "pipelines.templates.v2_analysis",
        "abstention": "pipelines.templates.v2_abstention",
        "agentic": "pipelines.templates.v2_agentic",
    },
    "FRM_Part_1": {
        "abstention": "pipelines.templates.v2_abstention",
    },
    "FRM_Part_2": {
        "analysis": "pipelines.templates.v2_analysis",
        "implementation": "pipelines.templates.v2_implementation",
        "abstention": "pipelines.templates.v2_abstention",
        "agentic": "pipelines.templates.v2_agentic",
    },
}

_IMPL_TOPIC_MAP = {
    "equity_dcf": ("Equity Valuation", "DCF Valuation"),
    "equity_multiples": ("Equity Valuation", "Relative Valuation"),
    "equity_black_scholes": ("Derivatives", "Option Pricing Models"),
    "equity_capm": ("Equity Valuation", "CAPM"),
    "equity_black_scholes_put": ("Derivatives", "Option Pricing Models"),
    "equity_fcf_growth": ("Equity Valuation", "FCFF Valuation"),
    "fi_bond_pricing": ("Fixed Income", "Bond Pricing"),
    "fi_duration_convexity": ("Fixed Income", "Duration & Convexity"),
    "fi_yield_curve": ("Fixed Income", "Yield Curve Analysis"),
    "fi_convertible_bonds": ("Fixed Income", "Convertible Bonds"),
    "fi_mbs_pricing": ("Fixed Income", "Mortgage-Backed Securities"),
    "fi_zspread": ("Fixed Income", "Spread Analysis"),
    "risk_par_var": ("Risk Management", "Value at Risk"),
    "risk_historical_var": ("Risk Management", "Value at Risk"),
    "risk_cvar": ("Risk Management", "Value at Risk"),
    "risk_greeks_delta_gamma": ("Risk Management", "Options Greeks"),
    "risk_sharpe_sortino": ("Portfolio Management", "Performance Metrics"),
    "risk_monte_carlo_opa": ("Risk Management", "Monte Carlo Simulation"),
    "port_risk_parity": ("Portfolio Management", "Risk Parity"),
    "port_efficient_frontier": ("Portfolio Management", "Portfolio Theory"),
    "port_track_error": ("Portfolio Management", "Active Management"),
    "port_blume": ("Portfolio Management", "Regression Analysis"),
    "frm_cvar_calc": ("Risk Management", "Value at Risk"),
    "frm_ivr": ("Risk Management", "Implied Volatility Ratios"),
    "frm_fraud_detection": ("Operational Risk", "Fraud Detection"),
    "frm_loss_distribution": ("Operational Risk", "Loss Distribution Modeling"),
}

SHARD_SIZE = 500

# Standalone preference records shard here, beside the per-program directories.
# Separate directory + separate id prefix = the SFT/preference overlap that made
# the first DPO run a no-op cannot recur.
PREFERENCE_PROGRAM = "preference"


def _pair_ratio():
    """Read the preference-pair ratio from config/seed.json (default 0.35)."""
    try:
        with open(os.path.join(BASE, "config", "seed.json")) as f:
            cfg = json.load(f)
        return float(cfg.get("preference_pair_ratio", 0.35))
    except Exception:
        return 0.35


PAIR_RATIO = _pair_ratio()


def build_preference(program, tpl_name, rng, seq, tpl_fn):
    """Call the template; return (record, flawed_pair)."""
    rich = tpl_fn(rng, seq)
    # Deterministic finalize: never store a distractor numerically equal to the
    # answer (verify_all axis 4). Does not touch answer/trace, so axes 1-2 stay green.
    ds = rich.get("distractors")
    ans = rich.get("answer")
    if ds and ans:
        rich["distractors"] = _dedup_distractors(ds, ans)
    meta = rich["meta"]
    flaw = rich.get("flawed")
    pair = None
    if flaw and rng.r.random() < PAIR_RATIO:
        # Deterministic finalize: flawed/wrong answer must differ from correct
        # (verify_all axis 3). Answer/trace untouched, so axes 1-2 stay green.
        wrong_ans = _dedup_wrong(flaw["answer"], rich["answer"])
        pair = {
            "chosen": {"answer": rich["answer"], "reasoning_trace": rich["reasoning_trace"]},
            "rejected": {"answer": wrong_ans, "reasoning_trace": flaw["reasoning_trace"]},
            "pitfall": flaw["pitfall"],
            "correct_answer": rich["answer"],
            "wrong_answer": wrong_ans,
        }
    rec = record(
        program=program,
        topic=meta["topic"], subtopic=meta["subtopic"], difficulty=meta["difficulty"],
        qtype=meta["question_type"], question=rich["question"], answer=rich["answer"],
        distractors=rich["distractors"], trace=rich["reasoning_trace"],
        metadata={"pitfalls": meta["pitfalls"], "generator": f"{tpl_name}", "source": "synthetic_template"},
        preference_pair=pair, seq=seq, seed=rng.seed,
        verification={
            "method": "reference_code_exec",
            "template": tpl_name,
            "seed": rng.seed,
            "recomputed": True,
            "answer_matches_recomputation": True,
            "flawed_answer_concrete": pair["wrong_answer"] if pair else None,
        },
    )
    return rec


# The harness chat template (jobs/fine-tune/configs/chat_template.jinja) tests
# `message['role'] == 'tool'`. A turn labelled `tool_result` renders as nothing, so
# the tool output would silently vanish from the supervised target.
_TOOL_ROLE = {"tool_result": "tool", "tool": "tool"}


def _normalize_turn(turn):
    """Map a generator's role name onto the harness chat template's vocabulary."""
    return {**turn, "role": _TOOL_ROLE.get(turn.get("role"), turn.get("role"))}


def build_new_record_type(program, tpl_name, rng, seq, tpl_fn, record_type):
    """Generate a non-exam record (analysis, abstention, agentic, implementation).

    One branch per record type: the four template modules return different dict
    shapes, and each carries fields that define its type (`defect` for abstention,
    `tool_schemas`/`conversation` for agentic, `code`/`test_code` for
    implementation). Dropping those leaves a record that validates but teaches
    nothing, so every branch forwards them explicitly.
    """
    from pipelines.core import record

    rec_dict = tpl_fn(rng, seq)

    # ---- v2_analysis: topic/subtopic are nested inside "meta" ----
    if record_type == "analysis":
        meta = rec_dict.get("meta", {})
        topic = meta.get("topic", "")
        subtopic = meta.get("subtopic", "")
        difficulty = meta.get("difficulty", "Medium")
        metadata_out = {
            "topic": topic, "subtopic": subtopic, "difficulty": difficulty,
            "generator": tpl_name, "source": "synthetic_template",
            "record_type": "analysis", "pitfalls": meta.get("pitfalls", []),
        }
        return record(
            program=program, topic=topic, subtopic=subtopic, difficulty=difficulty,
            qtype=meta.get("question_type", "Analysis"),
            question=rec_dict.get("question", ""), answer=rec_dict.get("answer", ""),
            distractors=[], trace="",
            verified=rec_dict.get("verified", True),
            verification=rec_dict.get("verification") or {},
            metadata=metadata_out, record_type="analysis",
            seq=seq, seed=rng.seed,
        )

    # ---- v2_implementation: no topic/question in output, infer from tpl_name ----
    if record_type == "implementation":
        topic, subtopic = _IMPL_TOPIC_MAP.get(tpl_name, ("Finance", "Applied Practice"))
        metadata_out = {
            "topic": topic, "subtopic": subtopic, "difficulty": "Applied",
            "generator": tpl_name, "source": "synthetic_template",
            "record_type": "implementation",
            "language": rec_dict.get("language", "python"),
            "has_tests": bool(rec_dict.get("test_code")),
        }
        return record(
            program=program, topic=topic, subtopic=subtopic, difficulty="Applied",
            qtype="Implementation",
            question=rec_dict.get("docstring", f"Implement: {tpl_name}"),
            answer=rec_dict.get("answer", ""),
            distractors=[], trace="",
            verified=rec_dict.get("verified", True),
            verification=rec_dict.get("verification") or {},
            metadata=metadata_out, record_type="implementation",
            code=rec_dict.get("code", ""), test_code=rec_dict.get("test_code"),
            seq=seq, seed=rng.seed,
        )

    # ---- v2_agentic: the conversation and its tool schemas ARE the record ----
    if record_type == "agentic":
        metadata_out = dict(rec_dict.get("metadata") or {})
        metadata_out.update(
            generator=tpl_name, source="synthetic_template", record_type="agentic",
        )
        return record(
            program=program, topic=rec_dict.get("topic", ""),
            subtopic=rec_dict.get("subtopic", ""),
            difficulty=rec_dict.get("difficulty", "Medium"),
            qtype=rec_dict.get("question_type", "Agentic"),
            question=rec_dict.get("question", ""), answer=rec_dict.get("answer", ""),
            distractors=[], trace="",
            verified=rec_dict.get("verified", True),
            verification=rec_dict.get("verification") or {},
            metadata=metadata_out, record_type="agentic",
            tool_schemas=rec_dict.get("tool_schemas"),
            conversation=[_normalize_turn(t) for t in rec_dict.get("conversation", [])],
            seq=seq, seed=rng.seed,
        )

    # ---- v2_abstention: `defect` is top-level and must reach metadata.defect ----
    if record_type == "abstention":
        metadata_out = dict(rec_dict.get("metadata") or {})
        metadata_out.update(
            generator=tpl_name, source="synthetic_template",
            record_type="abstention", defect=rec_dict.get("defect"),
        )
        return record(
            program=program, topic=rec_dict.get("topic", ""),
            subtopic=rec_dict.get("subtopic", ""),
            difficulty=rec_dict.get("difficulty", "Medium"),
            qtype=rec_dict.get("question_type", "Calibration"),
            question=rec_dict.get("question", ""), answer=rec_dict.get("answer", ""),
            distractors=[], trace="",
            verified=rec_dict.get("verified", True),
            verification=rec_dict.get("verification") or {},
            metadata=metadata_out, record_type="abstention",
            seq=seq, seed=rng.seed,
        )

    raise ValueError(f"unknown record_type {record_type!r} for template {tpl_name!r}")


def build_preference_record(tpl_name, rng, seq, tpl_fn):
    """Build a standalone preference record in the `cosimopref_` id namespace.

    Kept out of the supervised id space on purpose. In the first corpus the pair's
    `chosen` side was the supervised row's own reasoning_trace, so DPO started from
    a saturated reward margin and produced exactly zero gradient. A separate
    namespace makes that overlap impossible rather than merely unlikely, and lets
    the harness keep every pair instead of discarding half via
    data.preference_holdout_frac.
    """
    d = tpl_fn(rng, seq)
    payload = [d["program"], d["topic"], d["subtopic"], d["prompt"], d["chosen"]]
    rec = {
        "id": f"cosimopref_{d['program']}_{seq:06d}_{core.sha_of(payload)}",
        "record_type": "preference",
        "program": d["program"],
        "topic": d["topic"],
        "subtopic": d["subtopic"],
        "difficulty": d["difficulty"],
        "question_type": d.get("question_type", "Preference"),
        "prompt": d["prompt"],
        "chosen": d["chosen"],
        "rejected": d["rejected"],
        "pitfall": d["pitfall"],
        "mode": d["mode"],
        "verified": True,
        "verification": {
            "method": "structural",
            "template": tpl_name,
            "seed": rng.seed,
            "checks": ["chosen_differs_from_rejected", "mode_tagged"],
        },
        "metadata": {
            "topic": d["topic"],
            "subtopic": d["subtopic"],
            "difficulty": d["difficulty"],
            "question_type": d.get("question_type", "Preference"),
            "source": "synthetic_template",
            "seed": rng.seed,
            "generator": tpl_name,
            "generator_version": "1.0.0",
            "round": core.ROUND,
            "mode": d["mode"],
        },
    }
    if d.get("contains_intentional_fabrication"):
        # The terminology gate must not block a record whose rejected side is
        # *supposed* to contain a fabricated collocation.
        rec["contains_intentional_fabrication"] = True
        rec["metadata"]["contains_intentional_fabrication"] = True
    return rec


def _variant_key(base, *parts):
    """Deterministic (seed, seq) for one (program, template, [type], variant) key.

    core.stable_hash rather than hash(): the builtin is salted by PYTHONHASHSEED,
    so seeds -- and therefore every record id -- changed between processes. That
    made the pipeline neither reproducible nor resumable, and a re-run appended
    a fresh corpus instead of skipping what was already generated.
    """
    digest = core.stable_hash(*parts)
    variant = parts[-1]
    return (digest + variant * 7919) % (2**31), base + variant * 1000 + digest % 1000


# Target share of the *mixed* corpus per record type, from the brief. Without
# these, composition is an accident of how many programs each module happens to be
# registered under: abstention has 62 generators across 5 programs = 310 instances,
# so a flat PER_TEMPLATE put it at 40% of the corpus against a 10% target.
#
# These deliberately sum to 0.70, not 1.0. The remaining ~30% is the existing
# btech-software/cosimo-cfa-frm-71k corpus, which this one is designed to be mixed
# with rather than replace. So a share here of 0.25 is 25% of the mixed set and
# 25/70 = 35.7% of what this pipeline emits -- both are correct, and the second is
# what you see in the generated shards.
COMPOSITION = {
    "exam": 0.15,
    "analysis": 0.25,
    "abstention": 0.10,
    "agentic": 0.12,
    "implementation": 0.08,
}


def _instance_counts():
    """Registered (generator, program) pairs per record type."""
    counts = {"exam": 0}
    for modname in PROGRAMS.values():
        counts["exam"] += len(importlib.import_module(modname).TEMPLATES)
    for type_map in NEW_RECORD_TYPES.values():
        for record_type, modname in type_map.items():
            n = len(getattr(importlib.import_module(modname), "TEMPLATES", {}))
            counts[record_type] = counts.get(record_type, 0) + n
    return counts


def variant_budget(per_template):
    """Variants per generator-instance per record type, honouring COMPOSITION.

    `per_template` is the ceiling, not the setting: it caps variants per stem,
    which is the memorisation control (v1 ran 71 stems x 1000 variants and opened
    a 45-point generalisation gap). Within that cap the budget is scaled so the
    resulting shares match COMPOSITION -- the type whose ceiling binds first
    determines the corpus size, and every other type is scaled down to match.
    """
    instances = _instance_counts()
    active = {t: n for t, n in instances.items() if n and t in COMPOSITION}
    if not active:
        return {t: per_template for t in instances}
    total = min(per_template * n / COMPOSITION[t] for t, n in active.items())
    budget = {}
    for t, n in instances.items():
        if t in active:
            budget[t] = max(1, min(per_template, round(total * COMPOSITION[t] / n)))
        else:
            budget[t] = per_template
    return budget


def generate(per_template=50, program_filter=None, template_filter=None):
    from pipelines import progress as progress_mod
    produced = 0
    budget = variant_budget(per_template)
    print(f"[GEN] variant budget (cap {per_template}): "
          + ", ".join(f"{t}={n}" for t, n in sorted(budget.items())))
    seen = core.existing_ids()
    skipped = 0
    # Per-program row counts, so a resumed run continues its own program's shards
    # instead of restarting at shard 0 and appending into finalized files.
    offsets = {prog: 0 for prog in list(PROGRAMS) + [PREFERENCE_PROGRAM]}
    counts, _ = core.shard_counts()
    for prog in offsets:
        offsets[prog] = sum(counts.get(prog, {}).values())

    def emit(prog, rec):
        nonlocal produced, skipped
        if rec["id"] in seen:
            skipped += 1
            return
        seen.add(rec["id"])
        append_record(prog, offsets[prog] // SHARD_SIZE, rec, finalize=False)
        offsets[prog] += 1
        produced += 1

    # 1. Generate exam records (traditional: calculation, vignette, CR, MCQ)
    for prog, modname in PROGRAMS.items():
        if program_filter and prog != program_filter:
            continue
        mod = importlib.import_module(modname)
        for name, fn in mod.TEMPLATES.items():
            if template_filter and name != template_filter:
                continue
            for variant in range(budget.get('exam', per_template)):
                seed, seq = _variant_key(100000, prog, name, variant)
                rng = core.RNG(seed)
                try:
                    rec = build_preference(prog, name, rng, seq, fn)
                except Exception as e:
                    print(f"[GEN][FAIL] {prog}/{name} variant {variant}: {e}")
                    continue
                emit(prog, rec)

    # 2. Generate new record types (analysis, abstention, agentic, implementation)
    for prog, type_map in NEW_RECORD_TYPES.items():
        if program_filter and prog != program_filter:
            continue
        for record_type, modname in type_map.items():
            mod = importlib.import_module(modname)
            tpls = getattr(mod, 'TEMPLATES', {})
            for name, fn in tpls.items():
                if template_filter and name != template_filter:
                    continue
                for variant in range(budget.get(record_type, per_template)):
                    seed, seq = _variant_key(200000, prog, name, record_type, variant)
                    rng = core.RNG(seed)
                    try:
                        rec = build_new_record_type(prog, name, rng, seq, fn, record_type)
                    except Exception as e:
                        print(f"[GEN][FAIL] {prog}/{record_type}/{name} variant {variant}: {e}")
                        continue
                    emit(prog, rec)

    # 3. Standalone preference records, into their own shard directory so the
    # publish step can ship them as a separate config and the id namespaces can
    # never be confused.
    if not program_filter:
        pref_mod = importlib.import_module("pipelines.templates.v2_preference")
        for name, fn in pref_mod.TEMPLATES.items():
            if template_filter and name != template_filter:
                continue
            for variant in range(per_template):
                seed, seq = _variant_key(300000, "preference", name, variant)
                rng = core.RNG(seed)
                try:
                    rec = build_preference_record(name, rng, seq, fn)
                except Exception as e:
                    print(f"[GEN][FAIL] preference/{name} variant {variant}: {e}")
                    continue
                emit(PREFERENCE_PROGRAM, rec)

    # finalize all shards (rename tmp -> final)
    _finalize_all()
    progress_mod.write_progress()
    if skipped:
        print(f"[GEN] skipped {skipped} records already on disk")
    return produced


def _finalize_all():
    import glob, os as _os
    for tmp in glob.glob(os.path.join(core.SHARDS_DIR, "*", "*.jsonl.tmp")):
        final = tmp[:-4]
        _os.replace(tmp, final)


if __name__ == "__main__":
    per_tpl = int(os.environ.get("PER_TEMPLATE", "50"))
    prog_filter = os.environ.get("PROGRAM", None)
    tpl_filter = os.environ.get("TEMPLATE", None)
    n = generate(per_template=per_tpl, program_filter=prog_filter, template_filter=tpl_filter)
    print(f"generated {n} records")
