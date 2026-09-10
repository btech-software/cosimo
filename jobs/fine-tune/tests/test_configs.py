"""The shipped YAML configs and the layered override surface.

These tests are the guard against a silent knob change: every value asserted
here is one the build contract fixes, or one whose default a reader of the
README is entitled to rely on.
"""

from __future__ import annotations

import pytest
import yaml

from cosimo_ft import config as config_mod

from harness_fixtures import CONFIG_DIR

STAGES = ("data", "eval", "sft", "dpo", "orpo")


@pytest.fixture(scope="module")
def base() -> dict:
    return config_mod.load_config()


# --------------------------------------------------------------------------
# every config file parses and layers
# --------------------------------------------------------------------------


def test_every_config_file_is_a_yaml_mapping():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert {f.name for f in files} == {
        "base.yaml",
        "data.yaml",
        "eval.yaml",
        "sft.yaml",
        "dpo.yaml",
        "orpo.yaml",
        "assistant.yaml",
    }
    for path in files:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict), f"{path.name} must contain a mapping"


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_loads_on_top_of_base(stage, base):
    cfg = config_mod.load_config(stage=stage)
    # The stage layer adds its own section without dropping the shared one.
    assert cfg["model"] == base["model"]
    assert cfg["prompt"] == base["prompt"]
    assert stage in cfg or stage == "data" and "data" in cfg


def test_merge_order_is_base_then_stage_then_extra_then_set(tmp_path):
    extra = tmp_path / "extra.yaml"
    extra.write_text("model:\n  max_seq_length: 4096\nseed: 1\n", encoding="utf-8")
    cfg = config_mod.load_config(stage="sft", extra=[str(extra)], overrides=["seed=99"])
    assert cfg["model"]["max_seq_length"] == 4096  # extra beats base
    assert cfg["model"]["base_id"] == "Qwen/Qwen3.8-27B"  # base survives
    assert cfg["sft"]["run_name"] == "sft"  # stage layer survives
    assert cfg["seed"] == 99  # --set beats everything


def test_deep_merge_does_not_drop_sibling_keys():
    merged = config_mod.deep_merge(
        {"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20}, "c": 4}
    )
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}


def test_lists_replace_rather_than_append():
    merged = config_mod.deep_merge({"a": [1, 2, 3]}, {"a": [9]})
    assert merged["a"] == [9]


# --------------------------------------------------------------------------
# --set parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "path", "value"),
    [
        ("seed=7", ["seed"], 7),
        ("eval.batch_size=4", ["eval", "batch_size"], 4),
        ("model.load_in_4bit=true", ["model", "load_in_4bit"], True),
        ("model.revision=null", ["model", "revision"], None),
        ("eval.rel_tol=1.0e-2", ["eval", "rel_tol"], 0.01),
        ("model.base_id=some/model", ["model", "base_id"], "some/model"),
    ],
)
def test_parse_override_reads_values_as_yaml(item, path, value):
    assert config_mod.parse_override(item) == (path, value)


def test_override_applies_to_a_nested_key_without_mutating_the_input():
    cfg = config_mod.load_config(stage="eval")
    updated = config_mod.apply_override(cfg, "eval.batch_size=1")
    assert updated["eval"]["batch_size"] == 1
    assert cfg["eval"]["batch_size"] != 1


def test_unknown_override_keys_are_refused():
    cfg = config_mod.load_config(stage="eval")
    with pytest.raises(KeyError):
        config_mod.apply_override(cfg, "evla.batch_size=4")  # typo in the section
    with pytest.raises(KeyError):
        config_mod.apply_override(cfg, "eval.bathc_size=4")  # typo in the key
    with pytest.raises(KeyError):
        config_mod.apply_override(cfg, "seed.nested=4")  # not a section


def test_malformed_override_is_refused():
    with pytest.raises(ValueError):
        config_mod.parse_override("seed")
    with pytest.raises(ValueError):
        config_mod.parse_override("=7")


def test_config_hash_is_stable_and_sensitive(base):
    assert config_mod.config_hash(base) == config_mod.config_hash(dict(base))
    changed = config_mod.apply_override(base, "seed=1")
    assert config_mod.config_hash(changed) != config_mod.config_hash(base)
    assert len(config_mod.config_hash(base)) == 12


def test_get_returns_the_default_for_absent_keys(base):
    assert config_mod.get(base, "prompt.variation_rate") == 0.15
    assert config_mod.get(base, "prompt.nope", "fallback") == "fallback"
    assert config_mod.get(base, "seed.nope", None) is None


# --------------------------------------------------------------------------
# base.yaml — the locked decisions
# --------------------------------------------------------------------------


def test_base_model_and_precision(base):
    assert base["seed"] == 3407
    assert base["model"]["base_id"] == "Qwen/Qwen3.8-27B"
    # 8192, not 2048: the served target is a LangGraph ReAct loop whose
    # conversation accumulates tool calls and tool results on top of the persona.
    assert base["model"]["max_seq_length"] == 8192
    # QLoRA, unlike the Phi run: 27B parameters at bf16 leave no room for an 8k
    # sequence and its activations in 128 GB of unified memory.
    assert base["model"]["load_in_4bit"] is True, "QLoRA is the locked default at 27B"
    assert base["model"]["dtype"] == "bfloat16"
    assert base["dataset"]["hub_id"] == "btech-software/cosimo-quant-assistant-v3"


def test_the_corpus_is_v3_alone_with_no_v1_mix(base):
    """v3 leads and nothing is mixed in.

    The v2 build mixed v1 in at a 12% cap to buy exam depth, because v2's exam
    slice was thin. v3 generates its own exam records under an inventory family
    cap, so the crutch is gone -- and re-adding it would reintroduce the
    exam-heavy corpus that collapsed the first run's response style
    (spec §2, §12).
    """
    dataset = base["dataset"]
    assert dataset["preference_config"] == "preference"
    assert dataset["mix"] == [], "v3 trains on one corpus; see spec §8.1 `mix: []`"
    # The local-shard escape hatch exists but must not be the shipped default:
    # a committed absolute path would silently prepare one developer's tree.
    assert dataset["local_dir"] is None


def test_identity_block_is_the_contracted_persona(base):
    identity = base["prompt"]["identity"]
    assert identity.startswith(
        "You are Cosimo, a quantitative finance assistant at Btech Software."
    )
    # Each clause is load-bearing behaviour, not decoration: hedging over false
    # precision, no fabricated market data, tools for retrieval, and exam
    # liturgy kept in its lane.
    for phrase in (
        "Prefer ranges when the inputs do not identify a point",
        "Do not invent",
        "Use tools when a number must be retrieved",
        "Never use exam liturgy unless the user asked an exam item",
    ):
        assert phrase in identity
    assert "Microsoft" not in identity
    # Shortened deliberately (spec §8.1). The Phi-era block was 2,494 chars on
    # every example; length was not buying behaviour, and DPO's keep_end
    # truncation cuts the START of the prompt, which is where this sits.
    assert len(identity) < 600, "the v3 identity is short on purpose"


def test_short_identity_is_the_one_line_variant(base):
    assert base["prompt"]["identity_short"] == (
        "You are Cosimo, a quantitative finance assistant at Btech Software."
    )


def test_exam_protocol_carries_the_grading_contract(base):
    protocol = base["prompt"]["exam_protocol"]
    assert protocol.startswith("Solve the problem step by step")
    # v3's exam renderer composes four labelled options and closes on
    # `FINAL ANSWER: <letter> -- <value> <unit>`. Instructing the v2 `<value>`
    # form here would tell the model to produce something every supervised
    # target contradicts.
    assert protocol.rstrip().endswith("FINAL ANSWER: <letter> -- <value> <unit>")
    assert "\nFINAL ANSWER: <letter>" in protocol, "the contract must be its own line"
    assert base["prompt"]["final_answer_tag"] == "FINAL ANSWER:"


def test_variation_rate_is_fifteen_percent(base):
    assert base["prompt"]["variation_rate"] == 0.15


def test_chat_template_override_is_configured(base):
    assert base["chat"]["template_path"] == "configs/chat_template.jinja"
    assert config_mod.harness_path(base["chat"]["template_path"]).is_file()
    # ChatML. The trailing newline is part of the marker because the role sits
    # on its own line; 04_train_sft.py masks on the token ids of these exact
    # strings, and they must be what the template emits at a turn boundary.
    assert base["chat"]["instruction_part"] == "<|im_start|>user\n"
    assert base["chat"]["response_part"] == "<|im_start|>assistant\n"


# --------------------------------------------------------------------------
# data.yaml
# --------------------------------------------------------------------------


def test_holdout_entries_are_v3_scenario_families():
    """The v3 holdout axis is a full `scenario_id`, and it spans work types.

    `<work_type>.<family>` rather than a bare family name, because two work
    types could name a family the same thing and `normalize_v3_record` sets
    `stem_family` to the whole scenario id. Spanning more than one work type is
    the point of the list: a single entry would make the generalisation number
    a one-topic artefact, which is why the v2 list named six programs.
    """
    from cosimo_ft.data_schema import stem_family

    data = config_mod.load_config(stage="data")["data"]
    families = data["holdout_scenario_families"]
    assert len(families) == len(set(families)) >= 2
    assert len({f.rsplit(".", 1)[0] for f in families}) >= 2, (
        "hold out families from at least two work types, or the unseen-family "
        "measurement describes one corner of the corpus"
    )
    for family in families:
        assert "." in family, f"{family!r} is not a <work_type>.<family> scenario id"
        assert stem_family(family) == family, (
            f"{family} carries a v_/cr_/m_ wrapper prefix; holding out a wrapper "
            "leaves the base stem in training"
        )
    # v3 has no stem wrappers, so the v1/v2 axis stays empty. The key itself
    # remains because it is the union input `splits.assign` takes -- v3's
    # scenario families flow through the same parameter.
    assert data["holdout_families"] == []


def test_split_fractions_and_verification_gate():
    data = config_mod.load_config(stage="data")["data"]
    assert data["val_frac"] == 0.01
    # Larger than val_frac on purpose: test_frac is taken from the exam records
    # alone (~30% of the mixed corpus), because the other four record types have
    # no final-answer value for the grader to read. At 1% the headline slice
    # would be ~390 items rather than ~650, widening its Wilson interval as a
    # side effect of a training-mix decision.
    assert data["test_frac"] > data["val_frac"]
    assert data["max_train_records"] is None
    assert data["drop_unverified"] is True


# --------------------------------------------------------------------------
# sft.yaml
# --------------------------------------------------------------------------


def test_lora_defaults_match_the_contract():
    cfg = config_mod.load_config(stage="sft")
    lora = cfg["lora"]
    assert lora["r"] == 32 and lora["lora_alpha"] == 32
    assert lora["lora_dropout"] == 0.0
    assert lora["bias"] == "none", "only 'none' stays mergeable for 08_export_merge.py"
    assert lora["use_rslora"] is False
    assert lora["use_gradient_checkpointing"] == "unsloth"
    assert lora["target_modules"] == "auto", (
        "projection naming is model-specific -- Phi-4-mini-reasoning fuses them "
        "(qkv_proj, gate_up_proj) where Qwen does not -- so a hardcoded module "
        "list matches nothing on one of the two. 'auto' resolves per model."
    )


def test_sft_optimizer_avoids_bitsandbytes():
    sft = config_mod.load_config(stage="sft")["sft"]
    assert sft["optim"] == "adamw_torch_fused"
    assert "8bit" not in sft["optim"]


def test_sft_effective_batch_size_is_32():
    sft = config_mod.load_config(stage="sft")["sft"]
    effective = sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"]
    assert effective == 32, "learning_rate 2e-4 was chosen for an effective batch of 32"


def test_sft_schedule_defaults():
    sft = config_mod.load_config(stage="sft")["sft"]
    assert sft["learning_rate"] == 2.0e-4
    assert sft["lr_scheduler_type"] == "cosine"
    assert sft["warmup_ratio"] == 0.03
    assert sft["num_train_epochs"] == 1
    assert sft["max_steps"] == -1
    assert sft["weight_decay"] == 0.01
    assert sft["max_grad_norm"] == 1.0
    assert sft["bf16"] is True and sft["fp16"] is False
    assert sft["group_by_length"] is False
    assert sft["seed"] == 3407
    # A list: the exam corpus from 01_prepare_data.py plus the synthetic
    # tool-calling rows from 02_prepare_tool_data.py, concatenated by
    # 04_train_sft.py.
    assert sft["train_file"] == [
        "data/processed/sft_train.jsonl",
        "data/processed/tool_train.jsonl",
    ]
    assert sft["val_file"] == [
        "data/processed/sft_val.jsonl",
        "data/processed/tool_val.jsonl",
    ]


# --------------------------------------------------------------------------
# dpo.yaml / orpo.yaml
# --------------------------------------------------------------------------


def test_dpo_loss_type_is_a_list():
    dpo = config_mod.load_config(stage="dpo")["dpo"]
    assert dpo["loss_type"] == ["sigmoid"], (
        "TRL 0.24.0 types DPOConfig.loss_type as list[str]"
    )
    assert dpo["beta"] == 0.1
    assert dpo["learning_rate"] == 5.0e-6
    assert dpo["optim"] == "adamw_torch_fused"
    assert dpo["train_file"] == "data/processed/pref_train.jsonl"


def test_dpo_prompt_budget_cannot_truncate_the_identity():
    cfg = config_mod.load_config(stage="dpo")
    dpo = cfg["dpo"]
    assert dpo["max_length"] == cfg["model"]["max_seq_length"]
    assert dpo["max_completion_length"] + dpo["max_prompt_length"] <= dpo["max_length"]
    # The identity block alone is ~600 tokens and TRL's keep_end truncation drops
    # the START of the prompt, which is exactly where the identity lives.
    assert dpo["max_prompt_length"] >= 1024


def test_orpo_is_a_complete_alternative_path():
    cfg = config_mod.load_config(stage="orpo")
    assert cfg["lora"]["target_modules"] == "auto"
    assert cfg["lora"]["r"] == 32, "must match sft.yaml or the comparison is confounded"
    orpo = cfg["orpo"]
    assert orpo["beta"] == 0.1
    assert orpo["learning_rate"] == 8.0e-6
    assert orpo["max_length"] == cfg["model"]["max_seq_length"]
    assert orpo["train_file"] == "data/processed/pref_train.jsonl"


# --------------------------------------------------------------------------
# eval.yaml
# --------------------------------------------------------------------------


def test_eval_suites_cover_in_domain_generalisation_and_regression():
    cfg = config_mod.load_config(stage="eval")["eval"]
    assert cfg["suites"] == [
        "cosimo_test",
        "cosimo_unseen_stems",
        "gsm8k",
        "math500",
    ]
    assert set(cfg["samples"]) == set(cfg["suites"])


def test_evaluation_is_deterministic_by_default():
    cfg = config_mod.load_config(stage="eval")["eval"]
    assert cfg["temperature"] == 0.0, "a base-vs-tuned delta must not be sampling noise"
    # Not 768: that truncated the long chain-of-thought base model on 90-97% of
    # items, so its accuracy measured the decoding budget rather than the model.
    # Not 4096 either: at a fixed batch_size that reserved 74 736 token slots of
    # KV cache and was OOM-killed on a 121 GB unified-memory machine.
    assert cfg["max_new_tokens"] == 2048, (
        "the base model is a long-CoT reasoner; a small cap turns its accuracy "
        "into a measurement of the budget and inflates every tuned-model delta"
    )
    # The bound that makes max_new_tokens safe to raise: without it, batch_size
    # does not constrain memory at all.
    assert cfg["max_batch_tokens"] == 24576
    assert cfg["rel_tol"] == 1.0e-3


def test_eval_suite_names_are_the_ones_the_loader_knows():
    from cosimo_ft import benchmarks

    cfg = config_mod.load_config(stage="eval")["eval"]
    assert set(cfg["suites"]) <= set(benchmarks.ALL_SUITES)
