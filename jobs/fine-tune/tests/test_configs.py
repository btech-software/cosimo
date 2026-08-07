"""The shipped YAML configs and the layered override surface.

These tests are the guard against a silent knob change: every value asserted
here is one the build contract fixes, or one whose default a reader of the
README is entitled to rely on.
"""

from __future__ import annotations

import pytest
import yaml

from cosimo_ft import config as config_mod

from conftest import CONFIG_DIR

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
    assert cfg["model"]["base_id"] == "unsloth/Phi-4-mini-reasoning"  # base survives
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
    assert base["model"]["base_id"] == "unsloth/Phi-4-mini-reasoning"
    # 8192, not 2048: the served target is a LangGraph ReAct loop whose
    # conversation accumulates tool calls and tool results on top of the persona.
    assert base["model"]["max_seq_length"] == 8192
    assert base["model"]["load_in_4bit"] is False, "bf16 LoRA is the locked default"
    assert base["model"]["dtype"] == "bfloat16"
    assert base["dataset"]["hub_id"] == "btech-software/cosimo-quant-reasoning-v2"


def test_the_corpus_is_the_mixed_v2_primary_plus_a_capped_v1(base):
    """v2 leads, v1 is a capped supplement, and only v2 supplies pairs.

    The share cap is the mechanism that keeps the corpus majority non-exam.
    Uncapping v1 would put 71k exam rows against v2's 24k and rebuild the
    exam-only corpus that collapsed the first run's response style.
    """
    dataset = base["dataset"]
    assert dataset["preference_config"] == "preference"
    assert len(dataset["mix"]) == 1
    v1 = dataset["mix"][0]
    assert v1["hub_id"] == "btech-software/cosimo-cfa-frm-71k"
    assert 0.0 < v1["max_share"] <= 0.5
    # v1's pairs share ids with its supervised rows, which is the overlap that
    # made the first DPO run a zero-gradient no-op. v2's do not.
    assert v1["preference_config"] is None


def test_identity_block_is_the_contracted_persona(base):
    identity = base["prompt"]["identity"]
    assert identity.startswith(
        "You are Cosimo, a financial domain expert AI assistant created by "
        "Btech Software."
    )
    # Spot-check each paragraph so a trimmed persona is caught.
    for phrase in (
        "Head of Quantitative Asset Management",
        "You are also a game theorist",
        "von Neumann, Nash, and Aumann",
        "You are also a research engineer",
        "brutally honest about what you don't know",
    ):
        assert phrase in identity
    assert "Microsoft" not in identity


def test_short_identity_is_the_one_line_variant(base):
    assert base["prompt"]["identity_short"] == (
        "You are Cosimo, a financial domain expert AI assistant created by "
        "Btech Software."
    )


def test_exam_protocol_carries_the_grading_contract(base):
    protocol = base["prompt"]["exam_protocol"]
    assert protocol.startswith("Solve the problem step by step")
    assert protocol.rstrip().endswith("FINAL ANSWER: <value>")
    assert "\nFINAL ANSWER: <value>" in protocol, "the contract must be its own line"
    assert base["prompt"]["final_answer_tag"] == "FINAL ANSWER:"


def test_variation_rate_is_fifteen_percent(base):
    assert base["prompt"]["variation_rate"] == 0.15


def test_chat_template_override_is_configured(base):
    assert base["chat"]["template_path"] == "configs/chat_template.jinja"
    assert config_mod.harness_path(base["chat"]["template_path"]).is_file()
    assert base["chat"]["instruction_part"] == "<|user|>"
    assert base["chat"]["response_part"] == "<|assistant|>"


# --------------------------------------------------------------------------
# data.yaml
# --------------------------------------------------------------------------


def test_holdout_entries_are_families_not_generator_names():
    from cosimo_ft.data_schema import stem_family

    data = config_mod.load_config(stage="data")["data"]
    families = data["holdout_families"]
    assert len(families) == len(set(families)) == 6
    for family in families:
        assert stem_family(family) == family, (
            f"{family} carries a v_/cr_/m_ wrapper prefix; holding out a wrapper "
            "leaves the base stem in training"
        )


def test_split_fractions_and_verification_gate():
    data = config_mod.load_config(stage="data")["data"]
    assert data["val_frac"] == 0.01
    assert data["test_frac"] == 0.01
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
        "unsloth/Phi-4-mini-reasoning has fused projections (qkv_proj, o_proj, "
        "gate_up_proj, down_proj); a hardcoded 7-module list matches nothing"
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
