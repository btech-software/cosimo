"""Deterministic splitting, and the two leaks it exists to prevent.

The corpus has only 71 generator stems, so an IID split cannot measure
generalisation to unseen question structures: whole stem *families* are held out
instead. The other leak is the preference config, whose ids are the ids of the
``default`` config — the same assignment must be reused, or a question can be in
DPO training and in the test set at once.
"""

from __future__ import annotations

import pytest

from cosimo_ft import splits

PROGRAMS = {
    "CFA_Level_I": ["tvm_annuity_fv", "v_tvm_annuity_fv", "fi_modified_duration"],
    "CFA_Level_II": ["eq_fcff_dcf", "deriv_bsm_call"],
    "FRM_Part_1": ["mkt_cvar", "mkt_param_var"],
}
HOLDOUT = {"fi_modified_duration", "deriv_bsm_call", "mkt_cvar"}


def corpus(per_generator: int = 120) -> list[dict]:
    records = []
    for program, generators in PROGRAMS.items():
        for generator in generators:
            for index in range(per_generator):
                records.append(
                    {
                        "id": f"{generator}-{index:05d}",
                        "program": program,
                        "generator": generator,
                    }
                )
    return records


def assign(records=None, *, seed: int = 3407, val=0.1, test=0.1, holdout=HOLDOUT):
    return splits.assign_splits(
        records if records is not None else corpus(),
        val_frac=val,
        test_frac=test,
        seed=seed,
        holdout_families=holdout,
    )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_assignment_is_reproducible_for_a_seed():
    assert assign() == assign()


def test_a_different_seed_moves_records():
    assert assign(seed=3407) != assign(seed=11)


def test_input_order_does_not_change_the_assignment():
    records = corpus()
    shuffled = list(reversed(records))
    assert assign(records) == assign(shuffled)


def test_every_record_is_assigned_exactly_once():
    records = corpus()
    assignment = assign(records)
    assert set(assignment) == {r["id"] for r in records}
    assert len(assignment) == len(records)
    assert set(assignment.values()) <= set(splits.SPLIT_NAMES)


def test_duplicate_ids_keep_their_first_assignment():
    records = corpus()
    assignment = assign(records + records)
    assert assignment == assign(records)


# --------------------------------------------------------------------------
# the held-out stem families
# --------------------------------------------------------------------------


def test_holdout_families_never_reach_a_training_split():
    assignment = assign()
    for record in corpus():
        family = splits.stem_family(record["generator"])
        if family in HOLDOUT:
            assert assignment[record["id"]] == splits.UNSEEN_STEMS
        else:
            assert assignment[record["id"]] in splits.TRAINABLE_SPLITS


def test_a_wrapper_variant_is_held_out_with_its_base_stem():
    # v_tvm_annuity_fv is the same question structure as tvm_annuity_fv; holding
    # out only the base generator would leak it straight back into training.
    assignment = assign(holdout={"tvm_annuity_fv"})
    for record in corpus():
        if record["generator"] in ("tvm_annuity_fv", "v_tvm_annuity_fv"):
            assert assignment[record["id"]] == splits.UNSEEN_STEMS


def test_an_explicit_stem_family_column_is_honoured():
    records = [
        {"id": "a", "program": "P", "generator": "opaque_name", "stem_family": "held"},
        {"id": "b", "program": "P", "generator": "opaque_name", "stem_family": "kept"},
    ]
    assignment = assign(records, holdout={"held"})
    assert assignment["a"] == splits.UNSEEN_STEMS
    assert assignment["b"] in splits.TRAINABLE_SPLITS


def test_no_holdout_families_means_no_unseen_split():
    assignment = assign(holdout=set())
    assert splits.UNSEEN_STEMS not in set(assignment.values())


# --------------------------------------------------------------------------
# ratios
# --------------------------------------------------------------------------


def test_ratios_are_approximately_honoured():
    assignment = assign(val=0.1, test=0.1)
    pool = [s for s in assignment.values() if s != splits.UNSEEN_STEMS]
    n = len(pool)
    assert pool.count(splits.TEST) / n == pytest.approx(0.1, abs=0.01)
    assert pool.count(splits.VAL) / n == pytest.approx(0.1, abs=0.01)
    assert pool.count(splits.TRAIN) / n == pytest.approx(0.8, abs=0.02)


def test_small_fractions_still_produce_non_empty_eval_splits():
    # 1% is the shipped configs/data.yaml setting; per-stratum rounding must not
    # starve the small splits.
    assignment = assign(val=0.01, test=0.01)
    counts = {
        name: list(assignment.values()).count(name) for name in splits.SPLIT_NAMES
    }
    assert counts[splits.TEST] > 0 and counts[splits.VAL] > 0


def test_every_stratum_contributes_to_the_training_split():
    assignment = assign(val=0.1, test=0.1)
    trained = {
        record["generator"]
        for record in corpus()
        if assignment[record["id"]] == splits.TRAIN
    }
    assert trained == {
        generator
        for generators in PROGRAMS.values()
        for generator in generators
        if splits.stem_family(generator) not in HOLDOUT
    }


def test_invalid_fractions_are_refused():
    with pytest.raises(ValueError):
        assign(val=-0.1)
    with pytest.raises(ValueError):
        assign(test=1.0)
    with pytest.raises(ValueError):
        assign(val=0.6, test=0.6)


# --------------------------------------------------------------------------
# preference rows inherit the split of their id
# --------------------------------------------------------------------------


def test_preference_rows_inherit_the_split_of_their_id():
    records = corpus()
    assignment = assign(records)
    # The preference config carries the same ids in a different order and with
    # no generator column; 01_prepare_data.py looks each one up in this mapping.
    pref_ids = [r["id"] for r in reversed(records) if r["id"].endswith(("0", "5"))]
    for pref_id in pref_ids:
        assert assignment[pref_id] == assign(records)[pref_id]
    trainable = [i for i in pref_ids if assignment[i] in (splits.TRAIN, splits.VAL)]
    held_out = [i for i in pref_ids if assignment[i] == splits.UNSEEN_STEMS]
    assert trainable, "the fixture must produce some trainable preference rows"
    assert not set(trainable) & set(held_out)
    assert all(assignment[i] != splits.TEST for i in trainable)
