"""Invariants every fact computer must satisfy, swept over the whole plan.

The sweep is small by design (a few variants per family); determinism and
recomputation get their volume in CI via the smoke inventory, not by burning
developer time in a unit loop. What this file pins is the *contract*: a pack
is JSON, its seed is the seed the registry would derive, every number printed
in its question is an allowed number, and the only thing a computer may raise
is PackError.
"""

import json
import math

import pytest

import yaml

from pipelines.v3.config import SCHEMA_VERSION, VALID_REGISTERS, out_dir  # noqa: E402
from pipelines.v3.packs import COMPUTERS, FAMILIES, PackError, compute_pack  # noqa: E402
from pipelines.v3.seed import pack_seed  # noqa: E402
from verification.nums import nums  # noqa: E402

VARIANTS_PER_CELL = 12
TAXONOMY_FILE = "work_types.yaml"
KNOWN_COMPUTERS = {
    "valuation_fcff",
    "valuation_multiples",
    "risk_var",
    "portfolio_attribution",
    "execution_tca",
}


def _taxonomy() -> dict:
    import os
    from pipelines.v3.config import BASE_DIR

    with open(os.path.join(BASE_DIR, "taxonomy", TAXONOMY_FILE)) as f:
        return yaml.safe_load(f.read())


def _isclose(value: float, options: set[float]) -> bool:
    return any(math.isclose(value, o, rel_tol=1e-9, abs_tol=1e-9) for o in options)


def test_registry_matches_the_plan():
    """The plan file and the code registry agree, or generation must not start."""
    spec = _taxonomy()
    assert set(COMPUTERS) == set(spec), (
        f"work_types.yaml plans {sorted(spec)}, COMPUTERS registers {sorted(COMPUTERS)}"
    )
    assert set(FAMILIES) == set(spec)
    for work_type, cell in spec.items():
        planned = set(cell["families"])
        assert planned == set(FAMILIES[work_type]), (
            f"{work_type}: planned families {sorted(planned)} != computer's "
            f"{sorted(FAMILIES[work_type])}"
        )
        assert cell["computer"] in KNOWN_COMPUTERS, (
            f"{work_type}: plans computer {cell['computer']!r}, which is not a "
            f"registered module of pipelines.v3.packs"
        )
        assert any(f["holdout"] for f in cell["families"].values()), (
            f"{work_type}: no holdout family -- unseen_scenario_family would "
            "silently measure nothing"
        )
        assert 0.0 < cell["max_share"] <= 0.03


@pytest.mark.parametrize("work_type", sorted(COMPUTERS))
def test_pack_invariants_over_a_sweep(work_type):
    computer = COMPUTERS[work_type]
    for family in FAMILIES[work_type]:
        built = 0
        for variant in range(VARIANTS_PER_CELL):
            try:
                pack = computer(family, variant)
            except PackError:
                continue  # skipped variant: the inventory's path, not a failure
            built += 1
            data = json.loads(pack.to_json())
            assert data["schema_version"] == SCHEMA_VERSION
            assert data["scenario_id"] == f"{work_type}.{family}"
            assert data["work_type"] == work_type
            assert data["seed"] == pack_seed(work_type, family, variant)
            assert data["variant"] == variant
            assert data["register"] in VALID_REGISTERS
            assert pack.entities and all(isinstance(e, dict) for e in pack.entities)
            assert pack.formulas and pack.must_mention and pack.forbidden_claims
            assert isinstance(pack.question, str) and pack.question.strip()
            assert pack.allowed_numbers == sorted(
                set(float(x) for x in pack.allowed_numbers)
            )
            assert all(math.isfinite(x) for x in pack.allowed_numbers)
            # The central promise: nothing the question prints is outside the
            # allow-list. Exact membership -- the generator prints what it
            # allows, to the last digit, or the gate's tolerance is a lie.
            allowed = pack.number_set()
            for token in nums(pack.question):
                assert _isclose(token, allowed), (
                    f"{work_type}/{family}/{variant}: question prints {token!r} "
                    f"which is not in allowed_numbers"
                )


@pytest.mark.parametrize("work_type", sorted(COMPUTERS))
def test_recompute_is_idempotent(work_type):
    """Two computes of the same key are byte-identical (spec §5.3, verify axis 2).

    If this ever fails on a machine, that machine cannot host the corpus: the
    publish gate re-imports the computer and compares against the shard.
    """
    computer = COMPUTERS[work_type]
    for family in FAMILIES[work_type]:
        for variant in (0, 3, 11):
            try:
                a = computer(family, variant).to_json()
            except PackError:
                continue
            b = computer(family, variant).to_json()
            assert a == b, f"{work_type}/{family}/{variant} is not deterministic"


def test_unregistered_work_type_is_packerror_not_keyerror():
    with pytest.raises(PackError, match="no fact computer"):
        compute_pack("does.not.exist", "whatever", 0)


def test_smoke_out_dir_is_absolute_and_honours_env(monkeypatch):
    import os

    target = os.path.join(os.sep, "tmp", "opencode", "v3-shards-test")
    monkeypatch.setenv("COSIMO_V3_OUT", target)
    assert out_dir() == target
    monkeypatch.delenv("COSIMO_V3_OUT")
    assert os.path.isabs(out_dir())
