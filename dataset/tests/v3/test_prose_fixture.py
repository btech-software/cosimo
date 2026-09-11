"""The committed dummy's two duties: regenerate byte-identical, and carry the
50-row PR2 gate (arch spec §10: "50 analysis rows, invented-number rate 0 on
the dummy").

The drift test is not ceremony: a fixture nobody can regenerate is folklore,
and folklore with hashes gets copy-pasted until it hardens into a gold bar
that fails alone. The gate test then looks at the rendered corpus through the
*audit* lens -- the same recomputed-pack path ``verify_v3`` axis 3 uses -- so
"rate 0" is a statement about the corpus, not about the transport that fed it.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.prose import (  # noqa: E402
    row_id_from_coords,
    run_render_stage,
    select_prose_jobs,
)
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verification.invented_numbers import invented_numbers  # noqa: E402
from pipelines.v3.verification.prose import missing_mentions, whitelist_for  # noqa: E402

COMMITTED = os.path.join(_HERE, "fixtures", prose_harness.PROSE_FIXTURE_NAME)


def _regenerate_bytes(payload: dict, tmp_path) -> bytes:
    path = os.path.join(str(tmp_path), "regen.json")
    with open(path, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    with open(path, "rb") as handle:
        return handle.read()


def test_the_committed_fixture_is_byte_identical_to_its_harness(tmp_path):
    payload = prose_harness.build_fixture(
        tuple(prose_harness.DEFAULT_TYPES), prose_harness.DEFAULT_LIMIT
    )
    with open(COMMITTED, "rb") as handle:
        committed = handle.read()
    assert _regenerate_bytes(payload, tmp_path) == committed, (
        "the committed fixture drifted from its harness; regenerate it with "
        ".venv/bin/python dataset/tests/v3/fixtures/make_prose_fixture.py "
        "and inspect the diff before accepting it"
    )


def test_fifty_rows_render_and_the_invented_number_rate_is_exactly_zero(
    tmp_path, monkeypatch
):
    payload = json.load(open(COMMITTED, encoding="utf8"))
    # `train_rows`, not `entries`: the table now also carries a holdout arm so
    # the operator's `eval-slice` mode can be run offline (amendment §E gave
    # holdout families a render path, and a path with no fixture behind it can
    # only be exercised by billing a teacher). The PR2 gate is still fifty
    # *train* rows, exactly.
    assert payload["meta"]["train_rows"] == 50, (
        "the PR2 gate is fifty rows, not approximately"
    )
    assert payload["meta"]["holdout_rows"] > 0, (
        "the eval tree needs a fixture behind it or `eval-slice` is live-only"
    )
    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(inventory.load_plan())
    selected = select_prose_jobs(jobs, types=("analysis",), limit=None)[
        : payload["meta"]["jobs_examined"]
    ]
    stage.run_pack_stage(out, selected)
    monkeypatch.setenv(config.TEACHER_PROSE_ENV, payload["model"])
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, payload["model"])
    monkeypatch.delenv(config.LIVE_ENV, raising=False)
    report = run_render_stage(out, selected, Teacher(FixtureTransport(path=COMMITTED)))
    assert report["rendered"] == 50
    assert report["missing_packs"] == [] and report["dead_lettered"] == 0

    rows = write.read_jsonl(write.path_for("sft", "analysis", out))
    assert len(rows) == 50 and len({row["id"] for row in rows}) == 50
    offenders: list[str] = []
    uncovered: list[str] = []
    misderived: list[str] = []
    for row in rows:
        family = row["scenario_id"][len(row["work_type"]) + 1 :]
        pack = compute_pack(row["work_type"], family, row["variant"])
        offenders += [
            f"{row['id']}: {token!r}"
            for token in invented_numbers(
                row["answer"], pack.allowed_numbers, whitelist_for(pack.to_dict())
            )
        ]
        uncovered += [
            f"{row['id']}: {point!r}"
            for point in missing_mentions(pack.to_dict(), row["answer"])
        ]
        if row["id"] != row_id_from_coords(
            row["record_type"], row["work_type"], family, row["variant"]
        ):
            misderived.append(row["id"])
    assert offenders == [], f"invented-number rate must be exactly 0: {offenders[:5]}"
    assert uncovered == [], f"the contract demands every must_mention: {uncovered[:5]}"
    assert misderived == [], "every id must hash from its own coordinates"
