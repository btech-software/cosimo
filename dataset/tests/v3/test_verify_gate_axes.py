"""The publish-gate axes that came with PR4: teacher pinning (14), the gold-bar
near-duplicate fence (13), and what ``--quick`` withholds.

These are the last two board axes and the CI speed switch, and each is a claim
that has to be *seen* to hold: a pin the board does not read is a silent swap
away from being a mix, a fence with nothing on the far side tests nothing, and a
``--quick`` that silently shaded an expensive axis green instead of reporting it
withheld would let CI certify a corpus it never fully checked.

The corpus is the same real one the rest of the board tests build -- prose
through the production render stage -- so the rows carry the provenance stamps
the axes read, and the tampers below are edits to the bytes on disk, the work
of a hand rather than of the compositor.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.render.prose import run_render_stage, select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verify_v3 import AXES, EXPENSIVE_AXES, verify_dir  # noqa: E402

N_ROWS = 6
LANE_ENVS = {
    config.TEACHER_PROSE_ENV: prose_harness.DUMMY_MODEL,
    config.TEACHER_REASONING_ENV: prose_harness.DUMMY_MODEL,
    config.LIVE_ENV: "0",
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """One rendered prose corpus through the production path, built once."""
    out = str(tmp_path_factory.mktemp("v3gate"))
    payload = prose_harness.build_fixture(("analysis",), N_ROWS)
    jobs = inventory.expand_jobs(inventory.load_plan())
    selected = select_prose_jobs(jobs, types=("analysis",), limit=None)[
        : payload["meta"]["jobs_examined"]
    ]
    stage.run_pack_stage(out, selected)
    saved = {key: os.environ.get(key) for key in LANE_ENVS}
    os.environ.update(LANE_ENVS)
    try:
        run_render_stage(
            out, selected, Teacher(FixtureTransport(entries=payload["entries"]))
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return out


def _fresh(corpus_dir, tmp_path) -> str:
    copy = os.path.join(str(tmp_path), "corpus")
    shutil.copytree(corpus_dir, copy)
    return copy


def _rows(out: str) -> list[dict]:
    return write.read_jsonl(write.path_for("sft", "analysis", out))


def _rewrite(out: str, rows: list[dict]) -> None:
    write.write_jsonl(write.path_for("sft", "analysis", out), rows)


def _fail_axes(report: dict) -> list[str]:
    return [name for _, name in AXES if report["axes"][name]["failures"]]


# --------------------------------------------------------- axis 14: pinning --


def test_the_prose_corpus_pins_one_teacher_and_reads_clean(corpus):
    report = verify_dir(corpus)
    assert "teacher pinned" not in _fail_axes(report)
    axis = report["axes"]["teacher pinned"]
    assert axis["checked"] == N_ROWS, (
        "every prose row claims a teacher; all are audited"
    )
    assert axis.get("skipped") is not True


def test_a_blanked_provenance_pin_is_a_teacher_axis_finding(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    rows[0]["verification"]["teacher"]["model"] = "   "  # the pin, sanded off
    _rewrite(out, rows)
    report = verify_dir(out)
    assert "teacher pinned" in _fail_axes(report)
    assert report["ok"] is False


def test_a_corpus_that_mixes_teachers_is_red_until_declared(
    corpus, tmp_path, monkeypatch
):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    first = rows[0]["verification"]["teacher"]["model"]
    rows[0]["verification"]["teacher"]["model"] = "quietly-swapped-mid-run"
    _rewrite(out, rows)

    report = verify_dir(out)
    assert "teacher pinned" in _fail_axes(report), (
        "two house styles must not pass as range -- the v2 lesson (spec §6 axis 14)"
    )

    # The same bytes, but now the operator has *said* the pair is on purpose.
    monkeypatch.setenv(config.TEACHER_ALLOWLIST_ENV, f"{first},quietly-swapped-mid-run")
    declared = verify_dir(out)
    assert "teacher pinned" not in _fail_axes(declared), (
        "an allowed mix is exactly what the allow-list is for"
    )


def test_a_composed_exam_row_is_not_held_to_a_teacher_pin(corpus, tmp_path):
    """``verification.teacher`` may be ``None`` where the row was composed, not
    dictated (an exam item). That is a claim of computer authorship, not an
    absent pin -- the axis must read it that way and stay quiet."""
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    for row in rows:
        row["verification"]["teacher"] = None
    _rewrite(out, rows)
    report = verify_dir(out)
    assert "teacher pinned" not in _fail_axes(report), (
        "a row that says it has no teacher must not be red for lacking one"
    )
    assert report["axes"]["teacher pinned"]["checked"] == 0


# ------------------------------------------------- axis 13: gold-bar fence --


def _gold_bar(tmp_path, items) -> str:
    path = os.path.join(str(tmp_path), "gold_bar_v3.jsonl")
    write.write_jsonl(path, items)
    return path


def test_a_train_row_that_echoes_a_gold_item_is_a_leak_finding(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    answer = _rows(out)[0]["answer"]
    bar = _gold_bar(tmp_path, [{"id": "gold-1", "answer": answer}])
    report = verify_dir(out, gold_bar_path=bar)
    assert "gold-bar near-dup" in _fail_axes(report), (
        "a training row that reads like a held-out item is a leak, not coverage"
    )


def test_a_disjoint_gold_bar_fences_clean(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    bar = _gold_bar(
        tmp_path,
        [
            {
                "id": "gold-1",
                "answer": (
                    "The charterholder reviewed the annuity ledger by hand and "
                    "signed the memorandum with no reference to any generated row."
                ),
            }
        ],
    )
    report = verify_dir(out, gold_bar_path=bar)
    assert "gold-bar near-dup" not in _fail_axes(report)
    assert report["axes"]["gold-bar near-dup"]["checked"] == N_ROWS


def test_a_missing_gold_bar_is_reported_not_red(corpus, tmp_path):
    """The board has no excuse to shade a corpus red for a human artefact it was
    never handed; ``publish`` carries that stricter duty (see its own tests)."""
    out = _fresh(corpus, tmp_path)
    missing = os.path.join(str(tmp_path), "absent_gold_bar.jsonl")
    report = verify_dir(out, gold_bar_path=missing)
    axis = report["axes"]["gold-bar near-dup"]
    assert "gold-bar near-dup" not in _fail_axes(report)
    assert axis["checked"] == 0 and axis.get("note"), (
        "no bar: the axis says what it could not measure, and stays silent"
    )


# --------------------------------------------------------- --quick contract --


def test_quick_withholds_the_expensive_axes_and_reports_them_skipped(corpus):
    report = verify_dir(corpus, quick=True)
    for name in EXPENSIVE_AXES:
        axis = report["axes"][name]
        assert axis.get("skipped") is True, f"{name} must report itself witheld"
        assert axis["failures"] == [] and axis["checked"] == 0
    # The cheap integrity axes still ran, and the corpus is still certified on them.
    assert report["axes"]["schema"].get("skipped") is not True
    assert report["axes"]["schema"]["checked"] == N_ROWS
    assert report["axes"]["teacher pinned"]["checked"] == N_ROWS


def test_quick_is_not_a_way_to_certify_a_publish(corpus, tmp_path):
    """A quick run must not *silently* pass an axis a full run would fail: the
    gate is a full run, and the publish gate relies on the expensive axes having
    run -- so a tampered hidden-test suite that full verify catches is the whole
    point of never publishing off a ``--quick`` report."""
    # The three witheld axes are exactly the ones publish must require to have
    # run; the assertion ties the CLI's --quick to the board's own vocabulary.
    report = verify_dir(corpus, quick=True)
    assert set(
        name for name in report["axes"] if report["axes"][name].get("skipped")
    ) == set(EXPENSIVE_AXES)
