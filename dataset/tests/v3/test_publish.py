"""``publish`` is the last gate: it must refuse loudly and certify only what cleared.

The board *reports* (a missing gold bar is a note, not a red); publish *refuses*.
That difference is the whole reason they are two calls and not one, and each of
these tests pins one side of it -- the corpus that may go out, and the several
that may not: no fence to lean on, an axis gone red, and nothing at all to
publish. And a certification that had to write nothing to prove it certified
only what it had checked.
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
from pipelines.v3 import cli, config, inventory, stage, write  # noqa: E402
from pipelines.v3.render.prose import run_render_stage, select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402

N_ROWS = 6
LANE_ENVS = {
    config.TEACHER_PROSE_ENV: prose_harness.DUMMY_MODEL,
    config.TEACHER_REASONING_ENV: prose_harness.DUMMY_MODEL,
    config.LIVE_ENV: "0",
}
#: A held-out passage with nothing in common with a generated analysis -- the
#: fence has a far side, and nothing on the corpus's near side reaches it.
DISJOINT_GOLD = (
    "The charterholder read the annuity ledger by hand, cross-footed the "
    "column with a pencil, and signed the memorandum without opening a single "
    "generated row."
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("v3pub"))
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


def _outerr(capsys) -> str:
    """stdout+stderr of the captured run as one string (readouterr is a tuple)."""
    captured = capsys.readouterr()
    return captured.out + captured.err


def _stage_gold(tmp_path, answer: str = DISJOINT_GOLD) -> str:
    """A synthetic human bar with one clearly disjoint item, as an env override."""
    path = os.path.join(str(tmp_path), "gold_bar_v3.jsonl")
    write.write_jsonl(path, [{"id": "gold-1", "answer": answer}])
    return path


def test_publish_refuses_where_the_gold_bar_fence_is_missing(
    corpus, tmp_path, capsys, monkeypatch
):
    out = _fresh(corpus, tmp_path)
    monkeypatch.setenv(config.GOLDBAR_ENV, os.path.join(str(tmp_path), "absent.jsonl"))
    monkeypatch.setenv(config.PUBLISH_DIR_ENV, os.path.join(str(tmp_path), "publish"))
    assert cli.main(["publish", "--dry-run", "--out", out]) == cli.EXIT_DATA
    printed = _outerr(capsys)
    assert "PUBLISH REFUSED" in printed
    assert "gold bar" in printed, "a corpus cannot publish without the fence standing"


def test_publish_refuses_an_axis_gone_red_even_with_the_fence_up(
    corpus, tmp_path, capsys, monkeypatch
):
    out = _fresh(corpus, tmp_path)
    monkeypatch.setenv(config.GOLDBAR_ENV, _stage_gold(tmp_path))
    monkeypatch.setenv(config.PUBLISH_DIR_ENV, os.path.join(str(tmp_path), "publish"))
    rows = write.read_jsonl(write.path_for("sft", "analysis", out))
    # A prose row has one text and it is `answer`, so trailing whitespace no
    # longer "drifts from its assistant turn" -- there is no turn (§A). The
    # tamper that still breaks an axis is the one the amendment cares about:
    # a number the fact pack never authorised.
    rows[0]["answer"] = rows[0]["answer"] + " The breakeven is 9751.6362."
    write.write_jsonl(write.path_for("sft", "analysis", out), rows)
    assert cli.main(["publish", "--dry-run", "--out", out]) == cli.EXIT_DATA
    printed = _outerr(capsys)
    assert "PUBLISH REFUSED" in printed
    assert "invented numbers" in printed, (
        "the red axis must be named, not silently swallowed"
    )


def test_publish_refuses_a_corpus_with_nothing_in_it(tmp_path, capsys, monkeypatch):
    out = os.path.join(str(tmp_path), "empty")
    os.makedirs(out)
    monkeypatch.setenv(config.GOLDBAR_ENV, _stage_gold(tmp_path))
    monkeypatch.setenv(config.PUBLISH_DIR_ENV, os.path.join(str(tmp_path), "publish"))
    assert cli.main(["publish", "--dry-run", "--out", out]) == cli.EXIT_DATA
    assert "empty" in _outerr(capsys), (
        "a green board over no rows must not be a publish"
    )


def test_publish_dry_run_certifies_a_cleared_corpus_and_writes_nothing(
    corpus, tmp_path, capsys, monkeypatch
):
    out = _fresh(corpus, tmp_path)
    publish_dir = os.path.join(str(tmp_path), "publish")
    monkeypatch.setenv(config.GOLDBAR_ENV, _stage_gold(tmp_path))
    monkeypatch.setenv(config.PUBLISH_DIR_ENV, publish_dir)
    assert cli.main(["publish", "--dry-run", "--out", out]) == cli.EXIT_OK
    assert "would write" in _outerr(capsys)
    assert not os.path.isdir(publish_dir) or not os.listdir(publish_dir), (
        "a dry run that wrote the card was not a dry run"
    )


def test_publish_writes_the_card_only_when_the_corpus_cleared(
    corpus, tmp_path, capsys, monkeypatch
):
    out = _fresh(corpus, tmp_path)
    publish_dir = os.path.join(str(tmp_path), "publish")
    monkeypatch.setenv(config.GOLDBAR_ENV, _stage_gold(tmp_path))
    monkeypatch.setenv(config.PUBLISH_DIR_ENV, publish_dir)
    monkeypatch.setenv(config.HUB_REPO_ID_ENV, "cosimo/v3-corpus-2026-09")
    assert cli.main(["publish", "--out", out]) == cli.EXIT_OK
    card = os.path.join(publish_dir, config.DATASET_CARD_FILENAME)
    assert os.path.isfile(card), "a cleared corpus must leave its public card behind"
    with open(card, encoding="utf8") as handle:
        text = handle.read()
    assert "cosimo/v3-corpus-2026-09" in text, "the card names the hub it is for"
    assert "analysis" in text and "gold-bar near-dup" in text, (
        "the card reports the counts and the gates that cleared"
    )
