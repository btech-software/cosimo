"""The optional ``cosimo_v3_corpus`` DAG -- tested without needing Airflow installed.

The DAG module keeps two faces: a set of *pure* command builders and constants
that import in any venv, and a DAG object that only exists when ``apache-airflow``
does (which it deliberately is not -- it is not in the locked groups, only in
``ops/airflow/requirements.txt``). So the load-bearing invariants -- that every
task runs the very command the Makefile runs, that the fan-out maps the axes the
DAG claims to map, that the gate is dry-run by default -- are checked unconditionally
here, and only the wiring (task ids, pools, the dependency chain) is checked under
``pytest.importorskip("airflow")``. A DAG whose command strings drift from the
Makefile is the real risk this feature carries; that is what gets an unguarded test.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from pipelines.v3 import config, inventory

_REPO = os.path.dirname(config.BASE_DIR)
_DAG_DIR = os.path.join(_REPO, "ops", "airflow", "dags")
_MAKEFILE = os.path.join(_REPO, "Makefile")

if _DAG_DIR not in sys.path:
    sys.path.insert(0, _DAG_DIR)

import cosimo_v3_corpus as dag  # noqa: E402  (needs the path bootstrap just above)


def _makefile_recipes() -> dict[str, str]:
    """target -> the command ``make`` would actually run, for each v3-* stage.

    Expanded through ``make -n`` rather than read off the file. The recipes
    carry variables now (amendment §F gave the targets TYPES/LIMIT/WORK/OUT/
    LIVE/QUICK/HOLDOUT), so the literal text of a recipe line is
    ``$(V3) render $(V3_OUT) ...`` and comparing the DAG against *that* would
    compare it against a template nobody runs. ``make -n`` with no variables
    set prints the un-scoped line, which is exactly the command the DAG's
    un-scoped task must equal -- and it also proves the empty-by-default
    variables really do expand to nothing.
    """
    recipes: dict[str, str] = {}
    for target in (
        "v3-inventory",
        "v3-packs",
        "v3-smoke",
        "v3-render",
        "v3-verify",
        "v3-prefer",
        "v3-publish",
    ):
        printed = subprocess.run(
            ["make", "-s", "-n", target],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in printed.splitlines():
            if "dataset.pipelines.v3.cli" in line:
                # Collapse the runs of spaces an unset variable leaves behind:
                # `render $(V3_OUT) $(V3_TYPES)` with both empty prints as
                # `render   `, and the DAG builds its argv from a list.
                recipes[target] = " ".join(line.split())
    return recipes


@pytest.fixture(scope="module")
def recipes():
    return _makefile_recipes()


def test_the_dag_runs_the_very_commands_the_makefile_runs(recipes):
    # "works under make" == "schedulable": the un-scoped task command must be the
    # Makefile recipe byte-for-byte, or the DAG quietly schedules something else.
    assert dag.inventory_command() == recipes["v3-inventory"]
    assert dag.verify_command() == recipes["v3-verify"]
    assert dag.prefer_command() == recipes["v3-prefer"]
    assert dag.publish_command(dry_run=False) == recipes["v3-publish"]
    assert dag.packs_command() == recipes["v3-packs"]
    assert dag.render_command() == recipes["v3-render"]


def test_mapped_tasks_differ_from_their_base_only_by_their_scoping_flag(recipes):
    for work_type in dag._plan_work_types():
        assert dag.packs_command(work_type) == (
            f"{recipes['v3-packs']} --work-type {work_type}"
        )
    for record_type in dag.RENDER_TYPES:
        assert dag.render_command(record_type) == (
            f"{recipes['v3-render']} --types {record_type}"
        )


def test_the_render_fanout_is_exactly_the_lanes_render_accepts():
    # If a lane is added to the CLI and forgotten in the DAG, a whole record type
    # silently never renders; pin the DAG's map against the CLI's own validator.
    from pipelines.v3 import cli

    accepted = set(cli.BRIEF_KINDS) | {cli.AGENTIC_KIND, cli.EXAM_KIND, cli.IMPL_KIND}
    assert set(dag.RENDER_TYPES) == accepted


def test_the_packs_fanout_comes_from_the_taxonomy_not_a_hardcoded_list():
    assert dag._plan_work_types() == sorted(inventory.load_plan(config.taxonomy_path()))
    assert len(dag._plan_work_types()) == 5, "five work types on the current plan"


def test_the_publish_gate_is_dry_run_unless_an_operator_flips_it(monkeypatch):
    monkeypatch.delenv("COSIMO_V3_PUBLISH_DRY_RUN", raising=False)
    assert dag._publish_dry_run() is True
    assert dag.publish_command(dry_run=dag._publish_dry_run()).endswith("--dry-run")

    monkeypatch.setenv("COSIMO_V3_PUBLISH_DRY_RUN", "0")
    assert dag._publish_dry_run() is False
    assert "--dry-run" not in dag.publish_command(dry_run=dag._publish_dry_run())


def test_teacher_retry_budget_is_bounded_and_parses(monkeypatch):
    monkeypatch.delenv("COSIMO_V3_TEACHER_RETRIES", raising=False)
    assert dag._teacher_retries() == 2
    monkeypatch.setenv("COSIMO_V3_TEACHER_RETRIES", "5")
    assert dag._teacher_retries() == 5
    monkeypatch.setenv("COSIMO_V3_TEACHER_RETRIES", "0")
    assert dag._teacher_retries() == 0
    monkeypatch.setenv("COSIMO_V3_TEACHER_RETRIES", "not-a-number")
    assert dag._teacher_retries() == 2, "a bad value must fall back, not crash parse"


@pytest.mark.skipif(
    not dag.HAS_AIRFLOW,
    reason="apache-airflow is not installed in this venv (optional extra)",
)
def test_the_dag_wires_the_six_stages_and_two_mapped_waves():
    assert dag.dag is not None and dag.dag.dag_id == dag.DAG_ID
    by_id = {task.task_id: task for task in dag.dag.tasks}

    for task_id in ("inventory", "verify", "prefer", "publish"):
        assert task_id in by_id, task_id
    for work_type in dag._plan_work_types():
        assert f"packs.{work_type}" in by_id, work_type
    for record_type in dag.RENDER_TYPES:
        assert f"render.{record_type}" in by_id, record_type

    # pools: the teacher-bound wave is where the RPM budget lives; the rest is cpu.
    for task_id, task in by_id.items():
        expected = (
            dag.TEACHER_POOL
            if task_id.startswith("render.") or task_id == "prefer"
            else dag.CPU_POOL
        )
        assert task.pool == expected, task_id
        retries = task.retries if task.retries is not None else 0
        expected_retries = (
            dag._teacher_retries()
            if task_id.startswith("render.") or task_id == "prefer"
            else 0
        )
        assert retries == expected_retries, task_id

    packs_ids = {f"packs.{wt}" for wt in dag._plan_work_types()}
    render_ids = {f"render.{rt}" for rt in dag.RENDER_TYPES}
    assert by_id["inventory"]  # root
    for packs_id in packs_ids:
        assert "inventory" in by_id[packs_id].upstream_task_ids, packs_id
    for render_id in render_ids:
        assert packs_ids <= by_id[render_id].upstream_task_ids, render_id
    assert render_ids <= by_id["verify"].upstream_task_ids
    assert "verify" in by_id["prefer"].upstream_task_ids
    assert "prefer" in by_id["publish"].upstream_task_ids


def test_the_dag_renders_on_the_fixture_unless_a_variable_says_otherwise():
    """Amendment §F: the licence to spend is a deployment Variable, not a repo line.

    ``dataset_build.sh`` used to export ``COSIMO_V3_LIVE=1`` at the top of the
    file, which committed the licence to git and handed it to anything that
    sourced it -- a scheduled run included. The DAG asks the deployment
    instead, and the answer without Airflow (or without the Variable set) is
    the safe one.
    """
    assert "--live" not in dag.render_command(live=False)
    assert "--live" not in dag.prefer_command(live=False)
    assert dag.render_command(live=True).endswith("--live")
    # No Airflow in the locked test group, so the accessor must answer False
    # rather than raise -- a DAG module that cannot be imported without a
    # scheduler is a DAG module nobody can unit-test.
    assert dag.live_render_enabled() is False
    assert "--live" not in dag.render_command()


def test_the_eval_tree_is_a_separate_render_not_a_flag_on_the_training_one():
    """§E: holdout families render, and they render somewhere else."""
    assert dag.render_command(holdout=True, live=False).endswith("--holdout")
    assert "--holdout" not in dag.render_command(live=False)
