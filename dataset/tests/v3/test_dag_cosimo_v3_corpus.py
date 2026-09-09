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
import re
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
    """target -> command for each ``v3-*`` stage that shells the corpus CLI."""
    recipes: dict[str, str] = {}
    with open(_MAKEFILE, encoding="utf8") as handle:
        lines = handle.read().splitlines()
    target = re.compile(r"^([A-Za-z0-9_-]+):\s*$")
    for index, line in enumerate(lines):
        match = target.match(line)
        if not match:
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if following.startswith("\t") and "dataset.pipelines.v3.cli" in following:
            recipes[match.group(1)] = following.strip()
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
