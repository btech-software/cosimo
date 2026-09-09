"""``cosimo_v3_corpus`` -- the optional Airflow wrapper around the v3 corpus CLI (spec §7).

This is the PR4 deliverable "Airflow DAG that shells the CLI". It is *optional*:
the corpus CLI stays the source of truth (decision log §14.3, "Airflow wraps CLI;
CLI does not import Airflow"), so a Spark box and CI run the same commands with
no scheduler at all. Nothing here is imported by ``dataset/pipelines/v3``; the
dependency runs one way only, toward Airflow, and the import of Airflow itself is
confined to :func:`build_dag` below so this module stays importable (and its
command builders testable) in a venv that has never seen ``apache-airflow`` --
which is the case for the locked ``corpus``/``test`` groups. The optional
dependency is declared in ``ops/airflow/requirements.txt``, never in the locked
groups, so the generation box is never dragged toward a scheduler it does not
need (the same reason ``torch`` stays out of the corpus image, spec §9).

The graph is one CLI command per task, chained on the exit-code contract the CLI
publishes (``cli.py``: exit 0 means the bytes on disk are good, anything else
means do not schedule the downstream -- so ``BashOperator`` failing on a non-zero
exit *is* the gate; the ``publish`` task is simply the last such gate):

    inventory >> packs(work_type)... >> render(record_type)... >> verify
              >> prefer >> publish

Two mapped waves, each along its write-disjoint axis, are what keeps the fan-out
honest -- the axis is chosen by which shard each task alone owns, not by the
neatness of the grid:

* ``packs`` fans out over **work type**. Each task writes only
  ``fact_packs/<work_type>.jsonl`` (``stage.run_pack_stage``), and the "bad
  computer does not block risk while valuation renders" the spec names
  (§7:400) is exactly a per-computer -- i.e. per-work-type -- worry.
* ``render`` fans out over **record type** using the CLI's own ``--types``.
  Each lane writes only ``sft/<record_type>.jsonl``, so the eight tasks never
  touch one another's shard. Mapping render by *work type* instead would have
  N concurrent writers read-merge-replacing the same ``sft/analysis.jsonl`` and
  silently losing appends -- a shard must have one writer (AGENTS §5.3) -- so
  record type, not work type, is the safe parallel axis for the teacher stages.

XCom carries no completion bytes anywhere: every task resolves the same absolute
shard tree from ``config.out_dir()`` (derived from the module file, never the
worker's CWD) and reads and writes only disk, honouring §6.1's "XCom / object
store ... do not put completions in XCom" by keeping data out of XCom entirely.

Concurrency is a pool concern, not a code concern: the teacher-bound tasks
(``render``, ``prefer``) take the ``teacher`` pool sized to the endpoint's RPM,
the CPU tasks take ``cpu``. Both pools must be provisioned on the cluster; they
are named, never created, here. Retries are set only where a transient outage is
plausible -- the teacher transport -- and only there, because a red ``verify``
board or a refused ``publish`` is *data*, deterministic across retries, and
republishing a verdict is noise, not recovery.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../ops/airflow/dags
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # repo root
_DATASET = os.path.join(_REPO, "dataset")
for _p in (_DATASET, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DAG_ID = "cosimo_v3_corpus"

#: The exact string the Makefile's ``v3-*`` targets run, so a task's command is
#: byte-for-byte the command that already "works under make" (§13: schedulable).
CLI_PREFIX = "uv run --group corpus python -m dataset.pipelines.v3.cli"

#: The record types ``render`` recognises (``cli.py`` validates ``--types`` against
#: exactly this set). One mapped task each, each owning its own ``sft`` shard.
RENDER_TYPES = (
    "analysis",
    "memo",
    "grounded",
    "critique",
    "abstention",
    "agentic",
    "exam",
    "implementation",
)

TEACHER_POOL = "teacher"
CPU_POOL = "cpu"

_DOC_MD = """### Cosimo v3 corpus ETL

`inventory >> packs(work_type)... >> render(record_type)... >> verify >> prefer >> publish`

Each task is one `dataset.pipelines.v3.cli` command; exit codes are the gate.
`packs` maps over work type (disjoint `fact_packs/<work_type>.jsonl`); `render`
maps over record type (disjoint `sft/<record_type>.jsonl`). Teacher-bound tasks
run in the `teacher` pool; the rest in `cpu`. No completion bytes travel in XCom.
"""


def _publish_dry_run() -> bool:
    """The publish gate's mode. Dry-run unless an operator flips it off explicitly.

    ``publish --dry-run`` still runs the whole board and still refuses on any red
    axis -- it just prints the dataset card instead of writing it -- so the safe
    default for an optional, cluster-side scheduler is to certify without committing.
    """
    return os.environ.get("COSIMO_V3_PUBLISH_DRY_RUN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )


def _teacher_retries() -> int:
    """How many times a teacher-bound task may re-run on a transient transport failure.

    The render/prefer stages are idempotent (``write.append_unique`` skips ids already
    on disk), so a retry resumes rather than re-does -- but the number stays small: the
    stage's own repair loop already spent its strikes on the *content*, and these are for
    a socket that dropped, not a brief the model cannot satisfy.
    """
    try:
        return max(0, int(os.environ.get("COSIMO_V3_TEACHER_RETRIES", "2")))
    except ValueError:
        return 2


def _plan_work_types() -> list[str]:
    """The work types the ``packs`` wave maps over -- derived from the taxonomy, never hardcoded.

    ``inventory.load_plan`` reads the same ``work_types.yaml`` the ``inventory`` task
    commits as ``plan.json``. The two stay consistent *by construction within a run*:
    the ``inventory`` task runs ahead of the packs wave and regenerates ``plan.json``
    from this very taxonomy, so the ``--work-type`` the mapped task passes is always a
    member of the plan ``cmd_packs`` validates it against. Deriving the map from the
    taxonomy (not a stale ``plan.json`` that may not exist at DAG-parse time, and not a
    hardcoded list) keeps "add a work type" a taxonomy edit rather than a DAG edit. The
    import is lazy so importing this module in a bare venv does not pull the corpus
    package until a DAG is actually being built.
    """
    from dataset.pipelines.v3 import config, inventory

    return sorted(inventory.load_plan(config.taxonomy_path()))


def v3_command(
    subcommand: str,
    *,
    work_type: str | None = None,
    types: str | None = None,
    out: str | None = None,
    dry_run: bool = False,
) -> str:
    """Assemble one task's shell string. No scoping flag means the Makefile's line verbatim."""
    argv = [subcommand]
    if work_type:
        argv += ["--work-type", work_type]
    if types:
        argv += ["--types", types]
    if out:
        argv += ["--out", out]
    if dry_run:
        argv += ["--dry-run"]
    return " ".join([CLI_PREFIX, *argv])


def inventory_command(*, out: str | None = None) -> str:
    return v3_command("inventory", out=out)


def packs_command(work_type: str | None = None, *, out: str | None = None) -> str:
    return v3_command("packs", work_type=work_type, out=out)


def render_command(record_type: str | None = None, *, out: str | None = None) -> str:
    return v3_command("render", types=record_type, out=out)


def verify_command(*, out: str | None = None) -> str:
    return v3_command("verify", out=out)


def prefer_command(*, out: str | None = None) -> str:
    return v3_command("prefer", out=out)


def publish_command(*, dry_run: bool = False, out: str | None = None) -> str:
    return v3_command("publish", out=out, dry_run=dry_run)


def _airflow_available() -> bool:
    try:
        import airflow  # noqa: F401  (probing availability only; the name is unused)

        return True
    except ModuleNotFoundError:
        return False


#: Whether the DAG can be built in this environment. The command builders above
#: are pure and always importable; only the DAG object needs Airflow.
HAS_AIRFLOW = _airflow_available()


def build_dag():
    """Construct the ``cosimo_v3_corpus`` DAG. Imports Airflow only when called.

    Called by Airflow's parser at the bottom of this module (guarded on
    :data:`HAS_AIRFLOW`) and by the topology test when Airflow is installed.
    """
    from airflow import DAG
    from airflow.operators.bash import BashOperator

    default_args = {
        "owner": "cosimo-corpus",
        "email": False,
        "email_on_failure": False,
        "depends_on_past": False,
        "retries": 0,
    }
    teacher_args = {
        # `retries` + `retry_delay` are the universally-valid BaseOperator params;
        # the exponential-backoff knobs go by different names across Airflow versions
        # and unknown kwargs are silently dropped ("Ignoring unknown arguments"), so a
        # plain bounded delay is the honest, non-silent choice for a transient teacher blip.
        "retries": _teacher_retries(),
        "retry_delay": 30,
    }

    with DAG(
        dag_id=DAG_ID,
        schedule="@once",
        start_date=datetime(2026, 9, 1),
        catchup=False,
        max_active_runs=1,
        default_args=default_args,
        tags=["cosimo", "v3", "corpus", "etl"],
        doc_md=_DOC_MD,
    ) as dag:
        inventory = BashOperator(
            task_id="inventory", bash_command=inventory_command(), pool=CPU_POOL
        )
        packs = [
            BashOperator(
                task_id=f"packs.{work_type}",
                bash_command=packs_command(work_type),
                pool=CPU_POOL,
            )
            for work_type in _plan_work_types()
        ]
        renders = [
            BashOperator(
                task_id=f"render.{record_type}",
                bash_command=render_command(record_type),
                pool=TEACHER_POOL,
                **teacher_args,
            )
            for record_type in RENDER_TYPES
        ]
        verify = BashOperator(
            task_id="verify", bash_command=verify_command(), pool=CPU_POOL
        )
        prefer = BashOperator(
            task_id="prefer",
            bash_command=prefer_command(),
            pool=TEACHER_POOL,
            **teacher_args,
        )
        publish = BashOperator(
            task_id="publish",
            bash_command=publish_command(dry_run=_publish_dry_run()),
            pool=CPU_POOL,
        )

        # Airflow's `>>` handles operator-to-single and single-to-list, but NOT
        # list-to-list -- both operands are plain lists, so `packs >> renders` would
        # raise TypeError at DAG *import*. The fan-in is therefore spelled one pack at
        # a time (operator >> list), and the two waves reconverge on verify (list >>
        # single uses the reflected hook).
        inventory >> packs
        for pack in packs:
            pack >> renders
        renders >> verify >> prefer >> publish

    return dag


if HAS_AIRFLOW:
    dag = build_dag()
