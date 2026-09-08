"""The PR3 gate (arch spec §10): 20 conversations, 4 with faults, 4 no-call,
every final number grounded -- plus the fixture's duty to regenerate byte
identical and the transcripts' duty to survive the serving parser.

Three authorities, three lenses:

* the **stage** runs through the real ``Teacher``/``FixtureTransport`` pair
  over the committed replay table -- the same code path the offline cluster
  walks, so what goes green here ships there;
* the **audit** lens re-derives every check from ``compute_pack`` alone --
  the row's own claims about its schedule, its results and its grounding are
  verified against the recomputed pack, never against itself;
* the **serving parser** lens renders each conversation through the shipped
  chat template (the harness's own ``JinjaTokenizer``) and round-trips every
  call through the Hermes wire helpers -- a trajectory the server cannot
  parse back is not a trajectory, it is prose about one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_agentic_fixture as harness  # noqa: E402
from cosimo.tools import wire  # noqa: E402
from pipelines.v3 import cli, config, inventory, stage, write  # noqa: E402
from pipelines.v3.oracle import runtime  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.agentic import run_agentic_stage, select_agentic_jobs  # noqa: E402
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verification.agentic import expected_tool_contents, gate_violations  # noqa: E402

COMMITTED = os.path.join(_HERE, "fixtures", harness.AGENTIC_FIXTURE_NAME)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_HARNESS_ROOT = os.path.join(REPO, "jobs", "fine-tune")


def _regenerated_bytes(payload: dict, tmp_path) -> bytes:
    path = os.path.join(str(tmp_path), "regen.json")
    with open(path, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    with open(path, "rb") as handle:
        return handle.read()


def _serving_harness():
    """The harness's template renderer, loaded under a name that cannot collide
    with this suite's own conftest."""
    if "cosimo_ft" not in sys.modules:
        sys.path.insert(0, _HARNESS_ROOT)
    from cosimo_ft import chat  # noqa: E402

    name = "v3_harness_conftest"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_HARNESS_ROOT, "tests", "conftest.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    with open(
        os.path.join(_HARNESS_ROOT, "configs", "chat_template.jinja"), encoding="utf8"
    ) as handle:
        template = handle.read()
    return chat, module.JinjaTokenizer(template)


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """The PR3 corpus: packs staged, the agentic slice rendered by replay.

    The lane is pinned here, not inherited: a fixture that rides on some
    earlier test's environment is an order dependency, and order dependencies
    are how suites rot -- the same pin the offline cluster's env carries.
    """
    payload = json.load(open(COMMITTED, encoding="utf8"))
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, payload["model"])
    monkeypatch.delenv(config.LIVE_ENV, raising=False)
    out = str(tmp_path / "agentic_gate")
    jobs = inventory.expand_jobs(inventory.load_plan(config.taxonomy_path()))
    head = select_agentic_jobs(jobs, limit=payload["meta"]["ranks_walked"])
    from pipelines.v3.packs import PackError

    by_work: dict[str, list[dict]] = {}
    for job in head:
        try:
            by_work.setdefault(job.work_type, []).append(
                stage.pack_record(job.work_type, job.family, job.variant)
            )
        except PackError:
            continue
    for work_type, lines in by_work.items():
        write.write_jsonl(write.path_for("fact_packs", work_type, out), lines)
    teacher = Teacher(FixtureTransport(path=COMMITTED))
    report = run_agentic_stage(
        out, jobs, teacher, limit=payload["meta"]["ranks_walked"]
    )
    rows = write.read_jsonl(write.path_for("sft", "agentic", out))
    return payload, out, report, rows


# ------------------------------------------------------------- the fixture's duty


def test_the_committed_fixture_is_byte_identical_to_its_harness(tmp_path):
    payload = harness.build_fixture(
        harness.DEFAULT_LIMIT, harness.DEFAULT_FAULTED, harness.DEFAULT_NO_CALL
    )
    with open(COMMITTED, "rb") as handle:
        committed = handle.read()
    assert _regenerated_bytes(payload, tmp_path) == committed, (
        "the committed agentic fixture drifted from its harness; regenerate with "
        ".venv/bin/python dataset/tests/v3/fixtures/make_agentic_fixture.py and "
        "read the diff before accepting it"
    )


# ------------------------------------------------------------- the gate itself


def test_twenty_conversations_four_faulted_four_no_call(gate):
    payload, _out, report, rows = gate
    meta = payload["meta"]
    assert report["rendered"] == meta["limit"] == 20
    assert report["dead_lettered"] == 0 and report["missing_packs"] == []
    assert report["no_call"] == meta["no_call"] == 4
    assert report["faulted"] == meta["faulted"] == 4
    assert len({row["id"] for row in rows}) == 20, (
        "ids are the corpus's word for a coordinate"
    )
    drawn = sorted(
        {
            row["verification"]["render"]["fault"]
            for row in rows
            if row["verification"]["render"]["fault"]
        }
    )
    assert len(drawn) == 4, f"the faulted slice drew {drawn}, not four distinct faults"
    modes = [row["verification"]["render"]["mode"] for row in rows]
    assert modes.count("clean") == 12


def test_every_row_survives_the_audit_its_own_claims_recomputed(gate):
    """The row's claims -- schedule, results, grounding -- are audited against
    ``compute_pack`` alone; a row that lies about any of them fails here exactly
    as ``verify_v3`` will fail it at the board."""
    _payload, _out, _report, rows = gate
    for row in rows:
        family = row["scenario_id"][len(row["work_type"]) + 1 :]
        pack = compute_pack(row["work_type"], family, row["variant"])
        pack_dict = pack.to_dict()
        render_field = row["verification"]["render"]
        violations = gate_violations(
            pack_dict,
            row["messages"],
            mode=render_field["mode"],
            fault=render_field["fault"],
        )
        assert violations == [], f"{row['id']}: {violations}"


def test_tool_results_replay_against_the_recomputed_pack_byte_for_byte(gate):
    _payload, _out, _report, rows = gate
    for row in rows:
        family = row["scenario_id"][len(row["work_type"]) + 1 :]
        pack_dict = compute_pack(row["work_type"], family, row["variant"]).to_dict()
        calls = [
            {
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
            for message in row["messages"]
            for call in message.get("tool_calls") or ()
        ]
        stored = [
            message["content"]
            for message in row["messages"]
            if message.get("role") == "tool"
        ]
        expected = expected_tool_contents(
            pack_dict,
            calls,
            mode=row["verification"]["render"]["mode"],
            fault=row["verification"]["render"]["fault"],
        )
        assert stored == expected, (
            f"{row['id']}: the stored blocks are not the blocks the oracle "
            "would have returned for this schedule"
        )


def test_no_call_rows_called_nothing_and_clean_rows_looped(gate):
    _payload, _out, _report, rows = gate
    for row in rows:
        render_field = row["verification"]["render"]
        calls = sum(
            1
            for message in row["messages"]
            for _call in message.get("tool_calls") or ()
        )
        if render_field["mode"] == "no_call":
            assert calls == 0 and render_field["tool_calls"] == 0
            assert row["tool_names"] == [] and row["tool_schemas"] == []
        else:
            assert calls >= 2, "a looping conversation that never loops is a misfit"
            assert calls <= config.AGENTIC_MAX_TOOL_CALLS


# ------------------------------------------------------------- the CLI's board


def test_the_dag_command_lines_run_the_agentic_cell_end_to_end(
    tmp_path, monkeypatch, capsys
):
    payload = json.load(open(COMMITTED, encoding="utf8"))
    out = str(tmp_path / "corpus")
    plan = os.path.join(out, "plan.json")
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, payload["model"])
    monkeypatch.setenv(config.TEACHER_FIXTURE_ENV, COMMITTED)
    monkeypatch.delenv(config.LIVE_ENV, raising=False)
    assert cli.main(["inventory", "--out", plan]) == cli.EXIT_OK
    assert cli.main(["packs", "--plan", plan, "--out", out]) == cli.EXIT_OK
    capsys.readouterr()
    assert (
        cli.main(
            [
                "render",
                "--types",
                "agentic",
                "--limit",
                str(payload["meta"]["ranks_walked"]),
                "--out",
                out,
                "--plan",
                plan,
            ]
        )
        == cli.EXIT_OK
    )
    board = capsys.readouterr()
    assert re.search(r"agentic\s+rendered\s+20", board.out), board.out
    assert re.search(r"no_call\s+4", board.out), board.out
    assert re.search(r"faulted\s+4", board.out), board.out
    assert cli.main(["verify", "--out", out]) == cli.EXIT_OK
    board = capsys.readouterr()
    assert "tool schemas + roles" in board.out and "tool-result replay" in board.out
    assert "VERIFY FAIL" not in board.err


# ------------------------------------------------------------- the serving parser


def test_conversations_survive_the_serving_template_and_the_wire_parser(gate):
    """Axis 9's claim, audited from the server's side: the shipped template
    renders every transcript, and the Hermes round-trip returns every call
    byte-honest -- a training target the runtime cannot parse back is a
    format the model will learn to speak and the server will never hear."""
    chat, tokenizer = _serving_harness()
    _payload, _out, _report, rows = gate
    for row in rows:
        example = chat.render_tool_example(
            tokenizer, row["messages"], row["tool_schemas"] or None
        )
        assert example["text"].startswith(example["prompt"])
        assert example["completion"]
        for message in row["messages"]:
            calls = message.get("tool_calls") or ()
            if calls:
                local = [
                    {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                    for call in calls
                ]
                rendered = wire.render_tool_calls(local)
                assert wire.TOOL_CALL_OPEN in rendered
                assert wire.parse_tool_calls(rendered) == local, (
                    f"{row['id']}: the server parses a different call than the "
                    "corpus wrote"
                )
            if message.get("role") == "tool":
                rendered = wire.render_tool_result(
                    str(message.get("name") or ""), message["content"]
                )
                # The template, not this helper, writes the <|im_start|>
                # envelope (the chat template owns markers; the wire owns the
                # payload between them) -- so the honest round-trip here is a
                # parse-back, not a substring: a corpus whose results only
                # *look* present when grepped is a corpus the server cannot
                # serve.
                assert json.loads(rendered) == {
                    "name": str(message.get("name") or ""),
                    "content": message["content"],
                }


def test_the_final_turn_carries_no_markup_the_server_would_misread(gate):
    """The answer is prose; if it still carries a Hermes block the serving
    parser would lift it as an action, and a desk answer that acts is not an
    answer."""
    _payload, _out, _report, rows = gate
    for row in rows:
        assert wire.parse_tool_calls(row["answer"]) == []
        assert wire.TOOL_CALL_OPEN not in row["answer"]


def test_advertised_schemas_stay_inside_the_registry_the_server_registers(gate):
    _payload, _out, _report, rows = gate
    for row in rows:
        advertised = [schema["function"]["name"] for schema in row["tool_schemas"]]
        assert advertised == sorted(set(advertised))
        assert set(advertised) <= set(runtime.SCHEMA_NAMES)
        assert set(row["tool_names"]) == set(advertised)
