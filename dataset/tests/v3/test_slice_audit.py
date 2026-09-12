"""``slice_audit`` is the gate between a read slice and a paid run (§D, §F).

Two of its questions are new, and both are questions the sixteen-axis board
cannot ask of a single row: does any row in this slice contradict its own
pack, and do ``analysis`` and ``grounded`` on one scenario open on the same
sentence. The second is not a defect in either row; it is the evidence that
two record types are one brief wearing two names, which is what §B.2 split.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_DATASET, "tools"), _DATASET):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import slice_audit  # noqa: E402

from pipelines.v3 import row as rowlib, write  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.prose import row_id_from_coords  # noqa: E402

TCA = ("execution.tca.arrival", "large_cap_intraday", 0)


def _row(kind: str, answer: str, coords=TCA) -> dict:
    work_type, family, variant = coords
    pack = compute_pack(work_type, family, variant)
    return rowlib.student_row(
        row_id=row_id_from_coords(kind, work_type, family, variant),
        kind=kind,
        pack=pack.to_dict(),
        holdout=False,
        answer=answer,
        stamp={"pack_seed": f"{pack.seed:016x}"},
        teacher={"model": "test", "finish_reason": "stop", "usage": {}},
        render={"kind": kind, "attempts": 1},
        attempts=1,
        invented=[],
        missing=[],
        register_ok=True,
    )


def _slice(tmp_path, rows: list[dict]) -> str:
    out = str(tmp_path)
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["record_type"], []).append(row)
    for kind, kind_rows in by_kind.items():
        write.write_jsonl(write.path_for("sft", kind, out), kind_rows)
    # The ablation file is a separate question and is answered in the repo;
    # these tests are about the two checks above.
    return out


def _problems(out: str) -> list[str]:
    return [p for p in slice_audit.audit(out) if "ablation" not in p]


def test_a_contradicted_row_stops_the_slice(tmp_path):
    out = _slice(
        tmp_path,
        [
            _row(
                "analysis",
                "Cost is 28.16 bp of arrival; at 4.86% of ADV the schedule "
                "itself becomes the risk.",
            )
        ],
    )
    problems = _problems(out)
    assert any("contradict their own pack" in p for p in problems)
    assert any("schedule_risk_below_cap" in p for p in problems)


def test_a_clean_row_does_not(tmp_path):
    out = _slice(
        tmp_path,
        [
            _row(
                "analysis",
                "Shortfall is 28.16 bp against arrival, 26.66 bp of it impact. "
                "Participation is 4.86% of ADV, under the 10% cap, so the "
                "impact is the bill and the schedule is not the problem.",
            )
        ],
    )
    assert not any("contradict their own pack" in p for p in _problems(out))


def test_analysis_and_grounded_sharing_a_first_sentence_is_a_finding(tmp_path):
    """The live sample's own symptom: both rows opened '28.16 bp of arrival'."""
    opening = "28.16 bp of arrival, 26.66 bp impact, 1.5 bp half-spread. "
    out = _slice(
        tmp_path,
        [
            _row("analysis", opening + "Work it as a participation clip."),
            _row("grounded", opening + "Implied fill 295.7747."),
        ],
    )
    assert any("same sentence" in p for p in _problems(out))


def test_two_different_openings_are_not(tmp_path):
    out = _slice(
        tmp_path,
        [
            _row(
                "analysis",
                "28.16 bp of arrival, 26.66 bp impact. Work it as a clip.",
            ),
            _row(
                "grounded",
                "Implementation shortfall against arrival is 28.16 bp, with "
                "26.66 bp of impact and 1.5 bp of half-spread.",
            ),
        ],
    )
    assert not any("same sentence" in p for p in _problems(out))
