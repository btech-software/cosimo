"""The render stages: plan + fact packs + teacher in, shards out (spec §5.5).

The pack stage (PR1) proved a fact; a render stage makes a *row* out of it,
and the only entity in the corpus that can be wrong about its own output is
the language model -- so every module here is built around the same three
constraints:

* the teacher is an argument (``teacher_from_env`` / an injected transport),
  never an import-time side effect -- the offline suite and the cluster run
  the same code path;
* nothing reaches ``sft/`` without passing the prose gate
  (:mod:`pipelines.v3.verification.prose`), and the gate's verdict travels
  with the row in its ``verification`` block, so an audit never has to
  re-derive why a row is trusted;
* every stage is replayable over a half-finished corpus and converges
  (resume by ids, never by counts -- the same rule ``stage.py`` enforces for
  packs).

``prose.py`` is PR2's. ``agentic.py`` (PR3) and the exam/implementation
renderers (PR4) join it here; they share the repair policy, not a code path
into the teacher, because an agentic turn loop is a different animal from a
one-shot completion.
"""

from .prose import (
    render_prose_row,
    row_id,
    row_id_from_coords,
    run_render_stage,
    select_prose_jobs,
)

__all__ = [
    "render_prose_row",
    "row_id",
    "row_id_from_coords",
    "run_render_stage",
    "select_prose_jobs",
]
