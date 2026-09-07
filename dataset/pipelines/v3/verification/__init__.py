"""Verification for the v3 corpus (spec §6).

The v2 ``verification/`` package keeps serving v1/v2 shards untouched; v3
grows its axes here because v3 verifies a different unit (a recomputed
``FactPack`` behind every row, plus a language model in the loop whose output
cannot be trusted to stay inside the pack). Axes land here, PR by PR:

* :mod:`~pipelines.v3.verification.invented_numbers` -- axis 4 of the spec
  list: a completion may not contain a number the pack did not authorise.
* ``prose.py`` / ``shape.py`` arrive with the renderers (PR2/PR3); the
  full-board runner is :mod:`~pipelines.v3.verify_v3`.
"""
