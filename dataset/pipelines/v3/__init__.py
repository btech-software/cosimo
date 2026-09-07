"""Cosimo v3 generation backend: fact packs first, teacher prose second.

New in v3, added alongside -- never rewritten over -- the frozen v1/v2
templates (`pipelines.generate`, `pipelines.templates`). The unit of
generation is a deterministic, code-computed `FactPack`; a language-model
teacher is only ever allowed to *phrase* what the pack already established,
and every row is verified by recomputing the pack from its seed. See
``docs/prompts/rev_03/COSIMO_V3_ARCHITECTURE.md``.

Import rules (spec §3): this package may import `pipelines.core` and the
shared contracts in `cosimo.tools`; it must never import `jobs`.
"""
