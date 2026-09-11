"""The register table: ``work_types.yaml`` decides the voice, not the renderer.

Before the amendment a register was a literal inside each fact computer -- two
of them a bare string, two of them ``rng.choice(("desk_chat", "ic_memo"))``.
That made the register a *label* the row happened to carry, and a label is
exactly what :mod:`pipelines.v3.verification.register` cannot gate against: if
the only statement of "this family speaks desk chat" lives in the computer that
wrote the row, then a row and its contract can never disagree, and a gate that
can never fire is not a gate.

So the table moves to the plan, beside the families it describes, and this
module is the one reader of it. Three properties make that safe for a *pure*
fact computer to depend on:

* the plan file is source, committed, and already the authority
  ``inventory.load_plan`` validates every other pack decision against;
* it is read once and memoised on the resolved path, so recomputing a pack in
  the verify board costs no I/O after the first;
* the *draw* is unchanged where a family names one register (no rng call at
  all) and identical where it names several (``rng.choice`` over the list in
  file order) -- which is what keeps every pack seed, and therefore every row
  id, byte-stable across this change.
"""

from __future__ import annotations

import yaml

from .. import config
from .base import PackError

#: ``{path: {work_type: {family: (register, ...)}}}``. Keyed by resolved path
#: because the tests point ``taxonomy_path`` at fixture plans, and a cache keyed
#: on nothing would serve one test's table to the next.
_CACHE: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}


def _load(path: str) -> dict[str, dict[str, tuple[str, ...]]]:
    if path in _CACHE:
        return _CACHE[path]
    try:
        with open(path, encoding="utf8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        # A plan that cannot be read is the *inventory's* finding to report, in
        # its own words, at load time. Refusing here would turn every pack
        # recompute in a fixture-driven test into a yaml error about a file the
        # test never claimed to use.
        raw = {}
    table: dict[str, dict[str, tuple[str, ...]]] = {}
    for work_type, spec in raw.items() if isinstance(raw, dict) else ():
        if not isinstance(spec, dict):
            continue
        entries = spec.get("registers")
        if not isinstance(entries, dict):
            continue
        table[work_type] = {
            str(family): tuple(str(r) for r in (value or ()))
            for family, value in entries.items()
            if isinstance(value, (list, tuple))
        }
    _CACHE[path] = table
    return table


def registers_for(work_type: str, family: str) -> tuple[str, ...]:
    """The registers *family* may be written in, in file order.

    Raises :class:`PackError` for a family the plan does not describe: a
    scenario whose voice nobody declared is not a scenario the corpus knows how
    to gate, and the packs stage records that as the guard's own verdict rather
    than shipping a row under a guessed register.
    """
    table = _load(config.taxonomy_path())
    declared = (table.get(work_type) or {}).get(family)
    if not declared:
        raise PackError(
            f"{work_type}/{family}: the plan declares no register for this "
            "family; add it under the work type's `registers:` table "
            "(dataset/taxonomy/work_types.yaml)"
        )
    unknown = [r for r in declared if r not in config.VALID_REGISTERS]
    if unknown:
        raise PackError(
            f"{work_type}/{family}: plan names unknown register(s) "
            f"{', '.join(map(repr, unknown))} "
            f"(known: {', '.join(config.VALID_REGISTERS)})"
        )
    return declared


def pick_register(work_type: str, family: str, rng) -> str:
    """One register for this pack, drawn deterministically from the plan.

    A single-register family consumes **no** draw. That is not a micro-
    optimisation: every downstream value in a computer comes off the same
    seeded ``Random``, so spending a draw here where none was spent before
    would shift every subsequent number in the pack -- and with it every
    ``allowed_numbers`` entry the committed fixtures were captured against.
    """
    declared = registers_for(work_type, family)
    if len(declared) == 1:
        return declared[0]
    return rng.choice(list(declared))


def reset_cache() -> None:
    """Forget the memoised tables; the tests swap plan files between cases."""
    _CACHE.clear()


__all__ = ["pick_register", "registers_for", "reset_cache"]
