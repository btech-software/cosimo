"""The fact-computer registry: one pure-Python module per work type.

`work_types.yaml` is the plan; this mapping is the machinery. Verification
re-imports through :data:`COMPUTERS` and recomputes a pack from
``(work_type, family, variant)`` -- so a computer is load-bearing ground truth
twice over, once at generation and once at the publish gate, and it may never
depend on anything a shard contains.

First milestone is five computers (spec §5.3): valuation (FCFF, multiples),
market risk (VaR/ES), attribution, execution TCA. Fifteen is the first
publish bar; this registry grows by addition, never by edit of a released
module's arithmetic.
"""

from __future__ import annotations

from . import (
    execution_tca,
    portfolio_attribution,
    risk_var,
    valuation_fcff,
    valuation_multiples,
)
from .base import FactPack, PackError, assemble_numbers

COMPUTERS: dict[str, object] = {
    valuation_fcff.WORK_TYPE: valuation_fcff.compute,
    valuation_multiples.WORK_TYPE: valuation_multiples.compute,
    risk_var.WORK_TYPE: risk_var.compute,
    portfolio_attribution.WORK_TYPE: portfolio_attribution.compute,
    execution_tca.WORK_TYPE: execution_tca.compute,
}

FAMILIES: dict[str, tuple[str, ...]] = {
    valuation_fcff.WORK_TYPE: valuation_fcff.FAMILIES,
    valuation_multiples.WORK_TYPE: valuation_multiples.FAMILIES,
    risk_var.WORK_TYPE: risk_var.FAMILIES,
    portfolio_attribution.WORK_TYPE: portfolio_attribution.FAMILIES,
    execution_tca.WORK_TYPE: execution_tca.FAMILIES,
}


def compute_pack(work_type: str, family: str, variant: int) -> FactPack:
    """Build one scenario pack deterministically.

    Raises :class:`PackError` for an unregistered work type or a drawn
    parameter set that describes no coherent scenario -- the inventory treats
    that as *skip this variant*, not as a dead letter (spec §5.3).
    """
    computer = COMPUTERS.get(work_type)
    if computer is None:
        raise PackError(
            f"no fact computer registered for work type {work_type!r} "
            f"(registered: {sorted(COMPUTERS)})"
        )
    return computer(family, variant)


__all__ = [
    "COMPUTERS",
    "FAMILIES",
    "FactPack",
    "PackError",
    "assemble_numbers",
    "compute_pack",
]
