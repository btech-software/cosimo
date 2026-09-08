"""The wrong models, registered (analysis spec §5.10; arch spec §6 axes).

Every distractor in the corpus is produced here. Each derivation is a named,
pure deformation of the pack's own instrument -- a parameter read at the wrong
dial -- never ``correct + 7.0``: v1's ``_dedup_distractors`` band-aid (add
``7.0 + i`` when two options collided) manufactured distractors no examiner
would ever have written, and §5.10 names it as the disease. A derivation that
cannot produce a distinct number on a drawn scenario returns ``None`` and the
distractor is *dropped*, which is the spec's own instruction.

The instruments mirror the computers in :mod:`pipelines.v3.packs`: same
formulas, same order of operations, read here with one dial deliberately
mis-set. That is the whole pedagogy -- the wrong answer must be reachable by
a wrong *method*, because the corpus is teaching the method, not the number.

Pure functions of ``(inputs, computed)``; ``random`` never appears; ``None``
is the only failure value.
"""

from __future__ import annotations

import math


def _num(value) -> float | None:
    """A float from a pack scalar, or ``None`` -- the derivations stay total."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_dcf(inputs: dict) -> tuple | None:
    """The dcf scenario read from the pack's inputs; ``None`` if unreadable."""
    vals = [
        _num(inputs.get("revenue_m")),
        _num(inputs.get("ebit_margin")),
        _num(inputs.get("cash_tax_rate")),
        _num(inputs.get("capex_pct_revenue")),
        _num(inputs.get("nwc_pct_revenue_build")),
        _num(inputs.get("wacc")),
        _num(inputs.get("terminal_growth")),
        _num(inputs.get("explicit_years")),
    ]
    if any(v is None for v in vals):
        return None
    rev, ebit, tax, capex, nwc, wacc, g, n = vals
    if rev <= 0 or n < 1 or wacc <= g:
        return None
    return rev, ebit, tax, capex, nwc, wacc, g, int(n)


def _strip_pvfvp(
    inputs: dict,
    *,
    tax: float | None = None,
    nwc_on_level: bool = False,
    discount: bool = True,
    tv_extra_step: bool = True,
) -> float | None:
    """The FCFF enterprise value with the dials readable -- one may be wrong.

    Textbook path (all defaults) reproduces the computer's own identity:

        FCFF_t = Rev_t * ebit * (1 - tax) - capex% * Rev_t - nwc% * build_t
        EV     = sum PV(FCFF_1..n) + PV(TV_n)
        TV_n   = FCFF_n (1 + g) / (WACC - g)

    Each keyword is one named pitfall's dial: ``tax`` over-ridden to zero is
    the forgotten tax shield; ``nwc_on_level`` charges working capital on the
    revenue *level* instead of its build (v2's exact sign bug, per the fcff
    computer's own docstring); ``discount=False`` leaves the strip in nominal
    terms; ``tv_extra_step=False`` omits the ``(1+g)`` on the terminal value.
    """
    read = _read_dcf(inputs)
    if read is None:
        return None
    rev, ebit, tax_read, capex, nwc, wacc, g, n = read
    t = tax_read if tax is None else tax
    if t is None or not 0.0 <= t < 1.0:
        return None
    run = rev
    pv = 0.0
    fcff_last = 0.0
    for year in range(1, n + 1):
        nxt = run * (1.0 + g)
        build = nxt if nwc_on_level else nxt - run
        fcff = nxt * ebit * (1.0 - t) - capex * nxt - nwc * build
        factor = 1.0 if not discount else 1.0 / (1.0 + wacc) ** year
        pv += fcff * factor
        fcff_last = fcff
        run = nxt
    tv = fcff_last * ((1.0 + g) if tv_extra_step else 1.0) / (wacc - g)
    pv += tv if not discount else tv / (1.0 + wacc) ** n
    return pv


def _pit_dcf_no_tax_shield(inputs: dict, computed: dict) -> float | None:
    """Tax forgotten: the cash tax rate read as zero."""
    return _strip_pvfvp(inputs, tax=0.0)


def _pit_dcf_flat_working_capital(inputs: dict, computed: dict) -> float | None:
    """Working capital charged on the revenue level, not on its build."""
    return _strip_pvfvp(inputs, nwc_on_level=True)


def _pit_dcf_no_discounting(inputs: dict, computed: dict) -> float | None:
    """The time value of money ignored: the strip left undiscounted."""
    return _strip_pvfvp(inputs, discount=False)


def _pit_dcf_short_terminal(inputs: dict, computed: dict) -> float | None:
    """The terminal value one growth step short: no (1 + g) on the TV."""
    return _strip_pvfvp(inputs, tv_extra_step=False)


def _read_peers(inputs: dict) -> tuple | None:
    """The peer set split into (EV/EBITDA, P/E) columns; ``None`` if ragged."""
    peers = inputs.get("peers")
    if not isinstance(peers, list) or not peers:
        return None
    evs, pes = [], []
    for peer in peers:
        if not isinstance(peer, dict):
            return None
        ev, pe = _num(peer.get("ev_ebitda")), _num(peer.get("p_e"))
        if ev is None or pe is None:
            return None
        evs.append(ev)
        pes.append(pe)
    return evs, pes


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _per_share_from(inputs: dict, enterprise: float | None) -> float | None:
    """Bridge an enterprise value to a per-share read; ``None`` if unreadable."""
    if enterprise is None:
        return None
    shares = _num(inputs.get("shares_m"))
    net_debt = _num(inputs.get("net_debt_m"))
    if shares is None or shares <= 0 or net_debt is None:
        return None
    return (enterprise - net_debt) / shares


def _pit_multiples_mean_not_median(inputs: dict, computed: dict) -> float | None:
    """The mean taken for the median: the outlier peer sets the read."""
    peers = _read_peers(inputs)
    ebitda = _num(inputs.get("ebitda_m"))
    if peers is None or ebitda is None:
        return None
    mean_ev = sum(peers[0]) / len(peers[0])
    return _per_share_from(inputs, mean_ev * ebitda)


def _pit_multiples_bridge_wrong_way(inputs: dict, computed: dict) -> float | None:
    """The net-debt bridge crossed adding: the classic direction slip."""
    peers = _read_peers(inputs)
    ebitda = _num(inputs.get("ebitda_m"))
    net_debt = _num(inputs.get("net_debt_m"))
    shares = _num(inputs.get("shares_m"))
    if peers is None or ebitda is None or net_debt is None or not shares:
        return None
    return (_median(peers[0]) * ebitda + net_debt) / shares


def _pit_multiples_pe_on_enterprise(inputs: dict, computed: dict) -> float | None:
    """The P/E median applied to EBITDA: the income base read as the whole."""
    peers = _read_peers(inputs)
    ebitda = _num(inputs.get("ebitda_m"))
    shares = _num(inputs.get("shares_m"))
    if peers is None or ebitda is None or not shares:
        return None
    return _median(peers[1]) * ebitda / shares


def _pit_multiples_ev_as_equity(inputs: dict, computed: dict) -> float | None:
    """The enterprise value read out as if it were the equity (no bridge)."""
    peers = _read_peers(inputs)
    ebitda = _num(inputs.get("ebitda_m"))
    shares = _num(inputs.get("shares_m"))
    if peers is None or ebitda is None or not shares:
        return None
    return _median(peers[0]) * ebitda / shares


def _read_var(inputs: dict) -> tuple | None:
    vals = [
        _num(inputs.get("book_value_m")),
        _num(inputs.get("mu_daily")),
        _num(inputs.get("sigma_daily")),
        _num(inputs.get("horizon_days")),
        _num(inputs.get("z95")),
    ]
    if any(v is None for v in vals):
        return None
    value, mu, sigma, horizon, z95 = vals
    if value <= 0 or sigma <= 0 or horizon < 1:
        return None
    return value, mu, sigma, int(horizon), z95


def _pit_var_linear_scaling(inputs: dict, computed: dict) -> float | None:
    """The horizon scaled linearly instead of by the square root of time."""
    read = _read_var(inputs)
    if read is None:
        return None
    value, mu, sigma, horizon, z95 = read
    return value * (z95 * sigma - mu) * horizon


def _pit_var_drift_ignored(inputs: dict, computed: dict) -> float | None:
    """The drift dropped from the quantile: mu read as zero."""
    read = _read_var(inputs)
    if read is None:
        return None
    value, _, sigma, horizon, z95 = read
    return value * z95 * sigma * math.sqrt(horizon)


def _pit_var_one_day_as_horizon(inputs: dict, computed: dict) -> float | None:
    """The one-day figure sold as the horizon figure: no scaling at all."""
    read = _read_var(inputs)
    if read is None:
        return None
    value, mu, sigma, _, z95 = read
    return value * (z95 * sigma - mu)


def _pit_var_wrong_tail_constant(inputs: dict, computed: dict) -> float | None:
    """The 99% tail constant carried into the 95% read."""
    read = _read_var(inputs)
    z99 = _num(inputs.get("z99"))
    if read is None or z99 is None:
        return None
    value, mu, sigma, horizon, _ = read
    return value * (z99 * sigma - mu) * math.sqrt(horizon)


def _read_attribution(inputs: dict) -> tuple | None:
    """The sector table read and the benchmark sums + Carino bridge derived."""
    rows = inputs.get("sector_rows")
    if not isinstance(rows, dict) or not rows:
        return None
    cells = []
    for row in rows.values():
        if not isinstance(row, dict):
            return None
        vals = [
            _num(row.get("wb")),
            _num(row.get("wp")),
            _num(row.get("rb")),
            _num(row.get("rp")),
        ]
        if any(v is None for v in vals):
            return None
        cells.append(tuple(vals))
    rb = sum(wb * r for wb, _, r, _ in cells)
    rp = sum(wp * p for _, wp, _, p in cells)
    if rb == 0.0 or rp == rb:
        return None
    k = math.log((1.0 + rp) / (1.0 + rb)) / ((rp - rb) / rb)
    return cells, rb, rp, k


def _pit_attribution_interaction_dropped(inputs: dict, computed: dict) -> float | None:
    """The interaction effect dropped from the sum (the 'droppable' myth)."""
    read = _read_attribution(inputs)
    if read is None:
        return None
    cells, rb, _, k = read
    return (
        sum((wp - wb) * (r - rb) * k + wp * (p - r) * k for wb, wp, r, p in cells) * 1e4
    )


def _pit_attribution_brinson_plain(inputs: dict, computed: dict) -> float | None:
    """Brinson before Carino: the effects without the scaling bridge."""
    read = _read_attribution(inputs)
    if read is None:
        return None
    cells, rb, _, _ = read
    return (
        sum(
            (wp - wb) * (r - rb) + wp * (p - r) + (wp - wb) * (p - r)
            for wb, wp, r, p in cells
        )
        * 1e4
    )


def _pit_attribution_selection_benchmark_weights(
    inputs: dict, computed: dict
) -> float | None:
    """Selection computed at benchmark weights (the Brinson-Fachler dial)."""
    read = _read_attribution(inputs)
    if read is None:
        return None
    cells, _, _, k = read
    return sum(wb * (p - r) * k for wb, _, r, p in cells) * 1e4


def _read_tca(inputs: dict) -> tuple | None:
    vals = [
        _num(inputs.get("shares")),
        _num(inputs.get("adv_shares")),
        _num(inputs.get("quoted_spread_bps")),
        _num(inputs.get("sigma_daily")),
    ]
    if any(v is None for v in vals):
        return None
    shares, adv, spread, sigma = vals
    if shares <= 0 or adv <= 0 or sigma <= 0:
        return None
    return shares, adv, spread, sigma


def _pit_tca_half_spread_forgotten(inputs: dict, computed: dict) -> float | None:
    """The half-spread forgotten: only the impact is paid."""
    read = _read_tca(inputs)
    if read is None:
        return None
    shares, adv, _, sigma = read
    return sigma * math.sqrt(shares / adv) * 1e4


def _pit_tca_linear_impact(inputs: dict, computed: dict) -> float | None:
    """Impact linear in participation, not square-root."""
    read = _read_tca(inputs)
    if read is None:
        return None
    shares, adv, spread, sigma = read
    return spread / 2.0 + sigma * (shares / adv) * 1e4


def _pit_tca_full_spread(inputs: dict, computed: dict) -> float | None:
    """The whole quoted spread paid, not the half: crossed at the wrong side."""
    read = _read_tca(inputs)
    if read is None:
        return None
    shares, adv, spread, sigma = read
    return spread + sigma * math.sqrt(shares / adv) * 1e4


def _pit_tca_printable_mid(inputs: dict, computed: dict) -> float | None:
    """The printable mid read as achievable: the spread itself for free."""
    read = _read_tca(inputs)
    if read is None:
        return None
    _, _, spread, _ = read
    return spread / 2.0


#: The registry: named pitfalls per work type, in fixed order. The order is
#: part of the identity -- the sampler draws from this sequence under the
#: item's own seed, so a re-run of the same plan yields the same items.
PITFALL_DERIVATIONS: dict[str, tuple[tuple[str, "object"], ...]] = {
    "valuation.equity.dcf": (
        ("no cash tax shield", _pit_dcf_no_tax_shield),
        ("working capital on the level, not the build", _pit_dcf_flat_working_capital),
        ("the strip left undiscounted", _pit_dcf_no_discounting),
        ("terminal value one growth step short", _pit_dcf_short_terminal),
    ),
    "valuation.equity.multiples": (
        ("the mean taken for the median", _pit_multiples_mean_not_median),
        ("the net-debt bridge crossed the wrong way", _pit_multiples_bridge_wrong_way),
        ("the P/E quoted against the enterprise", _pit_multiples_pe_on_enterprise),
        ("the enterprise value read as the equity", _pit_multiples_ev_as_equity),
    ),
    "risk.market.var_es": (
        ("the horizon scaled linearly", _pit_var_linear_scaling),
        ("the drift dropped from the quantile", _pit_var_drift_ignored),
        ("the one-day figure sold as the horizon", _pit_var_one_day_as_horizon),
        ("the 99% tail constant on the 95% read", _pit_var_wrong_tail_constant),
    ),
    "portfolio.attribution.brinson_carino": (
        ("the interaction effect dropped", _pit_attribution_interaction_dropped),
        ("Brinson before Carino", _pit_attribution_brinson_plain),
        (
            "selection at benchmark weights",
            _pit_attribution_selection_benchmark_weights,
        ),
    ),
    "execution.tca.arrival": (
        ("the half-spread forgotten", _pit_tca_half_spread_forgotten),
        ("impact linear in participation", _pit_tca_linear_impact),
        ("the whole spread paid", _pit_tca_full_spread),
        ("the printable mid read as achievable", _pit_tca_printable_mid),
    ),
}

__all__ = ["PITFALL_DERIVATIONS"]
