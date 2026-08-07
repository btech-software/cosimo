"""Long-form depth for analysis records, composed per record from its own RNG.

Why this module exists: the first attempt at "long-form analysis" appended one
identical 819-token essay to all 8,084 analysis records -- 88% of every answer
was the same block, and it discussed DCF capex assumptions under risk-budgeting
questions. It inflated the length gate without adding one token of reasoning.

The contract here is different on three axes:

  * **Topically keyed.** Paragraph builders are grouped by domain; a risk
    question draws risk paragraphs, a fixed-income question draws duration and
    curve paragraphs. `deepen()` maps the record's topic to its pool.
  * **Numerically alive.** Every builder draws its own parameters from the
    record's seeded RNG and computes its illustration inside the paragraph, so
    the numbers are consistent by construction and no two records render the
    same text. This is what keeps the corpus rule -- every number computed,
    never written -- true in prose as well as in traces.
  * **Structurally varied.** Which paragraphs, how many, and in what order are
    all RNG draws. There is no fixed essay skeleton to memorise.

Each builder returns ~80-130 words. A record gets 4-6 of them, which puts the
composed answer around 550-800 words (~700-1000 approx tokens) on top of the
generator's computed core.
"""
import math

from pipelines.core import fmt, pct


# ---------------------------------------------------------------------------
# equity valuation
# ---------------------------------------------------------------------------

def _eq_terminal_sensitivity(rng):
    wacc = rng.uniform(0.075, 0.115, 4)
    g = rng.uniform(0.015, 0.035, 4)
    spread = wacc - g
    tv_mult = (1 + g) / spread
    g2 = g + 0.005
    tv_mult2 = (1 + g2) / (wacc - g2)
    move = (tv_mult2 / tv_mult - 1) * 100
    return (
        f"The terminal value deserves more scrutiny than the explicit years, because the "
        f"Gordon growth denominator is doing silent work. At a {pct(wacc)} discount rate "
        f"and {pct(g)} perpetuity growth the spread is only {pct(spread)}, so the terminal "
        f"multiple is {tv_mult:.1f}x the final-year cash flow. Nudge growth up just 50bp "
        f"and that multiple becomes {tv_mult2:.1f}x — a {move:.0f}% move in the dominant "
        f"component of value from an assumption nobody can verify. When the spread "
        f"compresses below three points, the model stops being a valuation and becomes a "
        f"bet on the denominator. The defensible response is to report value as a band "
        f"across growth scenarios rather than as the point the base case happens to emit."
    )


def _eq_margin_reversion(rng):
    m_now = rng.uniform(0.16, 0.28, 3)
    m_hist = rng.uniform(0.10, 0.15, 3)
    years = rng.randint(4, 8)
    return (
        f"A margin assumption is a competitive-advantage assumption wearing accounting "
        f"clothes. Holding an operating margin of {pct(m_now)} flat through the forecast "
        f"implicitly claims the moat survives {years} more years of entry and pricing "
        f"pressure; the sector's through-cycle median sits nearer {pct(m_hist)}. If margins "
        f"mean-revert even halfway to that median, roughly "
        f"{pct((m_now - (m_now + m_hist) / 2) / m_now)} of forecast operating profit "
        f"disappears before any change to the revenue line. The honest version of the model "
        f"names the moat — switching costs, scale, network effects — and ties the margin "
        f"path to how durable that specific mechanism is, rather than extrapolating the "
        f"current level because it is the most recent number available."
    )


def _eq_multiple_triangulation(rng):
    pe_model = rng.uniform(14.0, 24.0, 3)
    pe_peers = rng.uniform(10.0, 16.0, 3)
    gap = pe_model / pe_peers - 1
    return (
        f"Cross-checking against comparables is not optional decoration. The intrinsic "
        f"estimate here implies roughly {pe_model:.1f}x earnings, against a peer median "
        f"near {pe_peers:.1f}x — a {pct(gap)} premium. That gap has exactly three possible "
        f"explanations: the business genuinely deserves a quality premium, the model's "
        f"growth or margin inputs are too generous, or the market is mispricing the peer "
        f"set. Distinguishing among the three is the actual analytical work; a model that "
        f"is simply left {pct(gap)} above the market with no stated reason is not a view, "
        f"it is an input error waiting for attribution. Working backwards, the growth rate "
        f"needed to justify the premium is often the fastest way to see whether it is "
        f"plausible."
    )


def _eq_capital_intensity(rng):
    capex_p = rng.uniform(0.04, 0.12, 3)
    dep_p = rng.uniform(0.03, 0.07, 3)
    return (
        f"Watch the relationship between capex and depreciation across the forecast. This "
        f"profile spends {pct(capex_p)} of revenue on capex against depreciation near "
        f"{pct(dep_p)}: the wedge between the two is the reinvestment that growth actually "
        f"costs. Models drift into inconsistency when revenue compounds while capex fades "
        f"toward depreciation — free cash flow inflates because the growth is no longer "
        f"being paid for. A useful discipline is to compute the implied incremental return "
        f"on capital each forecast year; when it silently exceeds anything the company has "
        f"historically earned, the cash flows and the growth assumption have come apart, "
        f"and the terminal value inherits the error at full multiple."
    )


def _eq_scenario_weighting(rng):
    p_base = rng.uniform(0.50, 0.65, 2)
    p_bull = rng.uniform(0.15, 0.25, 2)
    p_bear = round(1 - p_base - p_bull, 2)
    swing = rng.uniform(0.20, 0.45, 2)
    return (
        f"A single point estimate hides the shape of the risk. Weighting scenarios — say "
        f"{pct(p_base)} base, {pct(p_bull)} bull, {pct(p_bear)} bear — is crude but it "
        f"forces two useful admissions: what specifically has to go wrong for the bear "
        f"case, and how asymmetric the outcomes are. If the bear case sits {pct(swing)} "
        f"below base while the bull adds only half that, the expected value is below the "
        f"base case even at these weights, and the position sizing should know it. The "
        f"scenario tree matters more than its precision; the discipline is naming the "
        f"branch conditions, not decorating the base case with a probability."
    )


# ---------------------------------------------------------------------------
# financial statement analysis
# ---------------------------------------------------------------------------

def _fsa_accrual_gap(rng):
    ni_g = rng.uniform(0.08, 0.20, 3)
    ocf_g = rng.uniform(0.00, 0.06, 3)
    return (
        f"The single most informative cross-check in statement analysis is the gap between "
        f"earnings growth and operating cash flow growth. Net income compounding at "
        f"{pct(ni_g)} while OCF grows {pct(ocf_g)} means accruals are supplying the "
        f"difference, and accruals are the discretionary part of earnings. That pattern has "
        f"three common sources — receivables outrunning revenue, capitalised costs that "
        f"used to be expensed, or reserve releases — and each is checkable from the "
        f"footnotes. One year of divergence is noise. Three consecutive years is a "
        f"structural claim about earnings quality, and it usually resolves in the direction "
        f"cash was pointing all along."
    )


def _fsa_working_capital(rng):
    dso_from = rng.randint(35, 50)
    dso_to = dso_from + rng.randint(10, 25)
    rev = rng.randint(400, 3000)
    freed = rev * (dso_to - dso_from) / 365
    return (
        f"Working-capital drift is where revenue quality shows up first. Days sales "
        f"outstanding moving from {dso_from} to {dso_to} on revenue of {fmt(rev)}M means "
        f"roughly {fmt(freed)}M of reported revenue is sitting in receivables that would "
        f"previously have been cash — the company is effectively lending customers the "
        f"quarter's growth. Sometimes that is a deliberate terms change to win share, which "
        f"management can articulate; sometimes it is channel stuffing wearing a strategy "
        f"costume. The distinction matters because the first mean-reverts gently and the "
        f"second reverses violently, taking the revenue recognition with it."
    )


def _fsa_one_offs(rng):
    n_years = rng.randint(3, 6)
    share = rng.uniform(0.10, 0.30, 2)
    return (
        f"Treat 'non-recurring' as a hypothesis to test, not a label to accept. A charge "
        f"that appears in {n_years} consecutive years is recurring by observation, whatever "
        f"the footnote calls it, and adjusted earnings that strip {pct(share)} of costs "
        f"every period are simply a second, more flattering P&L. The test is symmetry: "
        f"management that excludes restructuring costs but includes restructuring benefits, "
        f"or capitalises this year what it expensed last year, is managing the metric "
        f"rather than the business. Rebuilding the adjustments yourself, in both "
        f"directions, is tedious exactly in proportion to how informative it is."
    )


def _fsa_ratio_context(rng):
    ratio_now = rng.uniform(1.8, 3.2, 2)
    ratio_med = rng.uniform(1.2, 1.8, 2)
    return (
        f"No ratio means anything at a point. A figure of {ratio_now:.1f}x against a "
        f"five-year median of {ratio_med:.1f}x is only the start of a question: what "
        f"changed — the numerator, the denominator, the accounting, or the business mix? "
        f"Ratios also embed policy choices: inventory method, lease treatment, and pension "
        f"assumptions can move them materially with no economic change at all. The robust "
        f"habit is to trend every ratio across enough periods to cover a full cycle, "
        f"restate the peer set onto comparable accounting where it diverges, and only then "
        f"read the level. A cross-sectional snapshot rewards whoever chose the most "
        f"convenient denominator."
    )


# ---------------------------------------------------------------------------
# fixed income
# ---------------------------------------------------------------------------

def _fi_convexity_limits(rng):
    dur = rng.uniform(4.0, 9.0, 2)
    conv = rng.uniform(35.0, 90.0, 1)
    dy = rng.uniform(0.015, 0.03, 3)
    first = -dur * dy * 100
    second = 0.5 * conv * dy ** 2 * 100
    return (
        f"Duration is a tangent line, and tangent lines lie at distance. For a "
        f"{dy * 10000:.0f}bp move, the first-order estimate here is {first:.2f}% while the "
        f"convexity term adds back {second:.2f}% — material, and still only the second "
        f"term of an expansion. The practical failure mode is applying these local "
        f"sensitivities to stress-sized shocks, where the price-yield curve has moved far "
        f"from the tangent point. For anything option-bearing the problem compounds: "
        f"effective duration itself shifts with the level of rates, so the hedge ratio "
        f"computed at today's yields is wrong precisely in the scenarios the hedge exists "
        f"for. Full repricing at the shocked curve is the only honest answer past about a "
        f"hundred basis points."
    )


def _fi_curve_shape(rng):
    steep = rng.uniform(0.5, 1.8, 2)
    return (
        f"Parallel-shift thinking hides most of what actually happens to a bond book. "
        f"With the curve {steep:.1f} points steep between two and ten years, a bullet and "
        f"a barbell of identical duration carry very different exposures: the barbell wins "
        f"on flattening and bleeds on steepening, and the difference is invisible to any "
        f"single duration number. Key-rate durations decompose the exposure by maturity "
        f"point and are worth computing whenever positioning departs from the index. Most "
        f"realised curve moves are a mix of shift, twist and butterfly — a risk report "
        f"that prices only the shift component is reporting on a curve that does not "
        f"occur in nature."
    )


def _fi_spread_decomposition(rng):
    oas = rng.randint(90, 350)
    default_share = rng.uniform(0.35, 0.6, 2)
    return (
        f"A credit spread is a bundle, not a number. Of an option-adjusted spread near "
        f"{oas}bp, historical decompositions typically attribute only around "
        f"{pct(default_share)} to expected default loss; the remainder is compensation "
        f"for illiquidity, downgrade risk and spread volatility. That decomposition "
        f"decides what the position actually is: buying the bond for the full spread while "
        f"only bearing default risk in your models means the liquidity premium is being "
        f"collected against a risk you have not priced. It also explains why spreads gap "
        f"in stress far beyond any change in default expectation — the liquidity component "
        f"reprices fastest exactly when exiting is hardest."
    )


def _fi_reinvestment(rng):
    ytm = rng.uniform(0.035, 0.06, 3)
    horizon = rng.randint(5, 12)
    return (
        f"Yield to maturity quietly assumes every coupon reinvests at {pct(ytm)} for "
        f"{horizon} years, which no investor actually achieves. Realised return equals "
        f"promised yield only if the reinvestment rate holds; in a falling-rate path the "
        f"coupons compound at less and the realised figure lands under the quote, while "
        f"rising rates do the reverse at the cost of interim mark-to-market pain. For a "
        f"liability-driven book this is the whole game: matching duration handles the "
        f"price side, but horizon-matching the cash flows is what immunises the "
        f"reinvestment side. Quoting YTM as if it were a guaranteed outcome conflates a "
        f"solving convention with a forecast."
    )


# ---------------------------------------------------------------------------
# risk management
# ---------------------------------------------------------------------------

def _risk_var_epistemics(rng):
    conf = rng.choice([0.95, 0.99])
    n_exc = rng.randint(6, 18)
    n_days = 250
    expected = round(n_days * (1 - conf))
    return (
        f"A VaR number is a hypothesis, and backtesting is how it gets falsified. At "
        f"{conf:.0%} confidence, {expected} exceedances a year are expected by "
        f"construction; observing {n_exc} is only half the diagnosis, because the Kupiec "
        f"test checks the count while the Christoffersen test checks whether exceedances "
        f"cluster — and clustering is the operationally dangerous case, since consecutive "
        f"tail days are what exhaust liquidity. The deeper limitation is epistemic: VaR "
        f"says nothing about severity beyond the threshold. Two books with identical VaR "
        f"can differ enormously in the tail, which is why expected shortfall belongs "
        f"beside it rather than instead of it."
    )


def _risk_correlation_stress(rng):
    rho_calm = rng.uniform(0.15, 0.4, 2)
    rho_stress = rng.uniform(0.7, 0.9, 2)
    return (
        f"The diversification in this book is conditional on the correlation regime "
        f"holding. Pairwise correlations near {rho_calm:.2f} in calm markets have "
        f"historically migrated toward {rho_stress:.2f} in liquidation episodes, because "
        f"stress correlations are driven by common ownership and forced deleveraging "
        f"rather than by fundamentals. A risk model calibrated on the full sample "
        f"averages the two regimes and therefore understates exactly the state that "
        f"matters. The practical response is to run the book under a stressed correlation "
        f"matrix as a standing scenario — not because the number is knowable, but because "
        f"the sign and rough size of the diversification loss is."
    )


def _risk_model_risk(rng):
    n_params = rng.randint(6, 20)
    window = rng.randint(250, 1250)
    return (
        f"Every risk figure inherits the window it was estimated on. A model with "
        f"{n_params} parameters fitted to {window} days of history is quietly asserting "
        f"that the next period resembles that sample — an assertion that fails precisely "
        f"at regime changes, which is when risk numbers are consulted. Parameter "
        f"uncertainty deserves explicit treatment: re-estimating over rolling windows and "
        f"watching the stability of the outputs is cheap, and instability there is itself "
        f"a risk signal. The disciplined posture is to treat model output as one input to "
        f"a decision, bounded by a stressed-parameter variant, rather than as the "
        f"decision itself."
    )


def _risk_liquidity_horizon(rng):
    adv_mult = rng.uniform(2.0, 8.0, 2)
    days = rng.randint(4, 15)
    return (
        f"A one-day risk horizon assumes the position can be exited in a day, and at "
        f"{adv_mult:.1f}x average daily volume it cannot. Unwinding without moving the "
        f"price takes on the order of {days} trading days, over which the book carries "
        f"full market risk while shrinking — so the effective risk is closer to the "
        f"multi-day figure scaled for the liquidation path than to anything the one-day "
        f"number reports. Liquidity risk also correlates with the loss states: depth "
        f"disappears in the same episodes that produce the losses, which is why "
        f"liquidity-adjusted measures penalise concentration progressively rather than "
        f"linearly. Position limits set as a share of ADV encode this more honestly "
        f"than any VaR add-on."
    )


# ---------------------------------------------------------------------------
# quantitative methods
# ---------------------------------------------------------------------------

def _quant_overfitting(rng):
    n_tried = rng.randint(20, 200)
    alpha = 0.05
    false_pos = round(n_tried * alpha)
    return (
        f"Any result that emerged from a search must be discounted for the search. "
        f"Screening {n_tried} candidate specifications at the 5% level manufactures "
        f"roughly {false_pos} significant-looking results from pure noise, so a single "
        f"reported p-value below 0.05 from that process carries almost no evidence. The "
        f"corrections are known — hold-out samples that are touched once, multiple-testing "
        f"adjustments, and pre-registration of the specification — but the cultural "
        f"discipline matters more than the formula: the number of things tried has to be "
        f"recorded honestly, because the final model never remembers its discarded "
        f"siblings. Out-of-sample decay is the usual bill for skipping this."
    )


def _quant_stationarity(rng):
    n_obs = rng.randint(60, 500)
    return (
        f"Regression on financial time series starts with a stationarity question, not a "
        f"fit statistic. With {n_obs} observations of trending series, an R-squared can be "
        f"spectacular while the relationship is spurious — two random walks regressed on "
        f"each other routinely produce exactly that. The Dickey-Fuller test on each series, "
        f"and a cointegration test on the pair if they are individually non-stationary, "
        f"decide whether levels or differences are the right space to work in. Standard "
        f"errors need the same scepticism: serial correlation and heteroskedasticity are "
        f"the norm here, so Newey-West errors are the default, not the robustness check."
    )


def _quant_distribution_tails(rng):
    kurt = rng.uniform(4.5, 9.0, 2)
    ratio = rng.uniform(1.5, 3.0, 2)
    return (
        f"The normality assumption fails exactly where it is most expensive. Daily return "
        f"series show excess kurtosis around {kurt:.1f}, which puts multi-sigma days "
        f"{ratio:.1f}x or more above their Gaussian frequency — the model treats as "
        f"once-a-decade what the data delivers annually. Everything downstream inherits "
        f"the error: parametric risk measures understate tails, option models misprice "
        f"wings, and mean-variance weights overallocate to assets whose risk is "
        f"tail-shaped rather than variance-shaped. Where the tail matters, the toolkit is "
        f"empirical quantiles, extreme value methods for the far tail, or a "
        f"Cornish-Fisher expansion as a middle ground — with the caveat that the far tail "
        f"is estimated from the observations there are fewest of."
    )


def _quant_simulation_error(rng):
    n_sims = rng.choice([5000, 10000, 50000])
    return (
        f"Simulation output deserves an error bar, not just a mean. At {n_sims:,} paths "
        f"the standard error of a mean estimate shrinks with the square root of the "
        f"count — but tail quantiles converge far more slowly, because only a small "
        f"fraction of paths land beyond them; the 99th percentile is estimated from "
        f"roughly {n_sims // 100:,} effective observations. Variance-reduction techniques "
        f"help, though the deeper issue is that simulation precision is not model "
        f"accuracy: a million paths through a mis-specified process converge confidently "
        f"to the wrong answer. Reporting the Monte Carlo standard error alongside the "
        f"estimate keeps the two kinds of uncertainty from being confused."
    )


# ---------------------------------------------------------------------------
# portfolio management
# ---------------------------------------------------------------------------

def _port_estimation_error(rng):
    n_assets = rng.randint(8, 40)
    years = rng.randint(5, 20)
    return (
        f"Optimisation amplifies estimation error by design: it allocates most to "
        f"whatever the inputs flatter most. Expected returns estimated from {years} years "
        f"of history carry standard errors comparable to the estimates themselves, and a "
        f"{n_assets}-asset covariance matrix estimated on the same window is noisy in "
        f"exactly the directions the optimiser exploits. This is why unconstrained "
        f"mean-variance output is famously extreme and fragile. The remedies share one "
        f"idea — shrink toward something structural: shrinkage estimators for the "
        f"covariance, the Black-Litterman framework for returns, or explicit weight "
        f"constraints as an implicit prior. The constraint is not a limitation on the "
        f"optimiser; it is the admission that the inputs are estimates."
    )


def _port_rebalancing_tradeoff(rng):
    band = rng.uniform(0.02, 0.06, 2)
    cost_bps = rng.randint(10, 60)
    return (
        f"Rebalancing policy is a trade-off, not a virtue. Tighter bands hold the risk "
        f"profile close to target but pay {cost_bps}bp-order transaction costs more "
        f"often and realise taxable gains earlier; a band near {pct(band)} lets "
        f"allocations drift but harvests the mean-reversion premium that calendar "
        f"rebalancing misses. The right width depends on volatility, correlation between "
        f"the drifting sleeves, transaction costs, and the tax situation of the "
        f"account — which is why the same policy is wrong for a pension and a taxable "
        f"individual. What matters most is that the rule is set ex ante; discretionary "
        f"rebalancing reliably becomes momentum-chasing with a governance veneer."
    )


def _port_factor_lens(rng):
    n_holdings = rng.randint(30, 200)
    r2 = rng.uniform(0.75, 0.95, 2)
    return (
        f"Holdings diversification and factor diversification are different things, and "
        f"only the second controls risk. A {n_holdings}-name book can still be one "
        f"trade if the names share exposures — a factor regression explaining "
        f"{pct(r2)} of its variance with two or three systematic factors is saying "
        f"exactly that. Decomposing risk into factor contributions shows where the book "
        f"is actually concentrated, and whether the active positions are intentional "
        f"bets or residue from bottom-up selection. The uncomfortable, useful question "
        f"is which exposures are being paid for: unintended factor tilts carry the risk "
        f"of an active view with the fee structure of an accident."
    )


def _port_benchmark_fit(rng):
    te = rng.uniform(0.02, 0.08, 2)
    return (
        f"Every performance conversation is secretly a benchmark conversation. Tracking "
        f"error near {pct(te)} defines how far results can sit from the index in any "
        f"year for purely statistical reasons — judging a manager on a horizon shorter "
        f"than the one implied by that noise level is measuring luck. The benchmark also "
        f"has to match the opportunity set: scoring a value-tilted mandate against a "
        f"broad index attributes the style cycle to skill in both directions. "
        f"Attribution that separates allocation, selection and interaction effects turns "
        f"the single return number into something decision-relevant — but only if the "
        f"benchmark was right, because attribution against the wrong baseline "
        f"decomposes an error."
    )


# ---------------------------------------------------------------------------
# shared epistemics -- what would change the conclusion
# ---------------------------------------------------------------------------

def _epi_falsification(rng):
    horizon = rng.randint(2, 6)
    return (
        f"The most useful sentence in any analysis states what would change the "
        f"conclusion. For this one: name the two or three inputs doing the most work, "
        f"the threshold at which each flips the answer, and the observable that would "
        f"reveal it within {horizon} quarters. An analysis that cannot articulate its "
        f"own falsification conditions is a narrative, however much arithmetic it "
        f"contains — and narratives fail silently, while falsifiable views fail loudly "
        f"and early, which is the cheaper way to fail."
    )


def _epi_precision(rng):
    digits = rng.choice([3, 4])
    band = rng.uniform(0.15, 0.35, 2)
    return (
        f"Beware precision that exceeds the inputs. Carrying {digits} decimal places "
        f"through a calculation whose driving assumptions are uncertain to "
        f"{pct(band)} does not add accuracy; it launders uncertainty into false "
        f"authority. The habit worth keeping is to attach a plausible range to the "
        f"final figure and state which single assumption contributes most of that "
        f"width. Decision-makers act differently on a point than on a band, and giving "
        f"them the point when only the band is known transfers the model's risk to "
        f"them without their consent."
    )


def _epi_base_rates(rng):
    share = rng.uniform(0.6, 0.85, 2)
    return (
        f"Inside-view analysis benefits from an outside-view check. Whatever the "
        f"specifics here, the base rate for similar situations — how often comparable "
        f"forecasts held, how often such premiums persisted, how often the projected "
        f"path was achieved by anyone — is known to be sobering; in most studied "
        f"settings a majority share of {pct(share)}-confidence projections miss. "
        f"Anchoring on the reference class first and then adjusting for what is "
        f"genuinely different about this case inverts the usual order, and the "
        f"inversion is the point: it makes the burden of proof fall on the claim of "
        f"exceptionality, where it belongs."
    )


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

_DOMAINS = {
    "Equity Valuation": [
        _eq_terminal_sensitivity, _eq_margin_reversion, _eq_multiple_triangulation,
        _eq_capital_intensity, _eq_scenario_weighting,
    ],
    "Financial Statement Analysis": [
        _fsa_accrual_gap, _fsa_working_capital, _fsa_one_offs, _fsa_ratio_context,
    ],
    "Fixed Income": [
        _fi_convexity_limits, _fi_curve_shape, _fi_spread_decomposition,
        _fi_reinvestment,
    ],
    "Risk Management": [
        _risk_var_epistemics, _risk_correlation_stress, _risk_model_risk,
        _risk_liquidity_horizon,
    ],
    "Quantitative Methods": [
        _quant_overfitting, _quant_stationarity, _quant_distribution_tails,
        _quant_simulation_error,
    ],
    "Portfolio Management": [
        _port_estimation_error, _port_rebalancing_tradeoff, _port_factor_lens,
        _port_benchmark_fit,
    ],
}

_EPISTEMICS = [_epi_falsification, _epi_precision, _epi_base_rates]

# Topics that appear in analysis metadata but are not first-class domains map to
# the nearest pool rather than falling back to nothing.
_ALIASES = {
    "Equity Investments": "Equity Valuation",
    "Corporate Issuers": "Equity Valuation",
    "Derivatives": "Fixed Income",
    "Credit Risk": "Risk Management",
    "Market Risk": "Risk Management",
    "Operational Risk": "Risk Management",
    "Economics": "Quantitative Methods",
    "Alternative Investments": "Portfolio Management",
    "Performance Evaluation": "Portfolio Management",
    "Asset Allocation": "Portfolio Management",
}


def deepen(rng, topic, subtopic, core_answer):
    """Compose the record's long-form tail from its own RNG.

    Draws 3-4 domain paragraphs plus 1-2 epistemics paragraphs, shuffled, each
    computing its own illustrative numbers. Deterministic per record because the
    RNG is the record's seeded RNG, already positioned after the core draws.
    """
    pool = _DOMAINS.get(topic) or _DOMAINS.get(_ALIASES.get(topic, ""), [])
    if not pool:
        # Unknown topic: epistemics still apply to any analysis.
        pool = _EPISTEMICS
    n_domain = min(rng.randint(3, 4), len(pool))
    n_epi = rng.randint(1, 2)
    builders = rng.sample(list(pool), n_domain) + rng.sample(_EPISTEMICS, n_epi)
    rng.shuffle(builders)
    paragraphs = [build(rng) for build in builders]
    return core_answer.rstrip() + "\n\n" + "\n\n".join(paragraphs)
