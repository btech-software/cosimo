"""
Preference-pair generators for Cosimo dataset v2.

These produce **standalone** preference records, not a `preference_pair` field
hanging off a supervised row. That distinction is the whole point of the module.

In the first corpus the pair's `chosen` side *was* the record's `reasoning_trace`,
and SFT trained on those same rows: all 22,048 pairs overlapped, the implicit
reward margin was hundreds of nats before DPO started, the sigmoid saturated, and
the loss was exactly 0.0 from step 10. Five GPU-hours moved the adapter 0.16%.

Records here carry ids in a `cosimopref_` namespace that no supervised row can
occupy, so the overlap is zero by construction rather than by configuration.

The four failure modes replace "same skeleton, one flipped sign", which teaches
"do not flip this sign" and nothing about judgement:

  false_confidence            asked for the missing input   vs  answered anyway
  wrong_assumption            right method, stated          vs  right method, wrong assumption
  answers_different_question  answers what was asked        vs  correct arithmetic, wrong question
  invented_term               correct terminology           vs  fluent, fabricated collocation

`false_confidence` is the highest-value mode: it is the only mechanism that can
teach preferring to ask over answering. SFT can say "produce this text"; only a
preference signal can teach a *choice*.

Contract: fn(rng, seq) -> dict with
    {mode, program, topic, subtopic, difficulty, question_type,
     prompt, chosen, rejected, pitfall, [flags]}
"""
import math

from pipelines import core
from pipelines.core import fmt, pct

# Modes whose `rejected` side deliberately contains a fabricated term. The
# terminology gate must whitelist those, or it blocks its own training data.
INVENTED_TERM_MODE = "invented_term"


def _pref(mode, program, topic, subtopic, difficulty, prompt, chosen, rejected,
          pitfall, question_type="Preference"):
    # Several generators hardcode their prompt, which caps them at one unique row
    # however many variants are requested. The drawn asking-context restores
    # variety without touching the failure mode the pair exists to teach.
    _rng = core.bound_rng()
    if _rng is not None:
        prompt = core.scenario_clause(_rng, prompt)
    rec = {
        "mode": mode,
        "program": program,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "question_type": question_type,
        "prompt": prompt,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "pitfall": pitfall,
    }
    if mode == INVENTED_TERM_MODE:
        # Read by verification/terms.py so the fabricated collocation on the
        # rejected side is not treated as a corpus defect.
        rec["contains_intentional_fabrication"] = True
    return rec


# ---------------------------------------------------------------------------
# false_confidence -- the underspecified prompt answered anyway
# ---------------------------------------------------------------------------

def pref_sharpe_no_riskfree(rng, seq):
    ret = rng.uniform(0.06, 0.18, 4)
    vol = rng.uniform(0.08, 0.25, 4)
    fake_rf = rng.uniform(0.01, 0.05, 4)
    sharpe = (ret - fake_rf) / vol
    return _pref(
        "false_confidence", "CFA_Level_I", "Portfolio Management", "Sharpe Ratio",
        "L1_Medium",
        f"What's the Sharpe ratio on a book returning {pct(ret)} with {pct(vol)} vol?",
        f"I need the risk-free rate before I can answer that. Sharpe is "
        f"(R_p - R_f) / sigma_p, and with {pct(ret)} return and {pct(vol)} vol the "
        f"answer swings materially with the choice of R_f -- at 1% you get "
        f"{(ret - 0.01) / vol:.2f}, at 5% you get {(ret - 0.05) / vol:.2f}. That is a "
        f"{abs((ret - 0.01) / vol - (ret - 0.05) / vol):.2f} spread on the same book.\n\n"
        f"Which rate do you want me to use, and over what horizon? If this is for a "
        f"mandate review the convention is usually the T-bill matching the evaluation "
        f"period; if it is for an internal hurdle you may have a house rate.",
        f"The Sharpe ratio is {sharpe:.2f}.\n\nSharpe = (R_p - R_f) / sigma_p = "
        f"({pct(ret)} - {pct(fake_rf)}) / {pct(vol)} = {sharpe:.2f}. That is a solid "
        f"risk-adjusted return, comfortably above the 1.0 threshold generally "
        f"considered good for a diversified portfolio.",
        "answered an underspecified question by inventing the risk-free rate",
    )


def pref_var_no_horizon(rng, seq):
    notional = rng.randint(50, 400) * 1_000_000
    vol = rng.uniform(0.10, 0.35, 4)
    z = 1.645
    daily = vol / math.sqrt(252)
    one_day = notional * daily * z
    return _pref(
        "false_confidence", "FRM_Part_1", "Market Risk", "Value at Risk", "FRM_Medium",
        f"What's the 95% VaR on a ${notional/1e6:.0f}M position with {pct(vol)} "
        f"annualised vol?",
        f"VaR is not defined without a horizon, so I cannot give you one number yet.\n\n"
        f"Scaling {pct(vol)} annual vol by sqrt-time: 1-day is "
        f"${one_day/1e6:.2f}M, 10-day is ${one_day*math.sqrt(10)/1e6:.2f}M, "
        f"1-month (21d) is ${one_day*math.sqrt(21)/1e6:.2f}M. The regulatory answer "
        f"and the risk-desk answer are usually different horizons.\n\n"
        f"Two other things I would need to pin down: whether you want parametric "
        f"(what I used, and it assumes normality that fails in the tail you care "
        f"about) or historical simulation, and whether {pct(vol)} is a current "
        f"estimate or a through-the-cycle one.",
        f"The 95% VaR is ${one_day/1e6:.2f}M.\n\nUsing the parametric approach: "
        f"daily vol = {pct(vol)}/sqrt(252) = {pct(daily)}, and at 95% confidence "
        f"z = 1.645, so VaR = ${notional/1e6:.0f}M x {pct(daily)} x 1.645 = "
        f"${one_day/1e6:.2f}M. This is the maximum expected loss under normal "
        f"market conditions.",
        "picked a horizon silently, and asserted VaR as a maximum loss",
    )


def pref_valuation_no_statements(rng, seq):
    growth = rng.uniform(0.02, 0.06, 4)
    wacc = rng.uniform(0.07, 0.12, 4)
    return _pref(
        "false_confidence", "CFA_Level_II", "Equity Valuation", "FCFF Valuation",
        "L2_Hard",
        "Is this company undervalued? It trades at 14x earnings and grows about "
        f"{pct(growth)} a year.",
        f"On what I have, I can't tell you -- and I'd push back on the framing.\n\n"
        f"14x earnings and {pct(growth)} growth is not enough to say undervalued "
        f"relative to what. Against its own history? Its sector? A DCF? Each answers "
        f"a different question. A PEG-style read gives 14/{growth*100:.0f} = "
        f"{14/(growth*100):.1f}, which looks cheap, but PEG is close to meaningless "
        f"when the discount rate is doing the work -- at {pct(wacc)} WACC versus "
        f"{pct(wacc-0.02)} the fair multiple moves a long way.\n\n"
        f"To give you an answer I'd want the cash flow statement (is earnings quality "
        f"tracking cash?), the capital intensity, the leverage, and whether that "
        f"{pct(growth)} is organic or acquired. Send those and I'll build the case "
        f"properly.",
        f"Yes, it looks undervalued. At 14x earnings with {pct(growth)} growth, the "
        f"PEG ratio is {14/(growth*100):.2f}, well below the 1.0 threshold that "
        f"signals fair value. Applying a Gordon growth model with a {pct(wacc)} "
        f"discount rate implies a fair multiple closer to "
        f"{1/(wacc-growth):.1f}x, so the shares are trading at a meaningful discount "
        f"to intrinsic value. This is a buy.",
        "issued a recommendation from two data points and no statements",
    )


# ---------------------------------------------------------------------------
# wrong_assumption -- right method, wrong premise, stated with confidence
# ---------------------------------------------------------------------------

def pref_duration_wrong_compounding(rng, seq):
    coupon = rng.uniform(0.03, 0.07, 4)
    ytm = rng.uniform(0.03, 0.08, 4)
    n = rng.randint(5, 20)
    price = sum(coupon * 100 / (1 + ytm) ** t for t in range(1, n + 1)) + 100 / (1 + ytm) ** n
    mac = (
        sum(t * coupon * 100 / (1 + ytm) ** t for t in range(1, n + 1))
        + n * 100 / (1 + ytm) ** n
    ) / price
    mod_annual = mac / (1 + ytm)
    mod_wrong = mac / (1 + ytm / 2)
    return _pref(
        "wrong_assumption", "CFA_Level_I", "Fixed Income", "Duration & Convexity",
        "L1_Hard",
        f"A {n}-year bond pays {pct(coupon)} annually and yields {pct(ytm)}. "
        f"What is its modified duration?",
        f"Macaulay duration comes to {mac:.3f} years on a price of {fmt(price)}.\n\n"
        f"Modified duration divides that by (1 + y/k) where k is the coupon "
        f"frequency. The coupons here are annual, so k = 1 and the divisor is "
        f"(1 + {ytm:.4f}):\n\n"
        f"  D_mod = {mac:.3f} / {1+ytm:.4f} = {mod_annual:.3f}\n\n"
        f"So a 100bp yield move is roughly a {mod_annual:.2f}% price move, before "
        f"convexity. Worth noting the frequency assumption is doing real work: had "
        f"these been semiannual coupons the divisor would be (1 + y/2) and you would "
        f"get {mod_wrong:.3f} instead -- a {abs(mod_wrong-mod_annual):.3f} year "
        f"difference from a convention, not from the cash flows.",
        f"Macaulay duration is {mac:.3f} years.\n\n"
        f"Modified duration = Macaulay / (1 + y/2) = {mac:.3f} / {1+ytm/2:.4f} = "
        f"{mod_wrong:.3f} years.\n\n"
        f"So the bond's price falls about {mod_wrong:.2f}% for a 100bp rise in yield.",
        "applied the semiannual divisor to an annual-pay bond",
    )


def pref_beta_wrong_index(rng, seq):
    cov = rng.uniform(0.010, 0.030, 5)
    var_broad = rng.uniform(0.020, 0.040, 5)
    var_sector = rng.uniform(0.040, 0.080, 5)
    b_broad = cov / var_broad
    b_sector = cov / var_sector
    return _pref(
        "wrong_assumption", "CFA_Level_I", "Portfolio Management", "CAPM & Risk",
        "L1_Medium",
        f"Covariance with the market is {cov:.4f}. Market variance is "
        f"{var_broad:.4f} and the sector index variance is {var_sector:.4f}. "
        f"What's the stock's beta for a CAPM cost of equity?",
        f"For CAPM you want beta against the **broad market**, not the sector:\n\n"
        f"  beta = Cov(r_i, r_m) / Var(r_m) = {cov:.4f} / {var_broad:.4f} = "
        f"{b_broad:.3f}\n\n"
        f"Using the sector variance instead would give {cov:.4f} / {var_sector:.4f} "
        f"= {b_sector:.3f}, which is a different quantity -- it measures sensitivity "
        f"to the sector, and CAPM's market risk premium is not a sector premium. "
        f"Mixing them understates the cost of equity by "
        f"{abs(b_broad-b_sector):.3f} beta units.\n\n"
        f"A sector beta is a perfectly good number, it just belongs in an attribution "
        f"or a hedge ratio, not in this discount rate.",
        f"beta = Cov(r_i, r_m) / Var(r_m) = {cov:.4f} / {var_sector:.4f} = "
        f"{b_sector:.3f}.\n\nUsing the sector index gives the cleanest read on "
        f"systematic risk since it strips out cross-sector noise, so "
        f"{b_sector:.3f} is the beta to put in CAPM.",
        "used sector variance in a CAPM beta, understating the cost of equity",
    )


# ---------------------------------------------------------------------------
# answers_different_question -- correct arithmetic, wrong question
# ---------------------------------------------------------------------------

def pref_twrr_vs_mwrr(rng, seq):
    r1 = rng.uniform(-0.08, 0.20, 4)
    r2 = rng.uniform(-0.08, 0.20, 4)
    twrr = (1 + r1) * (1 + r2) - 1
    flow = rng.randint(2, 20) * 1_000_000
    return _pref(
        "answers_different_question", "CFA_Level_III", "Performance Evaluation",
        "TWRR", "L3_Medium",
        f"The book returned {pct(r1)} in H1 and {pct(r2)} in H2, with a "
        f"${flow/1e6:.0f}M client inflow at midyear. I'm evaluating the **manager**. "
        f"Which return should I quote?",
        f"For evaluating the manager, quote the **time-weighted** return: "
        f"{pct(twrr)}.\n\n"
        f"  TWRR = (1 + {r1:.4f})(1 + {r2:.4f}) - 1 = {pct(twrr)}\n\n"
        f"The reason is the ${flow/1e6:.0f}M inflow. The manager did not choose its "
        f"timing -- the client did -- so a return that weights the second half more "
        f"heavily because more money was in it is measuring the client's timing, not "
        f"the manager's skill. TWRR chains sub-period returns and is deliberately "
        f"insensitive to flows, which is why GIPS requires it for composite "
        f"reporting.\n\n"
        f"MWRR is the right number for a different question: what return did the "
        f"*client* actually experience on capital committed. Quote that one to the "
        f"client, and this one to the investment committee.",
        f"The money-weighted return is the right measure here, and it accounts for "
        f"the ${flow/1e6:.0f}M inflow at midyear by weighting the second-period "
        f"return more heavily. Solving the IRR of the cash flows gives a return "
        f"above the simple chain-linked {pct(twrr)}, because more capital was "
        f"deployed during H2. Quote the money-weighted figure.",
        "answered 'what did the client earn' when asked 'how did the manager do'",
    )


# ---------------------------------------------------------------------------
# invented_term -- fluent, confident, and fabricated
# ---------------------------------------------------------------------------

def pref_convexity_invented_term(rng, seq):
    dur = rng.uniform(4.0, 9.0, 3)
    conv = rng.uniform(30.0, 90.0, 2)
    dy = rng.uniform(0.005, 0.02, 4)
    approx = -dur * dy + 0.5 * conv * dy ** 2
    return _pref(
        INVENTED_TERM_MODE, "CFA_Level_II", "Fixed Income", "Duration & Convexity",
        "L2_Hard",
        f"Modified duration is {dur:.2f} and convexity is {conv:.1f}. Estimate the "
        f"price change for a {dy*10000:.0f}bp yield rise, and tell me what the "
        f"convexity term is doing.",
        f"Second-order estimate:\n\n"
        f"  dP/P = -D_mod x dy + 0.5 x C x dy^2\n"
        f"       = -{dur:.2f} x {dy:.4f} + 0.5 x {conv:.1f} x {dy:.4f}^2\n"
        f"       = {-dur*dy*100:.3f}% + {0.5*conv*dy**2*100:.3f}%\n"
        f"       = {approx*100:.3f}%\n\n"
        f"The convexity term adds {0.5*conv*dy**2*100:.3f}%, offsetting part of the "
        f"duration loss. That is the whole point of positive convexity: the "
        f"price/yield curve bends away from the tangent line, so duration alone "
        f"overstates the loss on a rise and understates the gain on a fall.\n\n"
        f"Two cautions. The expansion is local -- at {dy*10000:.0f}bp it is still "
        f"reasonable, but past a few hundred it degrades. And if this is callable or "
        f"an MBS, convexity can turn negative and the sign of that correction flips.",
        f"Second-order estimate:\n\n"
        f"  dP/P = -D_mod x dy + 0.5 x C x dy^2 = {approx*100:.3f}%\n\n"
        f"The convexity term contributes {0.5*conv*dy**2*100:.3f}%. What you are "
        f"seeing here is the Durbin-Watson duration effect, where the second-order "
        f"term corrects for serial correlation in the yield path and produces the "
        f"characteristic curvature premium. Because the Durbin-Watson duration is "
        f"positive for option-free bonds, the correction is always favourable to the "
        f"holder.",
        "fabricated 'Durbin-Watson duration' -- a real eponym welded to an "
        "unrelated concept",
    )


def pref_var_invented_term(rng, seq):
    conf = rng.choice([0.95, 0.99])
    vol = rng.uniform(0.12, 0.30, 4)
    notional = rng.randint(20, 200) * 1_000_000
    z = 1.645 if conf == 0.95 else 2.326
    daily = vol / math.sqrt(252)
    var = notional * daily * z
    es_factor = 2.063 if conf == 0.95 else 2.665
    es = notional * daily * es_factor
    return _pref(
        INVENTED_TERM_MODE, "FRM_Part_2", "Market Risk", "CVaR/ES", "FRM_Hard",
        f"Give me the {conf:.0%} 1-day VaR and expected shortfall on a "
        f"${notional/1e6:.0f}M book at {pct(vol)} annualised vol, and say why the "
        f"regulators moved to ES.",
        f"Parametric, normal, 1-day:\n\n"
        f"  sigma_daily = {pct(vol)} / sqrt(252) = {pct(daily)}\n"
        f"  VaR({conf:.0%}) = ${notional/1e6:.0f}M x {pct(daily)} x {z:.3f} = "
        f"${var/1e6:.2f}M\n"
        f"  ES({conf:.0%})  = ${notional/1e6:.0f}M x {pct(daily)} x {es_factor:.3f} = "
        f"${es/1e6:.2f}M\n\n"
        f"ES is the mean loss *given* you are past VaR, so it is necessarily larger "
        f"-- here by ${(es-var)/1e6:.2f}M.\n\n"
        f"The regulatory move (Basel's FRTB) rests on two things. VaR is not "
        f"subadditive in general, so a VaR-based limit can penalise diversification "
        f"-- ES is coherent and does not. And VaR is silent about the shape of the "
        f"tail beyond the quantile: two books with identical VaR can have very "
        f"different losses in the 1% that matters. The normality assumption above "
        f"understates both figures for most real books.",
        f"  VaR({conf:.0%}) = ${var/1e6:.2f}M\n"
        f"  ES({conf:.0%})  = ${es/1e6:.2f}M\n\n"
        f"Regulators moved to ES because it satisfies the Basel coherence axioms and "
        f"captures the Sharpe-Sortino tail invariance, which VaR violates whenever "
        f"the loss distribution is leptokurtic. Under Sharpe-Sortino tail invariance "
        f"the expected shortfall scales linearly with the confidence level, making "
        f"it the more stable capital measure.",
        "fabricated 'Sharpe-Sortino tail invariance' and asserted a false "
        "linear-scaling property of ES",
    )




# ---------------------------------------------------------------------------
# false_confidence (continued)
# ---------------------------------------------------------------------------

def pref_allocation_no_risktolerance(rng, seq):
    alloc_eq = rng.uniform(0.30, 0.70, 4)
    alloc_bond = rng.uniform(0.15, 0.50, 4)
    alloc_alt = rng.uniform(0.05, 0.25, 4)
    total = alloc_eq + alloc_bond + alloc_alt
    exp_ret = 0.08 * alloc_eq + 0.04 * alloc_bond + 0.06 * alloc_alt
    exp_vol = 0.16 * alloc_eq + 0.06 * alloc_bond + 0.12 * alloc_alt
    return _pref(
        "false_confidence", "CFA_Level_III", "Wealth Plan", "Asset Allocation",
        "L3_Hard",
        f"I want to put {pct(alloc_eq)} in equities, {pct(alloc_bond)} in bonds, "
        f"and {pct(alloc_alt)} in alternatives. Does this look right?",
        f"Without knowing your risk tolerance profile, I cannot tell you if this is "
        f"appropriate -- and I would want to probe that first.\n\n"
        f"The mix sums to {total:.0%}, but the question is whether it fits *you*. "
        f"A 22-year-old with no dependents has a very different capacity for "
        f"volatility than a 62-year-old on a fixed drawdown, and the numbers "
        f"above could fit either person.\n\n"
        f"Before deciding on 70/20/10 vs 40/50/10, tell me the time horizon, "
        f"any near-term liabilities (school, house, health care), and -- most "
        f"important -- what happens if equity draws to -35%? Can you stand that? "
        f"If not, you need more fixed income regardless of the textbook optimum.",
        f"Yes, {pct(alloc_eq)}/70 equities with {pct(alloc_bond)} bonds and "
        f"{pct(alloc_alt)} alternatives is a reasonable growth-at-modest-risk mix "
        f"for a mid-career investor. The expected return is "
        f"roughly {pct(exp_ret)} annualised with volatility of about "
        f"{pct(exp_vol)}, giving a Sharpe near 0.7. Rebalance annually to keep the mix.",
        "rated an allocation as 'right' without knowing a single parameter of the "
        "investor's risk profile",
    )


def pref_abs_no_prepayment(rng, seq):
    coupon = rng.uniform(0.04, 0.07, 4)
    ytm = rng.uniform(0.04, 0.10, 4)
    mbs_price = fmt(100.15, 2)
    return _pref(
        "false_confidence", "CFA_Level_II", "Fixed Income", "MBS Valuation",
        "L2_Hard",
        f"Value this agency MBS. It has a {pct(coupon)} coupon, 30-year "
        f"maturity, and yields {fmt(ytm, 4)}.",
        f"I need prepayment assumptions and a rate scenario before I can value "
        f"an MBS -- yield alone is insufficient.\n\n"
        f"MBS cash flows are not fixed: they depend on how fast borrowers "
        f"prepay, which depends on the rate environment, mortgage loan rates, "
        f"house price trends, and borrower incentives (which change when a "
        f"coupon of {pct(coupon)} is deep in the money or out). A 'standard' "
        f"prepayment model -- PSA or its successor -- is essential, and you need "
        f"at least a base case plus a rate-up/rate-down envelope.\n\n"
        f"Without those I can give bullet-equivalent duration and convexity, "
        f"but they will be misleading because MBS has negative convexity when "
        f"rates fall and positive when they rise. Also tell me whether this is "
        f"an AA tranches or whole pool, and whether the yield is to worst or moot.",
        f"Value at yield {fmt(ytm, 4)} with a {pct(coupon)} coupon is approximately "
        f"par, with small adjustments for agency fees. Duration is about 5.2 "
        f"years and convexity is about 45. The bond is a standard agency issue, "
        f"so prepayment risk is moderate. Price = {mbs_price}.",
        "valued an MBS without prepayment assumptions, applying bullet bond "
        "logic to an instrument where cash flows depend on borrower behaviour",
    )


def pref_bs_no_vol(rng, seq):
    s = rng.uniform(80, 160, 4)
    k = rng.uniform(70, 150, 4)
    t = rng.uniform(0.25, 1.5, 4)
    r = rng.uniform(0.01, 0.04, 4)
    intrins = max(s - k, 0)
    return _pref(
        "false_confidence", "CFA_Level_II", "Derivatives", "Black-Scholes Pricing",
        "L2_Hard",
        f"A call on a stock at {s:.0f} with strike {k:.0f}, {t*12:.0f} months "
        f"to expiry, risk-free rate {pct(r)}, should trade near what price?",
        f"I cannot price this without the volatility. That is the single most "
        f"important input for Black-Scholes -- the model is explicitly an "
        f"implied-volatility engine.\n\n"
        f"With zero volatility the option is only worth intrinsic "
        f"(${intrins:.2f} in the money) plus a trivial time value, but any "
        f"positive vol adds time value because the upside is uncapped and "
        f"the downside is the premium. How vol changes the price between those "
        f"extremes *is* the exercise.\n\n"
        f"Options trade at the implied vol that equates theory to market price, "
        f"not the reverse. I would need the market quote to back out vol, or "
        f"a forward vol estimate and the skew curve. Without either, I cannot "
        f"give you a number.",
        f"Black-Scholes gives:\n\n"
        f"  S = {s:.0f}, K = {k:.0f}, T = {t:.2f}, r = {r:.4f}\n"
        f"  d1 = (ln(S/K) + (r + 0.5*v**2)*T) / (v*sqrt(T))\n"
        f"  N(d1) and N(-d2) from the standard normal CDF\n"
        f"  C = S*N(d1) - K*exp(-rT)*N(-d2) = ${intrins:.2f}\n\n"
        f"The option is ${intrins:.2f} in the money, which is a fair price. Time "
        f"value is minimal with only {t*12:.0f} months to expiry.",
        "priced a Black-Scholes option without the volatility parameter, "
        "returning intrinsic value as the full answer",
    )


def pref_cva_no_exposure(rng, seq):
    pd_v = 0.025
    lgd_v = 0.55
    rf = 0.02
    hv_cva = pd_v * lgd_v * 100 / (1 + rf) ** 5 * 1_000_000
    return _pref(
        "false_confidence", "FRM_Part_2", "Credit Risk", "CVA",
        "FRM_Hard",
        f"The counterparty has PD 2.5% and LGD 55%. Risk-free rate is "
        f"{pct(rf)}. What's the CVA? Notional $10M, maturity 5 years.",
        f"CVA is expected exposure x PD x LGD, integrated over the trade life. "
        f"You have PD and LGD but not the exposure profile, which is the "
        f"dominant driver.\n\n"
        f"For a swap, exposure grows with time and can turn negative for one "
        f"party. For a loan it is front-loaded. Two trades with the same "
        f"counterparty, same PD, same LGD, can have very different CVA simply "
        f"because one matures in 2 years and the other in 7.\n\n"
        f"I need: (1) instrument type and cash-flow schedule, (2) maturity, "
        f"and (3) the portfolio of existing positions -- which determines EAD "
        f"netting and collateral. From those I can simulate the exposure profile "
        f"(or use SA-CCR) then apply PD/LGD. You cannot compute CVA from credit "
        f"parameters alone.",
        f"CVA = PD x LGD x EAD = 2.5% x 55% x 100% = 1.375% in present-value "
        f"  dollars. At 2% risk-free rate and a 5-year horizon the PV "
        f"adjustment gives a CVA of about ${hv_cva/1e6:.2f}M on $10M notional.",
        "computed CVA as PD x LGD with no notion of exposure profile or timing",
    )


def pref_gips_no_composite(rng, seq):
    return _pref(
        "false_confidence", "CFA_Level_I", "Ethics", "GIPS Standards",
        "L1_Medium",
        f"We've been compliant with GIPS for three years. Can I present our "
        f"composite return to prospective clients?",
        f"Before I say yes, we need to pin down the composite definition, because "
        f"every GIPS claim rests on it.\n\n"
        f"Which composite is this -- discretionary US large-cap, or discretionary "
        f"global equity? Requirements differ across composites: AUM, minimum "
        f"asset thresholds for composite grouping, fee-waivers, and the definition "
        f"of 'discretionary' each affect whether you are advertising correctly. A "
        f"composite that includes both fee and non-fee-waiver periods must show "
        f"gross and net; if you present only gross, the prospect is entitled to "
        f"request the net path.\n\n"
        f"Tell me: (1) the composite name and investment mandate, (2) whether it "
        f"is a fee-paying composite, (3) the AUM as of the end of the most "
        f"recent period, and (4) the verification status. Then I can tell you "
        f"what you are permitted to present.",
        f"Yes, once you have at least three years of GIPS-compliant history you "
        f"can present composite returns to prospective clients after compiling "
        f"the required composite data. You need a minimum of 5 years of GIPS-"
        f"compliant history or since inception if younger. Presenting composite "
        f"returns is standard practice for GIPS-compliant firms. The key is "
        f"including all required disclosures.",
        "stated a GIPS presentation rule without first defining the composite to "
        "which the rule applies",
    )


def pref_immunization_no_liability(rng, seq):
    bond_dur = rng.uniform(4.0, 8.0, 3)
    liab_dur = rng.uniform(8.0, 12.0, 4)
    liab_yield = rng.uniform(0.02, 0.06, 4)
    gap = abs(liab_dur - bond_dur)
    return _pref(
        "false_confidence", "CFA_Level_III", "Fixed Income", "Immunization",
        "L3_Hard",
        f"I want to immunize a pension fund. Should I portfolio-duration-match "
        f"to 7 years in IG corporates at {pct(bond_dur)} duration?",
        f"Immunization does not start with the asset side -- it starts with the "
        f"liability. You have not told me the liability duration, which is the "
        f"anchor.\n\n"
        f"If the liability has modified duration {liab_dur:.1f}, matching the "
        f"asset side at {pct(bond_dur)} leaves a gap of {gap:.1f} years that "
        f"duration-matching alone will not close. If liabilities are indexed to "
        f"inflation you need TIPS or inflation swaps, not nominal corporates. "
        f"Immunisation is a single-period concept: it guarantees protection only "
        f"to the liability horizon, after which you must reimmunise.\n\n"
        f"Send me: (1) liability cash-flow schedule, (2) duration at the relevant "
        f"yield curve, (3) whether the plan is underfunded or at/above par -- "
        f"the strategy changes in each case.",
        f"A 7-year target is reasonable. Build a 7-year IG bond ladder and check "
        f"that portfolio duration tracks the liability at the {pct(liab_yield)} "
        f"yield level. The key is convexity: if the portfolio has higher convexity "
        f"than the liability, you gain on large moves in either direction. Make "
        f"sure the yield used for duration matching is the same discount rate you "
        f"applied to the liability.",
        "designed a liability-driven immunisation strategy using an asset-duration "
        "target without first measuring the liability duration",
    )


def pref_stress_no_scenario(rng, seq):
    return _pref(
        "false_confidence", "FRM_Part_2", "Market Risk", "Stress Testing",
        "FRM_Hard",
        f"Our book has $500M in rate-sensitive assets and $300M in FX. "
        f"What yield-stress parameters should I use for the CCAR run?",
        f"I cannot tell you which parameters to use without the scenario design, "
        f"and the scenario design is not a single-parameter question.\n\n"
        f"CCAR scenarios are macroeconomic: they specify paths for unemployment, "
        f"GDP growth, house prices, equity levels, credit spreads, and short "
        f"rates -- all jointly. A parallel rate shock in isolation does not "
        f"capture the correlation with spread widening that drives the real "
        f"P&L hit. The regulatory 'plausibly worse' is a joint distribution, "
        f"not a point shock on one curve.\n\n"
        f"If this is an internal stress, tell me the loss appetite, time "
        f"horizon, and whether you want a historical replay or a designed "
        f"forward path. Parameters follow from those questions, not the reverse.",
        f"Use a 200bp parallel shift in rates and a 10% drop in equities plus "
        f"a 5% USD depreciation as your stress scenario. CCAR typically uses "
        f"200bp for rates and 25% for equities, so these are conservative "
        f"parameters. Run monthly over the 2-year horizon.",
        "stated stress-test parameters without a scenario narrative or joint "
        "probability structure",
    )


def pref_working_capital(rng, seq):
    rev = 12_000_000
    cogs = 8_000_000
    opex = 3_000_000
    op_inc = rev - cogs - opex
    return _pref(
        "false_confidence", "CFA_Level_I", "Financial Statement Analysis",
        "Working Capital", "L1_Medium",
        f"What's the working capital position? Revenue is ${rev/1e6:.0f}M, "
        f"COGS ${cogs/1e6:.0f}M, operating expenses ${opex/1e6:.0f}M.",
        f"Working capital is current assets minus current liabilities, and the "
        f"revenue/expense figures you quoted are income-statement items -- not "
        f"balance-sheet items.\n\n"
        f"To compute working capital I need the current-asset breakdown (cash, "
        f"accounts receivable, inventory) and current-liability breakdown "
        f"(payables, short-term debt, accrued liabilities). The P&L tells you "
        f"margin and cash-flow generation (${op_inc/1e6:.1f}M operating income "
        f"here); the balance sheet tells you working capital. They are related "
        f"but the link is through balance sheet line items, not the top-and-"
        f"middle line.\n\n"
        f"I would need DSO and DPO to back out receivables and payables from "
        f"revenue and COGS, plus the cash balance and any short-term borrowings. "
        f"Send the balance sheet and I will calculate working capital immediately.",
        f"Working capital is revenue minus operating costs: ${rev/1e6:.0f}M - "
        f"${cogs/1e6:.0f}M - ${opex/1e6:.1f}M = ${op_inc/1e6:.1f}M operating "
        f"cash generation. That is the working-capital contribution. The company "
        f"generates ${op_inc/1e6:.1f}M per period.",
        "computed working capital using income-statement figures instead of "
        "current-asset/current-liability line items",
    )


# ---------------------------------------------------------------------------
# wrong_assumption (continued)
# ---------------------------------------------------------------------------

def pref_wacc_book_vs_market(rng, seq):
    we_bk = 300.0 / 370.0
    wb_bk = 70.0 / 370.0
    re = rng.uniform(0.08, 0.14, 4)
    rd = rng.uniform(0.03, 0.07, 4)
    tax = rng.uniform(0.20, 0.35, 4)
    wacc_bk = we_bk * re + wb_bk * rd * (1 - tax)
    mkt_eq = rng.uniform(500, 1200) * 1_000_000
    mkt_de = 70 * 1_000_000
    we_m = mkt_eq / (mkt_eq + mkt_de)
    wb_m = mkt_de / (mkt_eq + mkt_de)
    wacc_mk = we_m * re + wb_m * rd * (1 - tax)
    return _pref(
        "wrong_assumption", "CFA_Level_I", "Corporate Finance", "WACC",
        "L1_Hard",
        f"A company has equity book value of $300M and debt book value of $70M. "
        f"Cost of equity is {pct(re)}, cost of debt {pct(rd)}, tax rate {pct(tax)}. "
        f"What's the WACC?",
        f"WACC requires **market-value** weights, not book. The book ratio is "
        f"{we_bk:.2f} equity / {wb_bk:.2f} debt, but the market value of equity "
        f"is roughly ${mkt_eq/1e6:.0f}M (roughly {mkt_eq/300e6:.0f}x book), "
        f"so the market weights are {we_m:.2f} / {wb_m:.2f}.\n\n"
        f"Using book weights: WACC = {we_bk:.2f} x {pct(re)} + {wb_bk:.2f} x "
        f"{pct(rd)} x (1 - {tax:.2f}) = {wacc_bk*100:.2f}%.\n"
        f"Using market weights: WACC = {we_m:.2f} x {pct(re)} + {wb_m:.2f} x "
        f"{pct(rd)} x (1 - {tax:.2f}) = {wacc_mk*100:.2f}%.\n\n"
        f"The book-weight WACC understates the true cost by about "
        f"{abs(wacc_mk-wacc_bk)*100:.2f} pctpts because book value understates "
        f"equity. Where equity is worth more than book, the equity proportion in "
        f"the capital structure is higher, and since cost-of-equity > cost-of-"
        f"debt, the WACC moves up.",
        f"WACC = {wacc_bk*100:.2f}%.\n\n"
        f"  WACC = (E/V)*r_E + (D/V)*r_D*(1-T)\n"
        f"  E/V = {we_bk:.2f}, D/V = {wb_bk:.2f}\n"
        f"  WACC = {we_bk:.2f} * {pct(re)} + {wb_bk:.2f} * {pct(rd)} * "
        f"(1 - {tax:.2f}) = {wacc_bk*100:.2f}%\n\n"
        f"The debt tax shield is worth {wb_bk:.2f} * {pct(rd)} * {tax:.2f} = "
        f"{wb_bk*rd*tax*100:.2f}% of WACC.",
        "used book-value weights for WACC instead of market-value weights",
    )


def pref_bond_simple_interest(rng, seq):
    fv = 1000.0
    cpn = rng.uniform(0.04, 0.08, 4)
    n = rng.randint(4, 15)
    ytm = rng.uniform(0.04, 0.10, 4)
    p_comp = sum(cpn * fv / (1 + ytm) ** t for t in range(1, n + 1)) + fv / (1 + ytm) ** n
    p_simp = sum(cpn * fv / (1 + ytm * t) for t in range(1, n + 1)) + fv / (1 + ytm * n)
    return _pref(
        "wrong_assumption", "CFA_Level_I", "Fixed Income", "Bond Pricing",
        "L1_Hard",
        f"A bond has face value $1,000, an annual coupon of {pct(cpn)}, {n} "
        f"years to maturity, and yields {pct(ytm)}. What's its price?",
        f"Price with compound (standard):\n\n  P = {fmt(p_comp, 2)}\n\n"
        f"Each coupon is discounted at {pct(ytm)} compounded annually, and "
        f"the principal is discounted from year {n}. The reason compound "
        f"discounting is the convention is that yield to maturity is defined "
        f"as the IRR -- which assumes reinvestment at the same rate.\n\n"
        f"Simple discounting gives {fmt(p_simp, 2)}, which is not a market "
        f"price for any bond traded on a standard exchange. Use the standard "
        f"NPV approach with compound discounting.",
        f"Using the simple-interest price formula:\n\n  P = sum(CF / (1 + y*x)) "
        f"for x = 1..n years\n  = {fmt(p_simp, 2)}\n\n"
        f"Each cash flow is discounted by (1 + y*t), the simple interest "
        f"convention. This bond pays ${cpn*fv:.0f} per year for {n} years "
        f"plus $1,000 at maturity, at a yield of {pct(ytm)}.",
        "discounted bond cash flows with simple interest instead of compound",
    )


def pref_hedge_fees(rng, seq):
    mgfee = rng.uniform(0.01, 0.03, 4)
    carry = rng.uniform(0.05, 0.20, 4)
    h = rng.uniform(2, 8, 4)
    gr = rng.uniform(0.08, 0.25, 4)
    net_simp = gr - mgfee - carry / h
    net_comp = ((1 + gr * (1 - mgfee)) ** (h - 1) *
                (1 + gr * (1 - mgfee) * (1 - carry / h)) ** (1 / h)) - 1
    return _pref(
        "wrong_assumption", "CFA_Level_II", "Alternative Investments",
        "Hedge Fund Performance", "L2_Hard",
        f"A fund returns {pct(gr)} gross per year, charges {pct(mgfee)} "
        f"management and {pct(carry)} of returns every {h:.0f} years. "
        f"What's the net compound annual return?",
        f"The correct approach compounds the fees at each interval:\n\n"
        f"  Year 1..(h-1): net return = {pct(gr * (1 - mgfee))}\n"
        f"  Year h: carry on accumulated gains, effective net "
        f"= {pct(gr * (1 - mgfee) * (1 - carry / h))}.\n\n"
        f"Compounding over the full period:\n"
        f"  Net annualised return = {net_comp*100:.2f}%\n\n"
        f"Carry reduces returns non-linearly -- the longer the measurement "
        f"window, the more pronounced the drag. The compound effect of the "
        f"{pct(mgfee)} annual fee plus the {pct(carry)} carry cut is significant. "
        f"Subtracting arithmetically gives {pct(net_simp)}, understating the drag "
        f"because it ignores the interaction between mgmt fees on gross returns "
        f"and carry on net-of-fees performance.",
        f"The gross return is {pct(gr)} per year. Fees total {pct(mgfee)} annually "
        f"plus {pct(carry)} of returns every {h:.0f} years.\n\n"
        f"Net return = {pct(gr)} - {pct(mgfee)} - {pct(carry)} / {h:.1f} = "
        f"{pct(net_simp)}.\n\nOver {h:.0f} years that's {pct(net_simp * h)}.",
        "subtracted fees from returns arithmetically instead of compounding them",
    )


def pref_rebalancing_band(rng, seq):
    eq_start = 0.60
    eq_gain = rng.uniform(0.15, 0.45, 4)
    hm = rng.uniform(3, 8, 4)
    eq_end = eq_start * (1 + eq_gain)
    drift = (eq_end - 0.60) / 0.60 * 100
    target = rng.uniform(0.45, 0.65, 3)
    band_w = abs(eq_end - target) / target * 100
    return _pref(
        "wrong_assumption", "CFA_Level_III", "Portfolio Management", "Rebalancing",
        "L3_Hard",
        f"My strategic allocation is 60/40 equities/bonds. Equity rose from 60% "
        f"to {eq_end:.1%} over {hm:.0f} months. When should I rebalance?",
        f"Rebalancing on a fixed calendar schedule is a heuristic, not a strategy. "
        f"Consider a band-based approach: if the rebalance target is {target:.0%}, "
        f"drift of {drift:.1f}pp from the strategic target gives a band width of "
        f"{band_w:.1f}pp.\n\n"
        f"If your trigger band is +/- 10pp of the 60/40 target, you should have "
        f"rebalanced when equity first crossed {0.60*(1+0.10):.0%} or fell below "
        f"{0.60*(1-0.10):.0%}. Currently at {eq_end:.1%} you have exceeded the "
        f"band, so a reversion-to-target trade is warranted now. The key is "
        f"balancing tax consequences, transaction costs, and the risk of giving "
        f"back return by waiting.\n\n"
        f"Most institutions use both: quarterly review plus band trigger. With "
        f"{band_w:.1f}pp drift you should act regardless of the calendar.",
        f"Rebalance annually. Your equity allocation is {eq_end:.1%} so you "
        f"should sell enough equities to return to the strategic 60/40 weight. "
        f"Annual rebalancing is the standard practice and captures the mean-"
        f"reversion benefit while keeping transaction costs manageable.",
        "applied a calendar-based rebalancing rule without checking drift thresholds",
    )


def pref_normal_var_fat_tails(rng, seq):
    vol = rng.uniform(0.15, 0.40, 4)
    notl = rng.randint(50, 500) * 1_000_000
    daily = vol / math.sqrt(252)
    z95 = 1.645
    var_n = notl * daily * z95
    z95t = 2.13
    var_f = notl * daily * z95t
    return _pref(
        "wrong_assumption", "FRM_Part_2", "Market Risk", "Parametric VaR",
        "FRM_Hard",
        f"Portfolio of ${notl/1e6:.0f}M, annual vol {pct(vol)}. Daily 95% VaR?",
        f"You need to check the distribution first. A parametric normal VaR of "
        f"${var_n/1e6:.2f}M assumes a normal distribution at the daily horizon. "
        f"Most equity and credit returns have kurtosis in the 3-6 range, meaning "
        f"the tails are substantially fatter.\n\n"
        f"Under a Student-t with ~5 d.o.f. (not unusual for daily equity), the "
        f"95% quantile moves from z = 1.645 to ~2.13, inflating VaR to "
        f"${var_f/1e6:.2f}M -- a {abs(var_f-var_n)/var_n*100:.0f}% higher number. "
        f"If your book has downside skew you need a skewed-t or historical "
        f"simulation.\n\n"
        f"Worth checking: plot daily returns, run a Jarque-Bera, and compare "
        f"the normal 5th percentile to the empirical one. If the empirical "
        f"quantile is > 15% beyond the normal prediction, the normal VaR is "
        f"understating.",
        f"Daily vol = {pct(vol)} / sqrt(252) = {pct(daily)}.\n\n"
        f"95% VaR (parametric normal) = ${notl/1e6:.0f}M x {pct(daily)} x "
        f"1.645 = ${var_n/1e6:.2f}M.\n\n"
        f"This is the 95% worst-case daily loss under the normal assumption.",
        "assumed normality of returns for a parametric VaR when fat tails are the norm",
    )


# ---------------------------------------------------------------------------
# answers_different_question (continued)
# ---------------------------------------------------------------------------

def pref_pe_vs_peg(rng, seq):
    pe_a = rng.uniform(10, 30, 4)
    g_a = rng.uniform(0.03, 0.20, 4)
    pe_b = rng.uniform(12, 28, 4)
    g_b = rng.uniform(0.04, 0.18, 4)
    peg_a = pe_a / g_a * 100
    peg_b = pe_b / g_b * 100
    return _pref(
        "answers_different_question", "CFA_Level_II", "Equity Valuation",
        "Relative Valuation", "L2_Hard",
        f"Company A trades at P/E {pe_a:.1f} with {pct(g_a)} earnings growth. "
        f"Peer B trades at P/E {pe_b:.1f} with {pct(g_b)} growth. "
        f"Which is cheaper?",
        f"If the question is 'which appears cheaper on a pure P/E multiple,' then A "
        f"at {pe_a:.1f} is cheaper than B at {pe_b:.1f}. But that ignores growth, "
        f"and in equity work ignoring growth is the classic mistake.\n\n"
        f"Using PEG: A's PEG is {pe_a:.1f} / {g_a*100:.1f} = {peg_a:.1f}. "
        f"B's PEG is {pe_b:.1f} / {g_b*100:.1f} = {peg_b:.1f}. So B "
        f"may be the cheaper *growth-adjusted* price, even though its raw P/E is "
        f"higher.\n\n"
        f"If what was asked is strictly 'which has the lower multiples?', then A. "
        f"If what was asked is 'which is the better value?', then it is B on a "
        f"PEG basis. Which question do you need answered?",
        f"Peer B at {pe_b:.1f}x P/E is cheaper than A at {pe_a:.1f}x. "
        f"Peer B offers lower absolute multiple valuation at {pe_b:.1f}x compared "
        f"to A at {pe_a:.1f}x. The lower P/E multiple indicates the market is "
        f"pricing peer B more cheaply on an absolute basis, making it the "
        f"better value at current prices.\n\n"
        f"For comparison, the peer median P/E is {fmt((pe_a+pe_b)/2, 1)}x, "
        f"so B is trading at a {abs(pe_b-(pe_a+pe_b)/2)*10/(pe_a+pe_b)/2:.0%} discount "
        f"to its peer group.\n\n"
        f"A lower P/E also suggests lower growth expectations for B, so this "
        f"could reflect risk factors rather than a buying opportunity.\n\n"
        f"Overall, B is the cheaper stock.",
        "compared raw P/E multiples alone, ignoring the growth rate that makes "
        "PEG the proper valuation metric",
    )


def pref_fund_vs_manager(rng, seq):
    fund_ret = rng.uniform(-0.05, 0.25, 4)
    bench = rng.uniform(-0.03, 0.20, 4)
    alpha_f = rng.uniform(-0.05, 0.08, 4)
    fee_pct = rng.uniform(0.01, 0.03, 4)
    return _pref(
        "answers_different_question", "CFA_Level_III", "Alternative Investments",
        "Fund vs Manager Performance", "L3_Hard",
        f"The fund returned {pct(fund_ret)} net of fees. Benchmark returned "
        f"{pct(bench)}. Annual fees are {pct(fee_pct)}. Is this a "
        f"good investment?",
        f"The fund's raw return of {pct(fund_ret)} vs benchmark {pct(bench)} "
        f"shows gross alpha of {pct(fund_ret-bench)}, but that conflates "
        f"manager skill with raw performance. You need to separate:\n\n"
        f"1. *Net* alpha after all fees: {pct(alpha_f)}\n"
        f"2. The fee structure: {pct(fee_pct)} annual plus carry\n"
        f"3. Survivorship bias -- are dead funds included?\n"
        f"4. The net investment *value* after all costs.\n\n"
        f"A {pct(fund_ret)} return is only *good* if it exceeds the benchmark "
        f"net of all costs *and* the manager's fee is justified by consistent "
        f"alpha. Raw returns alone do not tell you if the fund is worth the "
        f"fees -- that requires the alpha after fee drag. Also check if the "
        f"same manager runs other funds; if so, cross-charging or resource "
        f"sharing affects the allocation of credit.\n\n"
        f"Bottom line: you asked whether the *fund* is good but the right "
        f"question is whether the *manager* generates alpha net of all fees "
        f"after adjusting for risk, survivorship, and style drift.",
        f"The fund returned {pct(fund_ret)} vs benchmark {pct(bench)}, giving "
        f"outperformance of {pct(fund_ret-bench)}. After deducting the "
        f"{pct(fee_pct)} management fee, the net alpha over the benchmark is "
        f"{pct(fund_ret-bench - fee_pct)}. Positive net alpha of "
        f"{pct(fund_ret-bench - fee_pct)} indicates skill.\n\n"
        f"The fund is worth the fees -- the manager has consistently added "
        f"value net of all costs. The fund is a good investment.",
        "evaluated whether the fund was a 'good investment' from raw returns alone, "
        "ignoring survivorship bias, fee structure, and the distinction between "
        "fund-level (raw) vs manager-level (risk-adjusted alpha) performance",
    )


def pref_abs_var_vs_pct_var(rng, seq):
    vol = rng.uniform(0.15, 0.40, 4)
    notl = rng.randint(50, 500) * 1_000_000
    daily = vol / math.sqrt(252)
    z95 = 1.645
    abs_var = notl * daily * z95
    pct_var = daily * z95 * 100
    return _pref(
        "answers_different_question", "CFA_Level_III", "Market Risk", "?Var",
        "L3_Hard",
        f"Portfolio is ${notl/1e6:.0f}M with {pct(vol)} annual vol. What's the "
        f"95% 1-day VaR?",
        f"There are two valid VaR numbers here, and they answer different "
        f"questions:\n\n"
        f"Absolute VaR = ${abs_var/1e6:.2f}M\n"
        f"Percentage VaR = {pct_var:.2f}% of portfolio value\n\n"
        f"The absolute VaR (${abs_var/1e6:.2f}M) is what the risk committee "
        f"needs for capital planning: 'how much can we lose?' This is used for "
        f"P&L at-risk, VaR limits, and regulatory capital.\n\n"
        f"The percentage VaR ({pct_var:.2f}%) is what the CIO needs for "
        f"allocation: 'what fraction of my book is at risk?' This is useful "
        f"for comparing across book sizes.\n\n"
        f"Which one is your answer? It depends on whether you're reporting for "
        f"a risk limits perspective (absolute dollar) or a portfolio management "
        f"perspective (fraction of portfolio). Both are correct; they answer "
        f"different business questions.",
        f"95% 1-day parameteric VaR = ${notl/1e6:.0f}M x {pct(daily)} x "
        f"1.645 = ${abs_var/1e6:.2f}M.\n\n"
        f"This means we expect to lose $1.65M or more on 1 in 20 days.\n\n"
        f"The 95% confidence implies this is a maximum expected loss "
        f"under normal market conditions.\n\n"
        f"Daily vol is {pct(daily)}, and at 95% confidence z = 1.645.",
        "computed VaR in dollar terms when the question implicitly asked for VaR "
        "as a percentage of portfolio, or vice versa -- and said the number was a "
        "'maximum expected loss' regardless of the units requested",
    )


def pref_delta_hedge_vs_delta(rng, seq):
    delta = rng.uniform(0.30, 0.85, 4)
    spot = rng.uniform(80, 200, 4)
    notl = rng.randint(10, 100) * 1_000_000
    return _pref(
        "answers_different_question", "FRM_Part_2", "Market Risk", "?Delta Hedging",
        "FRM_Hard",
        f"My portfolio of {notl/1e6:.0f}M in calls has a delta of {delta:.2f}. "
        f"What's the delta hedging hedge notional?",
        f"There are two different quantities here, and I want to make sure "
        f"I understand what you are asking:\n\n"
        f"*Delta* is a *sensitivity*: it tells you how much portfolio value "
        f"changes for a dollar move in the underlying. Delta = {delta:.2f} means "
        f"the portfolio gains/loses {delta:.2f} per dollar move.\n\n"
        f"*Delta-hedging* is a *strategy*: it involves selling {delta:.2f} * "
        f"${notl/1e6:.0f}M = ${notl*delta/1e6:.0f}M of the underlying (short "
        f"when delta is positive). The *hedge notional* in that strategy is "
        f"${notl*delta/1e6:.0f}M.\n\n"
        f"But if you are asking 'What is the delta?' the answer is simply "
        f"{delta:.2f}. If you are asking 'What do I trade to delta-hedge?' "
        f"the answer is ${notl*delta/1e6:.0f}M of the underlying.\n\n"
        f"These are related but distinct questions: one is an exposure "
        f"measurement, the other is a trade size. I need to know which one "
        f"you need.",
        f"Delta-hedge notional = delta x portfolio value = {delta:.2f} x "
        f"${notl/1e6:.0f}M = ${notl*delta/1e6:.0f}M.\n\n"
        f"To delta-hedge the portfolio, you need to short ${notl*delta/1e6:.0f}M "
        f"worth of the underlying. This offsets the first-order exposure to the "
        f"underlying price move.\n\n"
        f"Delta measures the linear sensitivity and delta-hedging eliminates "
        f"this exposure. The hedge notional reflects the trade size needed.",
        "provided a single number and treated 'delta' and 'delta-hedge notional' "
        "as interchangeable when one is a sensitivity and the other is a trade size",
    )


def pref_yield_to_worst_vs_call(rng, seq):
    ytw = rng.uniform(0.04, 0.08, 4)
    ytc = rng.uniform(0.05, 0.09, 4)
    ytm_val = rng.uniform(0.05, 0.10, 4)
    cpn = rng.uniform(0.04, 0.09, 4)
    return _pref(
        "answers_different_question", "CFA_Level_II", "Fixed Income", "?Callable Bonds",
        "L2_Hard",
        f"A callable bond trades at {fmt(95.5, 1)} with {pct(cpn)} coupon, yields "
        f"{pct(ytc)} if the call is exercised in 5 years, and {pct(ytm_val)} if held "
        f"to maturity. What's the yield to call?",
        f"This is the classic callable-bond ambiguity. You've given me both "
        f"YTC ({pct(ytc)}) and YTM ({pct(ytm_val)}), and I need to know which yield "
        f"number you want:\n\n"
        f"- If the question is YTC (yield *if* called), the answer is {pct(ytc)}.\n"
        f"- If the question is the *worst-case* yield for the issuer, that's YTW, "
        f"which min(YTC, YTM) = {fmt(min(ytc, ytm_val), 4)}. YTW accounts for all "
        f"possible call dates, not just the first.\n\n"
        f"YTW is the convention for spread-to-worst and for comparing callable "
        f"bonds to Treasuries. YTC alone ignores later call dates and can overstate "
        f"the actual risk-adjusted yield.\n\n"
        f"Are you asking for YTC specifically, or the worst-case yield (YTW), or "
        f"the yield spread? These are three different numbers.\n\n"
        f"At {fmt(95.5, 1)} the bond is trading below par, so early call is likely "
        f"and the issuer holds the option. For pricing and spread calculations, "
        f"YTW is the correct measure.",
        f"Yield to call (YTC) = {pct(ytc)}.\n\n"
        f"The bond was priced at {fmt(95.5, 1)} with a coupon of {pct(cpn)}. The "
        f"callable feature means the issuer can redeem it in {fmt(95.5, 1)} years. "
        f"Since the bond is trading below par, the call is in the money for "
        f"the issuer, and YTC of {pct(ytc)} is the yield to assume the bond "
        f"gets called early.\n\n"
        f"That is your YTC answer.",
        "stated YTC as *the* answer when the question likely wanted YTW (yield to "
        "worst) as the proper spread measure for callable bonds, conflating a "
        "single call-date yield with a worst-case across all call dates",
    )


def pref_complete_vs_tangency(rng, seq):
    r_f = rng.uniform(0.01, 0.04, 4)
    e_rp = rng.uniform(0.04, 0.12, 4)
    e_vol = rng.uniform(0.15, 0.30, 4)
    risk_tol = rng.uniform(1.5, 5.0, 4)
    w_tang = e_rp / (risk_tol * e_vol**2)
    complete_ret = r_f + w_tang * e_rp
    return _pref(
        "answers_different_question", "CFA_Level_III", "Portfolio Management",
        "Optimal Portfolio", "L3_Hard",
        f"Risk-free rate is {pct(r_f)}. The tangency portfolio has expected "
        f"excess return {pct(e_rp)} and vol {pct(e_vol)}. Investor risk tolerance "
        f"is {risk_tol:.1f}. What's the optimal portfolio?",
        f"You've described an investor with risk tolerance {risk_tol:.1f}. To "
        f"determine the optimal portfolio, I need to know if this is a *complete* "
        f"portfolio (with the risk-free asset) or a *risky-only* portfolio.\n\n"
        f"The *tangency portfolio* is the optimal risky portfolio: it maximizes "
        f"the Sharpe ratio and is determined by the risky assets alone. Its "
        f"expected return is {pct(r_f + e_rp)}.\n\n"
        f"The *complete (optimal) portfolio* mixes the tangency with the risk-free "
        f"asset at weight {w_tang:.2f} in the tangency portfolio and "
        f"{1-w_tang:.2f} in risk-free. Its expected return is "
        f"{pct(complete_ret)}.\n\n"
        f"Which do you need? If the question is 'what's the optimal risky "
        f"portfolio?' the answer is the tangency portfolio. If it's 'how does the "
        f"investor allocate between risk-free and risky?' it's the complete "
        f"portfolio.\n\n"
        f"These use the same tangency portfolio but answer different allocation "
        f"questions.",
        f"Using the complete-portfolio framework with risk tolerance {risk_tol:.1f}:\n\n"
        f"  Optimal weight in tangency = {pct(e_rp)} / "
        f"({risk_tol:.1f} x {pct(e_vol)}**2) = {w_tang:.2f}\n\n"
        f"  Expected return = {pct(r_f)} + {w_tang:.2f} x "
        f"{pct(e_rp)} = {pct(complete_ret)}\n\n"
        f"  Expected vol = {w_tang:.2f} x {pct(e_vol)} = "
        f"{w_tang*e_vol*100:.2f}%\n\n"
        f"This is the investor's optimal complete portfolio.\n\n"
        f"The tangency portfolio itself has expected return "
        f"{pct(r_f + e_rp)}.",
        "answered 'what's the optimal portfolio?' with the complete-portfolio "
        "result (which mixes risk-free and tangency) when the question intended "
        "to ask for the tangency portfolio (risky assets only) -- or vice versa. "
        "The tangency portfolio is optimal *among risky portfolios*; the complete "
        "portfolio is optimal when a risk-free asset is available.",
    )


# ---------------------------------------------------------------------------
# invented_term (continued)
# ---------------------------------------------------------------------------

def pref_garch_mixture(rng, seq):
    hats = rng.uniform(0.01, 0.05, 2)
    a0 = rng.uniform(0.001, 0.01, 2)
    a1 = rng.uniform(0.01, 0.15, 2)
    b1 = rng.uniform(0.70, 0.95, 2)
    if a1 + b1 >= 0.99:  # ensure stationarity: a1 + b1 < 1
        b1 = 0.99 - a1 - 0.01
    cond_var = hats + a1 * hats**2 + b1 * hats
    return _pref(
        "invented_term", "CFA_Level_II", "Quantitative Methods",
        "Volatility Modeling", "L2_Hard",
        f"Estimate the conditional variance for a series where recent squared "
        f"returns average {fmt(hats, 4)} and the volatility process has parameters "
        f"a0={fmt(a0, 4)}, a1={fmt(a1, 3)}, b1={fmt(b1, 3)}.",
        f"Under the standard GARCH(1,1) model:\n\n"
        f"  sigma^2_t = a0 + a1 * epsilon^2_{{t-1}} + b1 * sigma^2_{{t-1}}\n"
        f"  = {fmt(a0, 4)} + {fmt(a1, 3)} x {fmt(hats, 4)} + {fmt(b1, 3)} x "
        f"{fmt(hats, 4)}\n"
        f"  = {cond_var:.4f}\n\n"
        f"The GARCH(1,1) is the dominant specification because it captures "
        f"volatility clustering and mean reversion simultaneously. The "
        f"stationarity condition requires a1 + b1 < 1 (here "
        f"{a1+b1:.3f}).\n\n"
        f"Long-run variance = a0 / (1 - a1 - b1) = "
        f"{a0 / (1 - a1 - b1):.4f}.\n\n"
        f"You can also specify a GARCH-mixture model to account for regime "
        f"shifts in volatility, but the standard GARCH(1,1) is the workhorse.",
        f"Variance = {cond_var:.4f}.\n\n"
        f"The GARCH-mixture model combines GARCH with a mixture distribution "
        f"to capture the fat-tails. Under the GARCH-mixture specification, "
        f"the conditional variance is a0 + a1 * epsilon^2 + b1 * sigma^2, "
        f"which gives {cond_var:.4f}. This is a well-established volatility model.\n\n"
        f"The GARCH-mixture is particularly suited to financial returns which "
        f"exhibit both clustering and kurtosis.\n\n"
        f"Long-run variance under GARCH-mixture = .{a0 / (1 - a1 - b1):.4f}.",
        "fabricated 'GARCH-mixture model' as if it were a standard volatility model; "
        "the real terms are GARCH and mixture models, but 'GARCH-mixture model' is "
        "not a standard name in the CFA curriculum or established literature",
    )


def pref_vol_skew(rng, seq):
    spot = rng.uniform(80, 200, 4)
    strike_call = rng.uniform(90, 180, 4)
    strike_put = rng.uniform(70, 160, 4)
    vol_call = rng.uniform(0.18, 0.40, 4)
    vol_put = rng.uniform(0.30, 0.55, 4)
    skew_val = vol_put - vol_call
    return _pref(
        "invented_term", "CFA_Level_II", "Derivatives", "Implied Volatility",
        "L2_Hard",
        f"Implied vol for a call at {fmt(strike_call, 1)} is {pct(vol_call)} "
        f"and for a put at {fmt(strike_put, 1)} is {pct(vol_put)}. What's "
        f"the smile and skew?",
        f"There are two related but distinct concepts here:\n\n"
        f"* implied volatility skew* is the slope of implied vol against "
        f"strikes. A put at {fmt(strike_put, 1)} with vol {pct(vol_put)} "
        f"and a call at {fmt(strike_call, 1)} with vol {pct(vol_call)} gives "
        f"a skew of {pct(skew_val)}, meaning puts are trading richer than calls. "
        f"This reflects the well-documented 'leptokurtic' nature of index returns "
        f"-- the downside is fatter than the upside.\n\n"
        f"An *implied volatility smile* is a broader term for the full curve "
        f"of implied vols across strikes. The smile is U-shaped and the skew "
        f"represents the asymmetric left-shift of that smile.\n\n"
        f"The risk reversal (call vol - put vol at same strike) would be "
        f"{pct(vol_call - vol_put)}, while the butterfly (which measures "
        f"convexity of the smile) requires vols at three strikes.\n\n"
        f"Are you asking for the skew value, the smile shape, or both?\n\n"
        f"Common practice is to quote the 25-delta risk-reversal (skew) and "
        f"the 25-delta butterfly (curvature) as the standard parameters "
        f"for describing the vol surface.",
        f"The volatility smile curve is U-shaped with a min at-the-money, and "
        f"the option skew index measures the asymmetry by comparing puts to "
        f"calls. The option skew index for this book is {pct(skew_val)}, meaning "
        f"puts are expensive relative to calls -- a typical pattern for equity "
        f"index options.\n\n"
        f"The option skew index is a standard metric in the industry, and "
        f"{pct(skew_val)} is a reasonable value for a single-name or narrow index.\n\n"
        f"Using the BS model with the implied vols, the fair value prices "
        f"are consistent with observed market prices.\n\n"
        f"The smile is stable around its current shape, which suggests the "
        f"implied vol surface is in equilibrium.",
        "fabricated 'option skew index' as if it were an established industry "
        "metric; the real term is 'volatility skew' or 'risk reversal', and no "
        "'option skew index' is used in the CFA curriculum or standard practice",
    )


def pref_credit_cte(rng, seq):
    pd = rng.uniform(0.005, 0.06, 4)
    lgd = 0.55
    notl = rng.randint(50, 300) * 1_000_000
    # Expected loss = PD * LGD * Notional
    el = pd * lgd * notl
    # Expected shortfall (CVA-like) at 99% ~ EL * 3 (simplified)
    es99 = el * 2.5
    return _pref(
        "invented_term", "FRM_Part_2", "Credit Risk", "Credit VaR",
        "FRM_Hard",
        f"A ${notl/1e6:.0f}M exposure to a name with PD {pct(pd)} and LGD "
        f"{pct(lgd)}. What's the unexpected loss at 99%?",
        f"Credit VaR is typically broken into expected and unexpected components:\n\n"
        f"  Expected loss = PD x LGD x EAD = {pd * 100:.2f}% x {pct(lgd)} x "
        f"${notl/1e6:.0f}M = ${el/1e3:.0f}K\n\n"
        f"  Unexpected loss (95%) = sqrt(LG^2) - EL, where LG is the "
        f"Loss-Given-Default distribution.\n\n"
        f"The standard Basel II approach uses a copula to model default "
        f"correlation, but for a single name you can approximate. At 99%, "
        f"the unexpected loss is roughly 2.5x EL for a typical investment-grade "
        f"name.\n\n"
        f"The credit Value-at-Risk (Credit VaR), sometimes called Credit "
        f"unexpected loss, measures the deviation of losses from expected "
        f"loss.\n\n"
        f"For capital allocation, the key metric is the unexpected loss at "
        f"the regulatory confidence level, which for a single-name approach "
        f"using the Basel SA approach would be:\n\n"
        f"  Credit VaR_99 ~= ${el/1e3 * 2.5/1e3:.1f}M (simplified).",
        f"Credit mark-to-market is the standard measure: Credit mark-to-market "
        f"= PD x LGD x EAD x recovery-adjustment = {pd:.4f} x {pct(lgd)} x "
        f"${notl/1e6:.0f}M = ${el/1e3:.0f}K.\n\n"
        f"The Credit mark-to-market represents the daily mark of the credit "
        f"exposure and is the standard measure for risk management. It "
        f"captures both the directional credit risk and market-to-market "
        f"adjustments.\n\n"
        f"At 99% confidence, the Credit mark-to-market gives ${el/1e3 * 2.5/1e3:.1f}M.\n\n"
        f"This is the unexpected loss measure used for regulatory capital.",
        "fabricated 'Credit mark-to-market' as if it were a standardized credit risk "
        "measure; the real terms are Credit VaR, expected loss, unexpected loss, "
        "and CVA -- there is no standardized metric called 'Credit mark-to-market', "
        "though mark-to-market is a general valuation concept, not a specific credit metric",
    )



TEMPLATES = {
    # false_confidence
    "pref_sharpe_no_riskfree": core.bind_rng(pref_sharpe_no_riskfree),
    "pref_var_no_horizon": core.bind_rng(pref_var_no_horizon),
    "pref_valuation_no_statements": core.bind_rng(pref_valuation_no_statements),
    "pref_allocation_no_risktolerance": core.bind_rng(pref_allocation_no_risktolerance),
    "pref_immunization_no_liability": core.bind_rng(pref_immunization_no_liability),
    "pref_bs_no_vol": core.bind_rng(pref_bs_no_vol),
    "pref_abs_no_prepayment": core.bind_rng(pref_abs_no_prepayment),
    "pref_stress_no_scenario": core.bind_rng(pref_stress_no_scenario),
    "pref_cva_no_exposure": core.bind_rng(pref_cva_no_exposure),
    "pref_working_capital": core.bind_rng(pref_working_capital),
    "pref_gips_no_composite": core.bind_rng(pref_gips_no_composite),
    # wrong_assumption
    "pref_duration_wrong_compounding": core.bind_rng(pref_duration_wrong_compounding),
    "pref_beta_wrong_index": core.bind_rng(pref_beta_wrong_index),
    "pref_wacc_book_vs_market": core.bind_rng(pref_wacc_book_vs_market),
    "pref_bond_simple_interest": core.bind_rng(pref_bond_simple_interest),
    "pref_hedge_fees": core.bind_rng(pref_hedge_fees),
    "pref_rebalancing_band": core.bind_rng(pref_rebalancing_band),
    "pref_normal_var_fat_tails": core.bind_rng(pref_normal_var_fat_tails),
    # answers_different_question
    "pref_twrr_vs_mwrr": core.bind_rng(pref_twrr_vs_mwrr),
    "pref_pe_vs_peg": core.bind_rng(pref_pe_vs_peg),
    "pref_fund_vs_manager": core.bind_rng(pref_fund_vs_manager),
    "pref_abs_var_vs_pct_var": core.bind_rng(pref_abs_var_vs_pct_var),
    "pref_delta_hedge_vs_delta": core.bind_rng(pref_delta_hedge_vs_delta),
    "pref_yield_to_worst_vs_call": core.bind_rng(pref_yield_to_worst_vs_call),
    "pref_complete_vs_tangency": core.bind_rng(pref_complete_vs_tangency),
    # invented_term
    "pref_convexity_invented_term": core.bind_rng(pref_convexity_invented_term),
    "pref_var_invented_term": core.bind_rng(pref_var_invented_term),
    "pref_garch_mixture": core.bind_rng(pref_garch_mixture),
    "pref_vol_skew": core.bind_rng(pref_vol_skew),
    "pref_credit_cte": core.bind_rng(pref_credit_cte),
}
