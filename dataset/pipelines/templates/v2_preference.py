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

from pipelines.core import fmt, pct

# Modes whose `rejected` side deliberately contains a fabricated term. The
# terminology gate must whitelist those, or it blocks its own training data.
INVENTED_TERM_MODE = "invented_term"


def _pref(mode, program, topic, subtopic, difficulty, prompt, chosen, rejected,
          pitfall, question_type="Preference"):
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


TEMPLATES = {
    # false_confidence -- the highest-value mode
    "pref_sharpe_no_riskfree": pref_sharpe_no_riskfree,
    "pref_var_no_horizon": pref_var_no_horizon,
    "pref_valuation_no_statements": pref_valuation_no_statements,
    # wrong_assumption
    "pref_duration_wrong_compounding": pref_duration_wrong_compounding,
    "pref_beta_wrong_index": pref_beta_wrong_index,
    # answers_different_question
    "pref_twrr_vs_mwrr": pref_twrr_vs_mwrr,
    # invented_term
    "pref_convexity_invented_term": pref_convexity_invented_term,
    "pref_var_invented_term": pref_var_invented_term,
}
