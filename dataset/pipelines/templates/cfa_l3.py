"""
CFA Level III templates.
"""
import math
from pipelines.core import fmt, pct, render_trace


PROG = "CFA_Level_III"

def _assume(a):
    return "ASSUMPTIONS: " + "; ".join(a) + ".\n"

def perf_twrr(rng, seq):
    # two sub-periods
    r1 = rng.uniform(0.02, 0.06, 3)
    r2 = rng.uniform(0.03, 0.08, 3)
    tw = (1+r1)*(1+r2)-1
    q = (f"Sub-period returns: {pct(r1,1)} then {pct(r2,1)}. Compute time-weighted rate of return "
         f"(annual, no external cash flows).")
    tr = render_trace(rng, [
            "TWRR = (1+r1)(1+r2)−1",
            "no external flows",
        ], [
            (f"Step 1", f"(1+{r1:.3f})×(1+{r2:.3f}) = {1+r1:.4f}×{1+r2:.4f} = {(1+r1)*(1+r2):.4f}."),
            (f"Step 2", f"TWRR = {pct(tw)}."),
            (f"Step 3", f"Trap: averaging returns arithmetically ({pct((r1+r2)/2)}) ignores compounding within the measurement period."),
        ])
    flaw = {"answer": f"{pct((r1+r2)/2)}", "pitfall": "geometric vs arithmetic",
            "reasoning_trace": render_trace(rng, [
                    "arithmetic average",
                ], [
                    (f"Step 1", f"Averaging {pct(r1,1)} and {pct(r2,1)} = {pct((r1+r2)/2)}. TWRR compounds sub-period returns: (1+{r1:.3f})(1+{r2:.3f})−1 = {pct(tw)}."),
                ])}
    return {"meta":{"topic":"Performance Evaluation","subtopic":"TWRR","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["geometric vs arithmetic","compounding"]},
            "question":q, "answer":f"{pct(tw)}", "distractors":[f"{pct((r1+r2)/2)}", f"{pct(r1+r2)}", f"{pct(tw*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"r1":r1,"r2":r2}}

def perf_mwrr(rng, seq):
    v0 = rng.randint(100, 200) * 100000
    cf = rng.randint(20, 40) * 100000
    v1 = rng.randint(150, 250) * 100000
    mw = (v1 - v0 - cf)/(v0 + cf*0.5)  # approx annual
    flawed_mw = (v1 - v0)/v0  # ignores the contribution entirely
    q = (f"MWRR (approx): beginning value {fmt(v0)}, external contribution {fmt(cf)} at mid-year, "
         f"ending value {fmt(v1)}. Compute MWRR ≈ (V1 − V0 − CF)/(V0 + ½·CF).")
    tr = render_trace(rng, [
            "MWRR ≈ (V1 − V0 − CF)/(V0 + ½·CF)",
        ], [
            (f"Step 1", f"Net change = {fmt(v1)} − {fmt(v0)} − {fmt(cf)} = {fmt(v1-v0-cf)}."),
            (f"Step 2", f"Average invested = {fmt(v0)} + ½×{fmt(cf)} = {fmt(v0 + 0.5*cf)}."),
            (f"Step 3", f"MWRR = {fmt(v1-v0-cf)}/{fmt(v0+0.5*cf)} = {pct(mw)}."),
            (f"Step 4", f"Trap: weighting the contribution at full value (no timing) misstates the return."),
        ])
    flaw = {"answer": f"{pct(flawed_mw)}", "pitfall": "ignoring contribution timing",
            "reasoning_trace": render_trace(rng, [
                    "ignoring timing weight",
                ], [
                    (f"Step 1", f"Dividing net change by beginning value only ({fmt(v1-v0-cf)}/{fmt(v0)} = {pct((v1-v0-cf)/v0)}) ignores that the contribution was invested for only half the period; MWRR weights flows by time invested."),
                ])}
    return {"meta":{"topic":"Performance Evaluation","subtopic":"MWRR","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["cash flow timing","MWRR vs TWRR"]},
            "question":q, "answer":f"{pct(mw)}", "distractors":[f"{pct((v1-v0-cf)/v0)}", f"{pct(mw*2)}", f"{pct((v1-v0)/v0)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"v0":v0,"cf":cf,"v1":v1}}

def imm_duration_gap(rng, seq):
    dur_assets = rng.uniform(4, 7, 3)
    dur_liab = rng.uniform(3, 6, 3)
    gap = dur_assets - dur_liab
    q = (f"Immunization: asset duration {dur_assets:.2f}, liability duration {dur_liab:.2f}. "
         f"Compute the duration gap; is the portfolio immunized?")
    tr = render_trace(rng, [
            "immunization requires duration matching",
        ], [
            (f"Step 1", f"Duration gap = {dur_assets:.2f} − {dur_liab:.2f} = {gap:.2f}."),
            (f"Step 2", f"{'Gap ≠ 0 → NOT immunized; a parallel yield shift of Δy moves asset and liability values differently.' if abs(gap)>0.05 else 'Gap ≈ 0 → immunized against parallel shifts.'}"),
            (f"Step 3", f"Trap: matching maturity but not duration leaves rate risk; duration is the right metric."),
        ])
    flaw_ans = f"gap 0.50; not immunized" if abs(gap) < 0.05 else f"gap 0.00; immunized"
    flaw = {"answer": f"{flaw_ans}", "pitfall": "duration vs maturity matching",
            "reasoning_trace": render_trace(rng, [
                    "matching maturity",
                ], [
                    (f"Step 1", f"Matching nominal maturities does NOT immunize; duration {dur_assets:.2f} vs {dur_liab:.2f} reveals a gap of {gap:.2f}. Duration (not maturity) drives parallel-shift sensitivity."),
                ])}
    return {"meta":{"topic":"Fixed Income","subtopic":"Immunization","difficulty":"L3_Hard",
                    "question_type":"Calculation","pitfalls":["duration gap","duration vs maturity"]},
            "question":q, "answer":f"gap {gap:.2f}; not immunized" if abs(gap)>0.05 else f"gap {gap:.2f}; immunized",
            "distractors":[f"gap {fmt(gap*2)}", f"gap {fmt(-gap)}", f"gap {fmt(gap+1)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"dur_assets":dur_assets,"dur_liab":dur_liab}}

def risk_var(rng, seq):
    value = rng.randint(100, 500) * 1000000
    sig = rng.uniform(0.10, 0.25, 3)
    z = rng.choice([1.645, 1.96, 2.576])
    var = value*sig*z
    q = (f"Portfolio value {fmt(value)}, σ {pct(sig,1)}, one-day 95% VaR (z={z}). "
         f"Compute VaR = V·σ·z.")
    tr = render_trace(rng, [
            "VaR = V × σ × z",
            "normal distribution",
        ], [
            (f"Step 1", f"σ×z = {pct(sig,1)}×{z} = {sig*z:.4f}."),
            (f"Step 2", f"VaR = {fmt(value)}×{sig*z:.4f} = {fmt(var)}."),
            (f"Step 3", f"Interpretation: 95% confidence → 5% chance of losing more than {fmt(var)} in one day."),
            (f"Step 4", f"Trap: using z=1.645 (90%) instead of {z} changes the confidence level; VaR scales with z."),
        ])
    flaw_z = 1.96 if z == 1.645 else 1.645
    flaw = {"answer": f"{fmt(value*sig*flaw_z)}", "pitfall": "confidence level",
            "reasoning_trace": render_trace(rng, [
                    "90% instead of 95%",
                ], [
                    (f"Step 1", f"Using z=1.645 for a 90% VaR gives {fmt(value)}×{pct(sig,1)}×1.645 = {fmt(value*sig*1.645)}, a lower tail than the requested {fmt(value*sig*z)} at the {pct(1-0.05)} confidence level."),
                ])}
    return {"meta":{"topic":"Risk Management","subtopic":"VaR","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["confidence level","z-score"]},
            "question":q, "answer":f"{fmt(var)}", "distractors":[f"{fmt(value*sig*1.645)}", f"{fmt(value*sig)}", f"{fmt(var*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"value":value,"sig":sig,"z":z}}

def attr_carino(rng, seq):
    port_ret = rng.uniform(0.06, 0.12, 3)
    bench_ret = rng.uniform(0.04, 0.09, 3)
    alloc_eff = rng.uniform(0.01, 0.04, 3)
    select_eff = port_ret - bench_ret - alloc_eff
    q = (f"Portfolio return {pct(port_ret,1)}, benchmark {pct(bench_ret,1)}, allocation effect "
         f"{pct(alloc_eff,1)}. Compute selection effect = (R_p − R_b) − allocation.")
    tr = render_trace(rng, [
            "total effect = allocation + selection",
        ], [
            (f"Step 1", f"Total effect = {pct(port_ret,1)} − {pct(bench_ret,1)} = {pct(port_ret-bench_ret,1)}."),
            (f"Step 2", f"Selection = {pct(port_ret-bench_ret,1)} − {pct(alloc_eff,1)} = {pct(select_eff)}."),
            (f"Step 3", f"Trap: reporting total effect as selection conflates allocation and security selection."),
        ])
    flaw = {"answer": f"{pct(port_ret-bench_ret)}", "pitfall": "allocation vs selection",
            "reasoning_trace": render_trace(rng, [
                    "total effect as selection",
                ], [
                    (f"Step 1", f"Quoting total effect {pct(port_ret-bench_ret,1)} as pure selection ignores allocation effect {pct(alloc_eff,1)}; selection = {pct(port_ret-bench_ret,1)} − {pct(alloc_eff,1)} = {pct(select_eff)}."),
                ])}
    return {"meta":{"topic":"Performance Evaluation","subtopic":"Attribution","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["allocation vs selection","attribution decomposition"]},
            "question":q, "answer":f"{pct(select_eff)}", "distractors":[f"{pct(port_ret-bench_ret)}", f"{pct(alloc_eff)}", f"{pct(select_eff*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"port_ret":port_ret,"bench_ret":bench_ret,"alloc_eff":alloc_eff}}

def tax_taxable_equiv(rng, seq):
    tax = rng.choice([0.20, 0.25, 0.30, 0.35])
    taxfree = rng.uniform(0.03, 0.06, 3)
    equiv = taxfree/(1-tax)
    q = (f"Tax-free yield {pct(taxfree,1)}, investor tax rate {pct(tax,1)}. "
         f"Compute the taxable-equivalent yield = tax-free/(1−tax).")
    tr = render_trace(rng, [
            "taxable-equivalent yield = tax-free / (1−tax)",
        ], [
            (f"Step 1", f"After-tax retention = 1 − {tax:.2f} = {1-tax:.2f}."),
            (f"Step 2", f"Taxable-equivalent = {pct(taxfree,1)}/{1-tax:.2f} = {pct(equiv)}."),
            (f"Step 3", f"Trap: adding tax rate to the yield ({pct(taxfree+tax,1)}) misstates the equivalence."),
        ])
    flaw = {"answer": f"{pct(taxfree+tax)}", "pitfall": "tax equivalence formula",
            "reasoning_trace": render_trace(rng, [
                    "adding tax rate",
                ], [
                    (f"Step 1", f"Adding the tax rate to the yield: {pct(taxfree,1)} + {pct(tax,1)} = {pct(taxfree+tax,1)}. Correct: tax-free / (1−tax) = {pct(taxfree,1)}/{1-tax:.2f} = {pct(equiv)}."),
                ])}
    return {"meta":{"topic":"Private Wealth","subtopic":"Tax-Aware Investing","difficulty":"L3_Easy",
                    "question_type":"Calculation","pitfalls":["taxable-equivalent yield","after-tax retention"]},
            "question":q, "answer":f"{pct(equiv)}", "distractors":[f"{pct(taxfree+tax)}", f"{pct(taxfree)}", f"{pct(equiv*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"tax":tax,"taxfree":taxfree}}

def spend_rule(rng, seq):
    spend_rate = rng.choice([0.04, 0.05, 0.06])
    prev_spend = rng.randint(4, 6) * 100000
    aum = rng.randint(80, 120) * 1000000
    infl = rng.uniform(0.02, 0.03, 3)
    cur_spend = aum*spend_rate
    next_spend = min(prev_spend*(1+infl), max(cur_spend, prev_spend*(1+infl)))
    # spending rule: spend = max(rate*AUM, prior*(1+infl)) but capped
    spending = max(cur_spend, prev_spend*(1+infl))
    q = (f"Spending rule: spend rate {pct(spend_rate,1)}, AUM {fmt(aum)}, prior spending {fmt(prev_spend)}, "
         f"inflation {pct(infl,1)}. Compute the spending amount: max(rate×AUM, prior×(1+π)).")
    tr = render_trace(rng, [
            "spending = max(rate×AUM, prior×(1+π))",
        ], [
            (f"Step 1", f"rate×AUM = {pct(spend_rate,1)}×{fmt(aum)} = {fmt(cur_spend)}."),
            (f"Step 2", f"prior×(1+π) = {fmt(prev_spend)}×{1+infl:.3f} = {fmt(prev_spend*(1+infl))}."),
            (f"Step 3", f"Spending = max of the two = {fmt(spending)}."),
            (f"Step 4", f"Trap: using only the AUM-based amount ignores inflation smoothing of prior spending."),
        ])
    flaw = {"answer": f"{fmt(prev_spend*(1+infl))}", "pitfall": "inflation smoothing",
            "reasoning_trace": render_trace(rng, [
                    "AUM-only spending",
                ], [
                    (f"Step 1", f"Using only rate×AUM = {fmt(cur_spend)} drops the inflation-indexed floor {fmt(prev_spend*(1+infl))}; the rule takes the higher to smooth purchasing power."),
                ])}
    return {"meta":{"topic":"Institutional Portfolio","subtopic":"Spending Policy","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["spending rule","inflation smoothing"]},
            "question":q, "answer":f"{fmt(spending)}", "distractors":[f"{fmt(cur_spend)}", f"{fmt(prev_spend)}", f"{fmt(spending*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"spend_rate":spend_rate,"prev_spend":prev_spend,"aum":aum,"infl":infl}}


def alloc_rebalancing(rng, seq):
    target = rng.choice([0.40, 0.50, 0.60])
    drift = rng.uniform(0.0, 0.08, 3)
    current = target + drift
    q = (f"Target equity weight is {pct(target)}, current actual weight is {pct(current)}. "
         f"Should the portfolio be rebalanced (threshold 3%)?")
    need = abs(current - target) > 0.03
    tr = render_trace(rng, [
            "rebalance if |actual − target| exceeds threshold",
        ], [
            (f"Step 1", f"Deviation = {pct(current)} − {pct(target)} = {pct(current-target)}."),
            (f"Step 2", f"Threshold 3%: {'rebalance' if need else 'no rebalance'} ({pct(abs(current-target))} deviation)."),
            (f"Step 3", f"Trap: ignoring drift gives no rebalance."),
        ])
    ans = "rebalance" if need else "no rebalance"
    flaw = {"answer": "no rebalance" if need else "rebalance", "pitfall": "drift magnitude",
            "reasoning_trace": render_trace(rng, [
                    "deviation ignored",
                ], [
                    (f"Step 1", f"Actual weight {pct(current)} ≈ target {pct(target)}; {ans}."),
                ])}
    return {"meta": {"topic":"Asset Allocation","subtopic":"Rebalancing","difficulty":"L3_Medium",
                     "question_type":"Calculation","pitfalls":["drift magnitude"]},
            "question":q, "answer":ans,
            "distractors":["rebalance" if not need else "no rebalance", "sell equity", "buy equity"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"target":target,"drift":drift}}

def deriv_covered_call(rng, seq):
    s0 = rng.randint(45, 60) * 10
    k = s0 + rng.randint(2, 6) * 10
    prem = rng.randint(2, 4)
    st = s0 + rng.randint(-3, 3) * 10
    if st <= k:
        profit = prem + (st - s0)
    else:
        profit = prem + (k - s0)
    q = (f"Covered call: buy stock at {fmt(s0)}, sell call (strike {fmt(k)}) for premium "
         f"{prem}. Spot at expiry {fmt(st)}. Compute profit.")
    tr = render_trace(rng, [
            "covered call profit = premium + stock gain (capped at strike)",
        ], [
            (f"Step 1", f"Stock P&L = {fmt(st-s0)}; premium = {prem}."),
            (f"Step 2", f"Profit = {prem} + {fmt(min(st,k)-s0)} = {fmt(profit)}."),
            (f"Step 3", f"Trap: adding the call payoff instead of premium gives {fmt(prem + max(0, st-k))}."),
        ])
    flaw = {"answer": f"{fmt(prem + max(0, st-k))}", "pitfall": "call payoff vs premium",
            "reasoning_trace": render_trace(rng, [
                    "call payoff added instead of premium",
                ], [
                    (f"Step 1", f"Profit = premium + call payoff = {prem} + {fmt(max(0, st-k))} = {fmt(prem+max(0, st-k))}."),
                ])}
    return {"meta": {"topic":"Derivatives","subtopic":"Options Strategies","difficulty":"L3_Medium",
                     "question_type":"Calculation","pitfalls":["call payoff vs premium"]},
            "question":q, "answer":f"{fmt(profit)}",
            "distractors":[f"{fmt(prem + max(0, st-k))}", f"{fmt(prem)}", f"{fmt(st-s0)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"s0":s0,"k":k,"prem":prem,"st":st}}

def alloc_monte_carlo(rng, seq):
    n_scenarios = rng.randint(1000, 5000)
    mu = rng.uniform(0.05, 0.10, 3)
    sigma = rng.uniform(0.12, 0.22, 3)
    v0 = rng.randint(50, 200) * 1000000
    t = rng.choice([1, 3, 5])
    # approximate 5th percentile: V0 * exp((mu-0.5*sigma^2)*t - 1.645*sigma*sqrt(t))
    log_return = (mu - 0.5 * sigma**2) * t - 1.645 * sigma * math.sqrt(t)
    pctile_5 = v0 * math.exp(log_return)
    q = (f"Monte Carlo simulation: {n_scenarios:,} scenarios. Expected return  {pct(mu,1)}, "
          f"volatility {pct(sigma,1)}, horizon {t}y, current value {fmt(v0)}. "
          f"Approximate the 5th percentile terminal value using log-normal.")
    tr = render_trace(rng, [
            "terminal value ~ log-normal",
            "5th pctile: exp((μ−0.5σ²)t − 1.645σ√t)",
        ], [
            (f"Step 1", f"Drift-adjusted μᵈ = {pct(mu,1)} − ½×{pct(sigma,1)}² = {mu - 0.5*sigma**2:.4f}."),
            (f"Step 2", f"√t = {math.sqrt(t):.2f}; 1.645×σ×√t = {1.645*sigma*math.sqrt(t):.4f}."),
            (f"Step 3", f"ln(V_T/V₀) = ({mu - 0.5*sigma**2:.4f})×{t} − {1.645*sigma*math.sqrt(t):.4f} = {log_return:.4f}."),
            (f"Step 4", f"V_T(5th) = {fmt(v0)}×e^{log_return:.4f} ≈ {fmt(pctile_5)}."),
            (f"Step 5", f"Trap: using the mean return (μ) without the ½σ² adjustment overstates the percentile."),
        ])
    wrong_pctile = v0 * math.exp(mu * t - 1.645 * sigma * math.sqrt(t))  # forgot log correction
    flaw = {"answer": f"{fmt(wrong_pctile)}", "pitfall": "log-normal drift adjustment",
            "reasoning_trace": render_trace(rng, [
                    "forgot ½σ² term",
                ], [
                    (f"Step 1", f"ln(V_T/V₀) = μt − 1.645σ√t = {mu*t:.4f} − {1.645*sigma*math.sqrt(t):.4f} = {mu*t - 1.645*sigma*math.sqrt(t):.4f}. Correct: μᵈ = μ − ½σ² = {mu - 0.5*sigma**2:.4f}, giving V_T(5th) ≈ {fmt(pctile_5)}."),
                ])}
    return {"meta":{"topic":"Portfolio Management","subtopic":"Scenario Analysis","difficulty":"L3_Hard",
                    "question_type":"Calculation","pitfalls":["log-normal drift","Monte Carlo interpretation"]},
            "question":q, "answer":f"{fmt(pctile_5)}",
            "distractors":[f"{fmt(wrong_pctile)}", f"{fmt(v0*math.exp(mu*t))}", f"{fmt(v0)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"mu":mu,"sigma":sigma,"v0":v0,"t":t}}


def fip_yield_curve(rng, seq):
    fut_exp = rng.uniform(0.02, 0.04, 3)
    cur_spread = rng.uniform(0.01, 0.03, 3)
    fut_spread = cur_spread - rng.uniform(0.003, 0.008, 3)
    q = (f"Yield-curve strategy: expected 1y inflation shift = {pct(fut_exp,1)}, "
          f"current term premium = {pct(cur_spread,1)}, expected to narrow to {pct(fut_spread,1)}. "
          f"What duration positioning in a steepening environment?")
    tr = render_trace(rng, [
            "steepening = short end stable",
            "long end higher",
        ], [
            (f"Step 1", f"Inflation expectations ↑ by {pct(fut_exp,1)} → central bank likely raises short rates or holds stance."),
            (f"Step 2", f"Term premium narrowing from {pct(cur_spread,1)} to {pct(fut_spread,1)} makes long bonds *less* attractive than steepening."),
            (f"Step 3", f"Optimal: short duration or barbell; long-duration bonds underperform in steepening as long rates rise more."),
            (f"Step 4", f"Trap: extending duration in steepening assumes downward-sloping curve shift; steepening = long end outpaces short end rises."),
        ])
    flaw = {"answer": "extend duration for capital gains", "pitfall": "yield-curve steepening",
            "reasoning_trace": render_trace(rng, [
                    "steepening implies falling yields at all maturities",
                ], [
                    (f"Step 1", f"Assuming all yields fall → capital gains from duration extension. But steepening means long-end rises more than short end; extending duration is a loss in this scenario."),
                ])}
    return {"meta":{"topic":"Fixed Income","subtopic":"Yield Curve Strategies","difficulty":"L3_Hard",
                    "question_type":"Calculation","pitfalls":["steepening","term premium"]},
            "question":q, "answer":"short duration or barbell",
            "distractors":["extend duration", "go overweight long-end only", "maintain bullet"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"fut_exp":fut_exp,"cur_spread":cur_spread,"fut_spread":fut_spread}}


def pe_estate_planning(rng, seq):
    estate_val = rng.randint(300, 800) * 1000000
    estate_tax_rate = rng.choice([0.30, 0.35, 0.40])
    annual_gift = rng.randint(25, 200) * 100000  # annual gift tax exclusion
    yrs = rng.randint(5, 15)
    tax_free = annual_gift * yrs * 2  # married couple
    tax_saved = (estate_val - tax_free) * estate_tax_rate * 0.5  # rough half saved
    q = (f"Estate planning: gross estate {fmt(estate_val)}, tax rate {pct(estate_tax_rate,1)}. "
          f"Using annual exclusion gifts of {fmt(annual_gift)}/year for {yrs}y by a couple. "
          f"Estimate approximate tax savings.")
    tr = render_trace(rng, [
            "gift tax exclusion reduces taxable estate",
            "married couple ×2",
        ], [
            (f"Step 1", f"Annual total exclusion = {fmt(annual_gift)} ×2 = {fmt(annual_gift*2)}."),
            (f"Step 2", f"Total excluded over {yrs}y = {fmt(annual_gift*2)} × {yrs} = {fmt(tax_free)}."),
            (f"Step 3", f"Taxable estate after exclusion ≈ {fmt(estate_val - tax_free)}."),
            (f"Step 4", f"Tax without gifts = {fmt(estate_val)}×{pct(estate_tax_rate,1)} = {fmt(estate_val*estate_tax_rate)}."),
            (f"Step 5", f"Tax after gifts ≈ {fmt(estate_val-tax_free)}×{pct(estate_tax_rate,1)} = {fmt((estate_val-tax_free)*estate_tax_rate)}."),
            (f"Step 6", f"Savings ≈ {fmt(tax_saved)}."),
            (f"Step 7", f"Trap: ignoring step-up in basis; assets received as inheritance get step-up, which gifts avoid."),
        ])
    flaw_ans = f"tax savings {fmt(estate_val * estate_tax_rate * 0.25)}"  # underestimates
    flaw = {"answer": flaw_ans, "pitfall": "gift tax exclusion math",
            "reasoning_trace": render_trace(rng, [
                    "forgot spouse exclusion",
                ], [
                    (f"Step 1", f"Only counting one spouse: excluded = {fmt(annual_gift)}×{yrs} = {fmt(annual_gift*yrs)}, overstating taxable estate by {fmt(annual_gift*yrs)} and understating savings."),
                ])}
    return {"meta":{"topic":"Private Wealth","subtopic":"Estate Planning","difficulty":"L3_Hard",
                    "question_type":"Calculation","pitfalls":["gift exclusion","step-up in basis"]},
            "question":q, "answer":f"~{fmt(tax_saved)} saved",
            "distractors":[f"~{fmt(estate_val * estate_tax_rate * 0.25)} saved",
                           f"~{fmt(estate_val * estate_tax_rate * 0.10)} saved", "no savings possible"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"estate_val":estate_val,"estate_tax_rate":estate_tax_rate,
                                                            "annual_gift":annual_gift,"yrs":yrs}}


def inst_endowment(rng, seq):
    endowment = rng.randint(500, 2000) * 1000000
    spend_rate = rng.choice([0.04, 0.045, 0.05])
    total_ret = rng.uniform(0.06, 0.10, 3)
    inv_exp = rng.uniform(0.015, 0.025, 3)  # investment expense
    spending = endowment * spend_rate
    surpl = endowment * (total_ret - inv_exp) - spending
    q = (f"Endowment: fund value {fmt(endowment)}, spending policy {pct(spend_rate,1)}, "
          f"total return target {pct(total_ret,1)}, annual investment expense {pct(inv_exp,1)}. "
          f"Is the spending sustainable (positive surplus)?")
    tr = render_trace(rng, [
            "spending must come from spending of total return net of expenses",
        ], [
            (f"Step 1", f"After-expense return = {pct(total_ret,1)} − {pct(inv_exp,1)} = {pct(total_ret - inv_exp,1)}."),
            (f"Step 2", f"Spending = {pct(spend_rate,1)}×{fmt(endowment)} = {fmt(spending)}."),
            (f"Step 3", f"Income after expenses = {fmt(endowment)}×{pct(total_ret - inv_exp,1)} = {fmt(endowment*(total_ret-inv_exp))}."),
            (f"Step 4", f"Surplus = {fmt(endowment*(total_ret-inv_exp))} − {fmt(spending)} = {fmt(surpl)}."),
            (f"Step 5", f"{'Yes—spending is sustainable (positive surplus).' if surpl > 0 else 'No—spending exceeds total return net of expenses; corpus erodes.'}"),
            (f"Step 6", f"Trap: comparing spending rate to total return (ignoring expenses) is misleading."),
        ])
    flaw_ans = "sustainable" if surpl <= 0 else "not sustainable"
    flaw_ans2 = 'ignores expenses—compares {pct(spend_rate,1)} < {pct(total_ret,1)}'
    flaw = {"answer": flaw_ans, "pitfall": "ignoring investment expenses",
            "reasoning_trace": render_trace(rng, [
                    "comparing spend rate to gross return",
                ], [
                    (f"Step 1", f"Surplus = {fmt(surpl)}. Quoting '{flaw_ans}' ignores {pct(inv_exp,1)} expense drag, so spend rate vs gross return is misleading."),
                ])}
    return {"meta":{"topic":"Institutional Portfolio","subtopic":"Endowment Management","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["spending vs return","investment expenses"]},
            "question":q, "answer":"sustainable" if surpl > 0 else "not sustainable",
            "distractors":["sustainable", "not sustainable", "depends on reinvestment"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"endowment":endowment,"spend_rate":spend_rate,
                                                            "total_ret":total_ret,"inv_exp":inv_exp}}


def perf_benchmark(rng, seq):
    active_ret = rng.uniform(0.02, 0.08, 3)
    active_beta = rng.uniform(0.85, 1.15, 3)
    info_ratio = active_ret / rng.uniform(0.02, 0.06, 3) * active_beta
    q = (f"Active-return risk-budgeting: active return (tracking-error scaled)  {pct(active_ret,1)}, "
          f"active beta {active_beta:.2f}. Compute the active-risk-adjusted metric: "
          f"active return × active beta.")
    tr = render_trace(rng, [
            "risk-budgeting aims to maximize active risk-adjusted performance",
        ], [
            (f"Step 1", f"Active beta = {active_beta:.2f} indicates leverage of active bets."),
            (f"Step 2", f"Metric = {pct(active_ret,1)} × {active_beta:.2f} = {active_ret*active_beta:.4f}."),
            (f"Step 3", f"Trap: ignoring active beta treats all active return equally; beta > 1 amplifies active risk."),
            (f"Step 4", f"If active beta > 1, the manager uses active concentration on higher-returning styles; benchmark: same."),
        ])
    flaw_ans = f"{pct(active_ret,1)}"
    flaw = {"answer": flaw_ans, "pitfall": "ignoring active beta",
            "reasoning_trace": render_trace(rng, [
                    "equal weight to all active return",
                ], [
                    (f"Step 1", f"Quoting {pct(active_ret,1)} without multiplying by active beta {active_beta:.2f} ignores active risk amplification; correct metric = {pct(active_ret,1)} × {active_beta:.2f} = {active_ret*active_beta:.4f}."),
                ])}
    return {"meta":{"topic":"Portfolio Management","subtopic":"Active Risk Budgeting","difficulty":"L3_Hard",
                    "question_type":"Calculation","pitfalls":["active beta","risk-budgeting"]},
            "question":q, "answer":f"{active_ret*active_beta:.4f}",
            "distractors":[f"{pct(active_ret,1)}", f"{pct(active_ret*active_beta*2)}", f"{pct(active_ret/active_beta)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"active_ret":active_ret,"active_beta":active_beta}}


def risk_extreme(rng, seq):
    port_sig = rng.uniform(0.12, 0.25, 3)
    mean_ret = rng.uniform(0.06, 0.12, 3)
    # Normal distribution values at z=-1.645 (5th percentile):
    # phi(z) ≈ 0.1031, Phi(z) = 0.05
    z_5 = -1.645
    var5 = mean_ret + z_5 * port_sig
    # CVaR (expected shortfall) = μ - σ·φ(z)/Φ(z) where z is the VaR cutoff
    # Using phi(-1.645) ≈ 0.1031, Φ(-1.645) = 0.05
    phi_z = 0.1031
    cdf_z = 0.05
    cvar = mean_ret - port_sig * phi_z / cdf_z
    q = (f"Extreme-value / tail-risk: portfolio μ={pct(mean_ret,1)}, σ={pct(port_sig,1)}. "
          f"Compute the 5% VaR and CVaR (expected shortfall).")
    q2 = (f"VaR(5%) = μ + z·σ = {pct(mean_ret,1)} + ({z_5:.3f}) × {pct(port_sig,1)} = {var5:.4f}. "
          f"CVaR = μ − σ·φ(z)/Φ(z) ≈ {pct(mean_ret,1)} − {pct(port_sig,1)}×{phi_z:.4f}/{cdf_z:.2f} = {cvar:.4f}.")
    tr = render_trace(rng, [
            "VaR at 5%: z = −1.645",
            "φ(−1.645) ≈ 0.1031",
            "Φ(−1.645) = 0.05",
        ], [
            (f"Step 1", f"z(5%) = {z_5:.3f}."),
            (f"Step 2", f"VaR(5%) = {pct(mean_ret,1)} + ({z_5:.3f}) × {pct(port_sig,1)} = {var5:.4f}."),
            (f"Step 3", f"φ({z_5:.3f}) = {phi_z:.4f}; Φ({z_5:.3f}) = {cdf_z:.2f}."),
            (f"Step 4", f"CVaR ≈ {pct(mean_ret,1)} − {pct(port_sig,1)} × {phi_z:.4f}/{cdf_z:.2f} = {cvar:.4f}."),
            (f"Step 5", f"Interpretation: on the worst 5% of outcomes, average loss is {pct(cvar,1)}."),
            (f"Step 6", f"Trap: using just VaR (ignoring the tail beyond VaR) underestimates expected loss."),
        ])
    flaw_ans = f"CVaR ≈ {var5:.4f}"  # confusing VaR = CVaR
    flaw = {"answer": flaw_ans, "pitfall": "VaR vs CVaR",
            "reasoning_trace": render_trace(rng, [
                    "equating VaR with CVaR",
                ], [
                    (f"Step 1", f"Stating CVaR = {var5:.4f} (just VaR) ignores that conditional expectation below VaR is pulled further down. Correct CVaR ≈ {cvar:.4f} after the φ/Φ tail adjustment."),
                ])}
    return {"meta":{"topic":"Risk Management","subtopic":"Extreme Value / Tail Risk","difficulty":"L3_Hard",
                    "question_type":"Calculation","pitfalls":["tail expectation","conditional VaR","VaR vs CVaR"]},
            "question":q+q2, "answer":f"CVaR ≈ {cvar:.4f}",
            "distractors":[f"CVaR ≈ {var5:.4f}", f"CVaR ≈ {mean_ret:.4f}", f"CVaR ≈ {mean_ret + port_sig:.4f}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"port_sig":port_sig,"mean_ret":mean_ret,"var5":var5,"cvar":cvar}}


def alloc_black_litter(rng, seq):
    eq_prem = rng.uniform(0.04, 0.07, 3)
    risk_av = rng.choice([1.5, 2.0, 2.5])
    cov_mat_diag = rng.uniform(0.04, 0.10, 3)
    # implied equilibrium weight = expected return / (risk aversion * variance)
    eq_weight = eq_prem / (risk_av * cov_mat_diag)
    eq_weight = min(eq_weight, 1.0)  # cap
    q = (f"Black-Litterman: equity risk premium {pct(eq_prem,1)}, investor risk aversion {risk_av}, "
          f"market variance {pct(cov_mat_diag,1)}. Compute the implied equilibrium weight.")
    tr = render_trace(rng, [
            "We_q = E[R_q",
        ], [
            (f"Step 1", f"Implied weight = {pct(eq_prem,1)} / ({risk_av} × {pct(cov_mat_diag,1)}) = {eq_prem:.4f} / ({risk_av * cov_mat_diag:.4f}) = {eq_weight:.4f}."),
            (f"Step 2", f"Cap at 1.0 if exceeded."),
            (f"Step 3", f"Trap: ignoring risk aversion (setting δ=1) overweights the asset."),
            (f"Step 4", f"Trap: using standard deviation instead of variance gives {eq_prem / (risk_av * math.sqrt(cov_mat_diag)):.4f}."),
        ])
    flaw1 = eq_prem / math.sqrt(cov_mat_diag) * 0.5  # used std instead of var
    flaw = {"answer": f"{flaw1:.4f}", "pitfall": "std vs variance",
            "reasoning_trace": render_trace(rng, [
                    "used std instead of variance",
                ], [
                    (f"Step 1", f"Wrong: {pct(eq_prem,1)} / ({risk_av} × √{pct(cov_mat_diag,1)}) = {flaw1:.4f}. Correct: divide by δ × variance = {risk_av * cov_mat_diag:.4f}, giving {eq_weight:.4f}."),
                ])}
    return {"meta":{"topic":"Portfolio Management","subtopic":"Black-Litterman","difficulty":"L3_Hard",
                    "question_type":"Calculation","pitfalls":["variance vs std","risk aversion"]},
            "question":q, "answer":f"{eq_weight:.4f}",
            "distractors":[f"{flaw1:.4f}", f"{eq_prem:.4f}", f"{eq_prem * cov_mat_diag:.4f}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"eq_prem":eq_prem,"risk_av":risk_av,
                                                            "cov_mat_diag":cov_mat_diag}}


def pw_behavioral(rng, seq):
    initial = rng.randint(50, 200) * 100000
    gain = rng.randint(10, 40) * 100000
    final_val = initial + gain
    ref_pt = rng.randint(80, 120) * 100000
    val_fn_gain = 1.0  # risk seeking in gains
    val_fn_loss = 2.25  # loss aversion
    if final_val >= ref_pt:
        utils_gain = (final_val - ref_pt) ** val_fn_gain
    else:
        utils_loss = ref_pt - final_val
        utils_gain = -(val_fn_loss * utils_loss ** 2)
    q = (f"Behavioral portfolio theory: current value {fmt(final_val)}, reference point {fmt(ref_pt)}. "
          f"Using value function v(x) = x^α for gains (α=1) and v(x) = −λ·|x|^β for losses (λ=2.25). "
          f"Compute the subjective value.")
    tr = render_trace(rng, [
            "prospect theory reference-dependence",
        ], [
            (f"Step 1", f"Current value {fmt(final_val)} vs reference {fmt(ref_pt)}."),
            (f"Step 2", f"{'Outcome is above reference point: subjective value = (Final − Reference)^1 = ' if final_val >= ref_pt else f'Outcome is below reference point: subjective value = −2.25 × |{final_val} − {ref_pt}|^2.5 = '}"),
        ])
    if final_val >= ref_pt:
        subj_val = utils_gain
        trace_detail = f"Step 3. v({final_val - ref_pt}) = ({final_val - ref_pt})^1 = {utils_gain:.2f}.\n"
        trace_detail += f"Step 4. Trap: using endowment effect only (ignoring reference point) gives value {fmt(final_val)}.\n"
    else:
        subj_val = utils_gain
        trace_detail = f"Step 3. v({final_val - ref_pt}) = −2.25 × {abs(final_val - ref_pt)}^1 = {utils_gain:.2f}.\n"
        trace_detail += f"Step 4. Trap: using absolute value {fmt(final_val)} ignores loss aversion.\n"
    tr = render_trace(rng, [
            "prospect theory reference-dependence",
        ], [
            (f"Step 1", f"{fmt(final_val)} vs reference {fmt(ref_pt)}."),
            (f"Step 2", f"{'Gain' if final_val >= ref_pt else f'Loss of {fmt(abs(final_val-ref_pt))} from reference.'}{trace_detail}"),
            (f"Step 5", f"Subjective value = {utils_gain:.2f}."),
            (f"Step 6", f"Key: loss aversion (λ = 2.25) makes losses feel ~2.25× larger than comparable gains."),
        ])
    flaw_ans = f"{fmt(final_val)}"  # using absolute value
    flaw = {"answer": flaw_ans, "pitfall": "absolute value vs reference point",
            "reasoning_trace": render_trace(rng, [
                    "reference-independent valuation",
                ], [
                    (f"Step 1", f"Using absolute value {fmt(final_val)} ignores the reference point {fmt(ref_pt)} and gives subjective value {utils_gain:.2f}. Prospect theory shows evaluation reference-dependent."),
                ])}
    return {"meta":{"topic":"Private Wealth","subtopic":"Behavioral Finance","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["reference dependence","loss aversion"]},
            "question":q, "answer":f"{utils_gain:.2f}",
            "distractors":[f"{fmt(final_val)}", f"{fmt(initial)}", f"{fmt(gain)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"initial":initial,"gain":gain,"final_val":final_val,"ref_pt":ref_pt}}




# ---------------- Level III coverage extension ----------------
# Merged from cfa_l3_new.py. Fixed on the way in: four f-string syntax errors
# and one rng.random() call (core.RNG exposes the generator as .r).

def eq_factor_model(rng, seq):
    """Multi-factor model: market, size, value exposures with Jensen alpha."""
    rf = rng.uniform(0.02, 0.04, 3)
    mkt_prem = rng.uniform(0.05, 0.09, 3)
    size_prem = rng.uniform(0.01, 0.04, 3)
    val_prem = rng.uniform(0.01, 0.03, 3)
    s_mkt = rng.uniform(0.80, 1.20, 3)
    s_size = rng.uniform(-0.30, 0.30, 3)
    s_val = rng.uniform(-0.20, 0.40, 3)
    alpha = rng.uniform(-0.02, 0.02, 3)
    expected = rf + s_mkt*mkt_prem + s_size*size_prem + s_val*val_prem + alpha
    q = (f"Three-factor model: rf={pct(rf,1)}, MKP={pct(mkt_prem,1)}, "
         f"SML={pct(size_prem,1)}, VAL={pct(val_prem,1)}. Loadings: "
         f"bMKT={s_mkt:.2f}, bSML={s_size:.2f}, bVAL={s_val:.2f}. "
         f"Alpha={pct(alpha,1)}. Compute expected return.")
    tr = render_trace(rng, [
            "Expected = rf + beta_MKT*MKT + beta_SML*SML + beta_VAL*VAL + alpha",
        ], [
            (f"Step 1", f"MKP contrib = {s_mkt:.2f} x {pct(mkt_prem,1)} = {s_mkt*mkt_prem:.4f}."),
            (f"Step 2", f"SML contrib = {s_size:.2f} x {pct(size_prem,1)} = {s_size*size_prem:.4f}."),
            (f"Step 3", f"VAL contrib = {s_val:.2f} x {pct(val_prem,1)} = {s_val*val_prem:.4f}."),
            (f"Step 4", f"E[R] = {pct(rf)} + {s_mkt*mkt_prem:.4f} + {s_size*size_prem:.4f} + {s_val*val_prem:.4f} + {pct(alpha)} = {pct(expected)}."),
            (f"Step 5", f"Trap: ignoring tilts gives {pct(rf+mkt_prem)}."),
        ])
    wrong = rf + mkt_prem
    flaw = {"answer": f"{pct(wrong)}", "pitfall": "ignoring factor tilts",
            "reasoning_trace": render_trace(rng, [
                    "single-factor = rf + MKP",
                ], [
                    (f"", f"Quoting {pct(rf,1)}+{pct(mkt_prem,1)}={pct(wrong)} ignores beta_SML={s_size:.2f} and beta_VAL={s_val:.2f} tilts."),
                ])}
    return {"meta": {"topic": "Equity", "subtopic": "Factor Models",
                     "difficulty": "L3_Medium", "question_type": "Calculation",
                     "pitfalls": ["factor tilts", "Jensen alpha"]},
            "question": q, "answer": f"{pct(expected)}",
            "distractors": [f"{pct(wrong)}", f"{pct(rf+mkt_prem)}", f"{pct(rf)}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"s_mkt": s_mkt, "s_size": s_size, "s_val": s_val, "expected": expected,
                       "alpha": alpha, "rf": rf}}

def port_mean_variance(rng, seq):
    """Two-asset MVO: compute portfolio return, vol, Sharpe."""
    wf1 = rng.uniform(0.40, 0.65, 3)
    w2 = 1.0 - wf1
    mu1, mu2 = rng.uniform(0.08, 0.12, 3), rng.uniform(0.04, 0.07, 3)
    sig1, sig2 = rng.uniform(0.12, 0.20, 3), rng.uniform(0.08, 0.15, 3)
    rho = rng.uniform(-0.3, 0.5, 3)
    port_mu = wf1*mu1 + w2*mu2
    port_var = wf1**2*sig1**2 + w2**2*sig2**2 + 2*wf1*w2*rho*sig1*sig2
    port_sig = math.sqrt(max(port_var, 0))
    rf = rng.uniform(0.02, 0.04, 3)
    sharpe = (port_mu - rf)/port_sig
    no_corr_sig = math.sqrt(wf1**2*sig1**2 + w2**2*sig2**2)
    q = (f"Two-asset portfolio: A ({wf1:.2f}, mu={pct(mu1,1)}, sigma={pct(sig1,1)}), "
         f"B ({w2:.2f}, mu={pct(mu2,1)}, sigma={pct(sig2,1)}), rho={rho:.3f}. "
         f"RF={pct(rf)}. Compute portfolio mu, sigma, Sharpe.")
    tr = render_trace(rng, [
            "mu_p = w1*mu1 + w2*mu2",
        ], [
            (f"Step 1", f"mu_p = {pct(wf1)}x{pct(mu1,1)} + {pct(w2)}x{pct(mu2,1)} = {pct(port_mu)}."),
            (f"Step 2", f"Var = {wf1:.2f}^2x{sig1:.4f}^2 + {w2:.2f}^2x{sig2:.4f}^2 + 2x{wf1:.2f}x{w2:.2f}x{rho:.3f}x{sig1:.4f}x{sig2:.4f} = {port_var:.6f}."),
            (f"Step 3", f"sigma_p = {pct(port_sig)}."),
            (f"Step 4", f"Sharpe = ({pct(port_mu)}-{pct(rf)})/{pct(port_sig)} = {sharpe:.4f}."),
            (f"Step 5", f"Trap: rho=0 gives sigma={pct(no_corr_sig)}."),
        ])
    flaw = {"answer": f"sigma={pct(no_corr_sig)}, Sharpe={(port_mu-rf)/no_corr_sig:.4f}",
            "pitfall": "ignoring correlation",
            "reasoning_trace": render_trace(rng, [
                    "rho = 0",
                ], [
                    (f"", f"Dropping Cov term {2*wf1*w2*rho*sig1*sig2:.6f} gives wrong sigma. Correct: {pct(port_sig)} with rho={rho:.3f}."),
                ])}
    return {"meta": {"topic": "Portfolio Management", "subtopic": "Mean-Variance Optimization",
                     "difficulty": "L3_Medium", "question_type": "Calculation",
                     "pitfalls": ["covariance term", "sharpe"]},
            "question": q,
            "answer": f"mu={pct(port_mu)}, sigma={pct(port_sig)}, Sharpe={sharpe:.4f}",
            "distractors": [f"{pct(mu1)}, sigma={pct(sig1)}",
                          f"mu={pct(port_mu)}, sigma={pct(no_corr_sig)}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"wf1": wf1, "mu1": mu1, "mu2": mu2, "sig1": sig1, "sig2": sig2, "rho": rho}}

def fip_ldi(rng, seq):
    """Liability-Driven Investing: surplus duration gap analysis."""
    port_val = rng.randint(200, 500) * 1_000_000
    liab_val = rng.randint(180, 480) * 1_000_000
    dur_port = rng.uniform(5.0, 10.0, 3)
    dur_liab = rng.uniform(7.0, 12.0, 3)
    dy = rng.uniform(0.0025, 0.0075, 4)
    surplus = port_val - liab_val
    d_port = -dur_port * port_val * dy
    d_liab = -dur_liab * liab_val * dy
    d_surplus = d_port - d_liab
    q = (f"LDI: portfolio {fmt(port_val)}, D={dur_port:.2f}. "
         f"Liabilities {fmt(liab_val)}, D={dur_liab:.2f}. "
         f"Surplus={fmt(surplus)}. Yields +{dy*100:.2f}bps. Effect on surplus?")
    tr = render_trace(rng, [
            "Delta V = -D x V x dY",
        ], [
            (f"Step 1", f"dA = -{dur_port:.2f} x {fmt(port_val)} x {dy:.4f} = {fmt(d_port)}."),
            (f"Step 2", f"dL = -{dur_liab:.2f} x {fmt(liab_val)} x {dy:.4f} = {fmt(d_liab)}."),
            (f"Step 3", f"dSurplus = {fmt(d_port)} - {fmt(d_liab)} = {fmt(d_surplus)}."),
            (f"Step 4", f"{'Surplus ' + fmt(d_surplus)}."),
            (f"Step 5", f"Trap: duration matching != full immunization (convexity gap remains)."),
        ])
    flaw = {"answer": "surplus unchanged", "pitfall": "ignoring duration gap",
            "reasoning_trace": render_trace(rng, [
                    "D_A = D_L is enough",
                ], [
                    (f"", f"{dur_port:.2f} != {dur_liab:.2f}, gap={abs(dur_port-dur_liab):.2f}. dSurplus = {fmt(d_surplus)}."),
                ])}
    return {"meta": {"topic": "Fixed Income", "subtopic": "Liability-Driven Investing",
                     "difficulty": "L3_Hard", "question_type": "Calculation",
                     "pitfalls": ["duration gap", "convexity mismatch"]},
            "question": q,
            "answer": f"surplus changes by {fmt(d_surplus)}",
            "distractors": ["surplus unchanged",
                          f"surplus changes by {fmt(d_port)}",
                          f"surplus changes by {-d_surplus}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"port_val": port_val, "dur_port": dur_port, "dur_liab": dur_liab,
                       "d_surplus": d_surplus}}

def fip_interest_rate_risk(rng, seq):
    """Key-rate duration: parallel vs twist scenarios."""
    port_val = rng.randint(300, 700) * 1_000_000
    krd_2y = rng.uniform(0.15, 0.35, 3)
    krd_5y = rng.uniform(0.40, 0.65, 3)
    krd_10y = rng.uniform(0.25, 0.50, 3)
    dy_parallel = 0.005
    dv_all = -port_val * (krd_2y + krd_5y + krd_10y) * dy_parallel
    dv_twist = -port_val * (krd_2y*(-0.003) + krd_10y*(0.007))
    q = (f"Portfolio {fmt(port_val)}, KRD 2y={krd_2y:.2f}, 5y={krd_5y:.2f}, 10y={krd_10y:.2f}. "
         f"Delta-V (+50bp parallel vs twist 2y:-30bp/10y:+70bp).")
    tr = render_trace(rng, [
            "Delta V ~ -KRD x dY x V",
        ], [
            (f"Step 1", f"Parallel: sum KRD={krd_2y+krd_5y+krd_10y:.2f}. Delta-V = {-port_val:.0f} x {krd_2y+krd_5y+krd_10y:.2f} x 0.005 = {fmt(dv_all)}."),
            (f"Step 2", f"Twist: -V x ({krd_2y:.2f}x(-0.003) + {krd_10y:.2f}x0.007) = {fmt(dv_twist)}."),
            (f"Step 3", f"Trap: parallel DV01 != twist. KRD handles non-parallel shifts."),
        ])
    flaw = {"answer": f"twist={fmt(dv_all)}", "pitfall": "parallel vs twist",
            "reasoning_trace": render_trace(rng, [
                    "parallel DV01",
                ], [
                    (f"", f"Twist: {fmt(dv_twist)}, not {fmt(dv_all)}. KRD breaks yield curve into tenors."),
                ])}
    return {"meta": {"topic": "Fixed Income", "subtopic": "Interest Rate Risk",
                     "difficulty": "L3_Hard", "question_type": "Calculation",
                     "pitfalls": ["parallel vs twist", "key-rate duration"]},
            "question": q,
            "answer": f"parallel: {fmt(dv_all)}; twist: {fmt(dv_twist)}",
            "distractors": [f"parallel: {fmt(dv_all)}; twist: {fmt(dv_all)}",
                          f"parallel: {fmt(-dv_all)}; twist: {fmt(-dv_twist)}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"port_val": port_val, "dv_all": dv_all, "dv_twist": dv_twist,
                       "krd_2y": krd_2y, "krd_10y": krd_10y}}

def fip_bond_indexing(rng, seq):
    """Bond indexing: stratified sampling vs optimization."""
    n_bonds = rng.randint(200, 500)
    bench_dur = rng.uniform(5.5, 8.5, 3)
    bmv = rng.randint(5_000, 10_000) * 1_000_000
    strat = rng.r.random() > 0.5
    te = rng.uniform(0.30, 0.70, 2) if strat else rng.uniform(0.10, 0.30, 2)
    cost = rng.uniform(0.04, 0.11, 2) if strat else rng.uniform(0.02, 0.05, 2)
    n_baskets = rng.randint(8, 15) if strat else 0
    n_constraints = rng.randint(3, 7) if not strat else 0
    q = (f"Bond indexing: {n_bonds} bonds, bench D={bench_dur:.2f}, BMV={fmt(bmv)}. "
         f"{'Stratified' if strat else 'Optimization'} sampling. "
         f"TE={te:.2f}bps, cost={pct(cost)}. Evaluate strategy.")
    trap = (f" Trap: stratified simpler but TE~{te*1.8:.1f}bps vs optimization "
            f"{'factor-constrained' if not strat else ''}." if strat else
            f" Trap: optimization lower TE but risk model in {n_constraints} factors.")
    trap_clean = (f" Trap: stratified simpler but may have {te*1.8:.1f}bps TE; optimization "
                  f"{n_constraints}-factor model has risk." if strat else
                  f" Trap: cheaper optimization has {n_constraints} constraints to manage.") + trap[-20:]
    tr = render_trace(rng, [
            "index replication: minimize TE s.t. factor constraints",
        ], [
            (f"Step 1", f"{'Stratified: ' + str(n_baskets) + ' buckets' if strat else 'Optimization: ' + str(n_constraints) + ' risk factors'}.\\"),
            (f"Step 2", f"TE = {te:.2f}bps. Annual cost = {fmt(int(bmv * cost))}.\\"),
            (f"Step 3", f"{trap}"),
        ])
    flaw = {"answer": "full replication at lowest cost",
            "pitfall": "ignoring TE trade-off",
            "reasoning_trace": render_trace(rng, [
                    "cheapest option",
                ], [
                    (f"", f"Cheapest ignores {te:.2f}bps TE or {n_constraints} factors."),
                ])}
    return {"meta": {"topic": "Fixed Income", "subtopic": "Bond Indexing",
                     "difficulty": "L3_Medium", "question_type": "Calculation",
                     "pitfalls": ["tracking error", "strategy trade-offs"]},
            "question": q,
            "answer": f"{'Stratified' if strat else 'Optimization'} sampling, TE={te:.2f}bps",
            "distractors": ["stratified sampling", "full replication", "enhanced indexing"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"n_bonds": n_bonds, "bench_dur": bench_dur, "te": te, "cost": cost}}

def risk_risk_budgeting(rng, seq):
    """Risk budgeting: marginal and percentage contribution to risk."""
    port_val = rng.randint(200, 500) * 1_000_000
    n = 4
    raw_w = [rng.uniform(0.15, 0.40) for _ in range(n)]
    t = sum(raw_w)
    w = [wi/t for wi in raw_w]
    vols = [rng.uniform(0.08, 0.22, 3) for _ in range(n)]
    corr = [[0.0]*n for _ in range(n)]
    for i in range(n):
        corr[i][i] = 1.0
        for j in range(i+1, n):
            c = rng.uniform(0.1, 0.6, 3)
            corr[i][j] = c
            corr[j][i] = c
    port_var = sum(w[i]*w[j]*vols[i]*vols[j]*corr[i][j] for i in range(n) for j in range(n))
    port_sig = math.sqrt(max(port_var, 1e-8))
    # PCTR approx: w_i * beta_i / sigma_p * 100 where beta_i = sum_j w_j * sigma_ij / sigma_p
    beta_vec = []
    for i in range(n):
        b = sum(w[j]*vols[j]*corr[i][j] for j in range(n)) / port_sig
        beta_vec.append(b)
    mctr = [w[i]*beta_vec[i]*port_sig for i in range(n)]
    pctr = [mctr[i]/port_sig*100 for i in range(n)]
    # normalize pctr to sum to 100
    pctr_sum = sum(pctr)
    pctr = [p/pctr_sum*100 for p in pctr]
    top_i = max(range(n), key=lambda i: pctr[i])
    q = (f"Risk budget: 4-asset portfolio, value {fmt(port_val)}. "
         f"w=({', '.join(f'{ww:.2f}' for ww in w)}), "
         f"vol=({', '.join(f'{vv:.2f}' for vv in vols)}). "
         f"sigma_p={pct(port_sig)}. % risk contribution each?")
    top_pct = pctr[top_i]
    tr = render_trace(rng, [
            "PCTR additive: sum = 100%",
        ], [
            (f"Step 1", f"Beta_i = sum_j(w_j * sigma_ij) / sigma_p.\\"),
            (f"Step 2", f"MCTR_i = w_i * beta_i * sigma_p.\\"),
            (f"Step 3", f"PCTR_i = MCTR_i / sigma_p * 100.\\  A1: {pctr[0]:.1f}%, A2: {pctr[1]:.1f}%, A3: {pctr[2]:.1f}%, A4: {pctr[3]:.1f}%.\\Top: Asset {top_i+1} ({top_pct:.1f}%).\\"),
            (f"Step 4", f"Trap: equal weights != equal risk contribution.\\"),
        ])
    flaw = {"answer": "each equally (25% risk)",
            "pitfall": "weight = risk contribution",
            "reasoning_trace": render_trace(rng, [
                    "w_i = PCTR_i",
                ], [
                    (f"", f"Volatility and correlation differences mean weights don't equal risk shares. Asset {top_i+1} w={w[top_i]:.2f} but PCTR={top_pct:.1f}%."),
                ])}
    return {"meta": {"topic": "Portfolio Management", "subtopic": "Risk Budgeting",
                     "difficulty": "L3_Hard", "question_type": "Calculation",
                     "pitfalls": ["weight vs risk", "marginal contribution"]},
            "question": q,
            "answer": f"A{top_i+1}: {top_pct:.1f}% risk. sigma_p={pct(port_sig)}",
            "distractors": ["each equally (25% risk)",
                          "A1: highest due to weight",
                          "A4 highest due to volatility"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"weights": w, "vols": vols, "port_sig": port_sig, "pctr": pctr,
                       "top_i": top_i, "top_pct": top_pct}}

def pe_wealth_transfer(rng, seq):
    """Wealth transfer: estate tax with exemptions and gifting."""
    estate = rng.randint(500, 1500) * 1_000_000
    exemption = rng.randint(1200, 1400) * 1_000_000
    annual_excl = rng.randint(150_000, 200_000)
    yrs = rng.randint(10, 20)
    tax_rate = rng.choice([0.30, 0.35, 0.40])
    d_exemption = exemption * 2
    excl_total = annual_excl * 2 * yrs
    taxable = max(estate - d_exemption - excl_total, 0)
    no_plan_tax = estate * tax_rate
    with_plan_tax = taxable * tax_rate
    saved = no_plan_tax - with_plan_tax
    q = (f"Wealth transfer: gross estate {fmt(estate)}. Exemption {fmt(exemption)}/spouse. "
         f"Annual exclusion {fmt(annual_excl)}/person for {yrs}y. Tax rate {pct(tax_rate)}. "
         f"Compute estate tax savings from the strategy.")
    tr = render_trace(rng, [
            "exemption + annual exclusions reduce taxable estate",
        ], [
            (f"Step 1", f"Dual exemption = {fmt(d_exemption)}."),
            (f"Step 2", f"Exclusions = {fmt(annual_excl * 2)} x {yrs} = {fmt(excl_total)}."),
            (f"Step 3", f"Taxable = max({fmt(estate)}-{fmt(d_exemption)}-{fmt(excl_total)}, 0) = {fmt(taxable)}."),
            (f"Step 4", f"Tax no-plan: {fmt(no_plan_tax)}. Tax with-plan: {fmt(with_plan_tax)}."),
            (f"Step 5", f"Savings = {fmt(saved)}."),
            (f"Step 6", f"Trap: step-up in basis makes inheritance better than gifts for appreciated assets."),
        ])
    flaw = {"answer": f"savings={fmt(no_plan_tax*0.50)}",
            "pitfall": "ignoring exemptions",
            "reasoning_trace": render_trace(rng, [
                    "no exemption/gifting planning",
                ], [
                    (f"", f"Not accounting for {fmt(d_exemption)} + {fmt(excl_total)} tax sheltering."),
                ])}
    return {"meta": {"topic": "Private Wealth", "subtopic": "Wealth Transfer",
                     "difficulty": "L3_Hard", "question_type": "Calculation",
                     "pitfalls": ["exclusion math", "step-up in basis"]},
            "question": q, "answer": f"savings ~{fmt(saved)}",
            "distractors": [f"~{fmt(no_plan_tax*0.50)}",
                          f"tax={fmt(taxable)}", "no savings possible"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"estate": estate, "d_exemption": d_exemption, "excl_total": excl_total,
                       "saved": saved, "taxable": taxable}}


TEMPLATES = {
    "perf_twrr": perf_twrr, "perf_mwrr": perf_mwrr, "imm_duration_gap": imm_duration_gap,
    "risk_var": risk_var, "attr_carino": attr_carino, "tax_taxable_equiv": tax_taxable_equiv,
    "spend_rule": spend_rule,
    "alloc_rebalancing": alloc_rebalancing,
    "deriv_covered_call": deriv_covered_call,
    "alloc_monte_carlo": alloc_monte_carlo,
    "fip_yield_curve": fip_yield_curve,
    "pe_estate_planning": pe_estate_planning,
    "inst_endowment": inst_endowment,
    "perf_benchmark": perf_benchmark,
    "risk_extreme": risk_extreme,
    "alloc_black_litter": alloc_black_litter,
    "pw_behavioral": pw_behavioral,

    # Level III coverage extension
    "eq_factor_model": eq_factor_model,
    "port_mean_variance": port_mean_variance,
    "fip_ldi": fip_ldi,
    "fip_interest_rate_risk": fip_interest_rate_risk,
    "fip_bond_indexing": fip_bond_indexing,
    "risk_risk_budgeting": risk_risk_budgeting,
    "pe_wealth_transfer": pe_wealth_transfer,
}
