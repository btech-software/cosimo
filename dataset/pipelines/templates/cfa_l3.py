"""
CFA Level III templates.
"""
import math
from pipelines.core import fmt, pct

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
    tr = (_assume([f"TWRR = (1+r1)(1+r2)−1", f"no external flows"]) +
          f"Step 1. (1+{r1:.3f})×(1+{r2:.3f}) = {1+r1:.4f}×{1+r2:.4f} = {(1+r1)*(1+r2):.4f}.\n"
          f"Step 2. TWRR = {pct(tw)}.\n"
          f"Step 3. Trap: averaging returns arithmetically ({pct((r1+r2)/2)}) ignores compounding "
          f"within the measurement period.")
    flaw = {"answer": f"{pct((r1+r2)/2)}", "pitfall": "geometric vs arithmetic",
            "reasoning_trace": (_assume([f"arithmetic average"]) +
            f"Step 1. Averaging {pct(r1,1)} and {pct(r2,1)} = {pct((r1+r2)/2)}. "
            f"TWRR compounds sub-period returns: (1+{r1:.3f})(1+{r2:.3f})−1 = {pct(tw)}.")}
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
    tr = (_assume([f"MWRR ≈ (V1 − V0 − CF)/(V0 + ½·CF)"]) +
          f"Step 1. Net change = {fmt(v1)} − {fmt(v0)} − {fmt(cf)} = {fmt(v1-v0-cf)}.\n"
          f"Step 2. Average invested = {fmt(v0)} + ½×{fmt(cf)} = {fmt(v0 + 0.5*cf)}.\n"
          f"Step 3. MWRR = {fmt(v1-v0-cf)}/{fmt(v0+0.5*cf)} = {pct(mw)}.\n"
          f"Step 4. Trap: weighting the contribution at full value (no timing) misstates the return.")
    flaw = {"answer": f"{pct(flawed_mw)}", "pitfall": "ignoring contribution timing",
            "reasoning_trace": (_assume([f"ignoring timing weight"]) +
            f"Step 1. Dividing net change by beginning value only ({fmt(v1-v0-cf)}/{fmt(v0)} = "
            f"{pct((v1-v0-cf)/v0)}) ignores that the contribution was invested for only half the period; "
            f"MWRR weights flows by time invested.")}
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
    tr = (_assume([f"immunization requires duration matching"]) +
          f"Step 1. Duration gap = {dur_assets:.2f} − {dur_liab:.2f} = {gap:.2f}.\n"
          f"Step 2. {'Gap ≠ 0 → NOT immunized; a parallel yield shift of Δy moves asset and liability values differently.' if abs(gap)>0.05 else 'Gap ≈ 0 → immunized against parallel shifts.'}\n"
          f"Step 3. Trap: matching maturity but not duration leaves rate risk; duration is the right metric.")
    flaw_ans = f"gap 0.50; not immunized" if abs(gap) < 0.05 else f"gap 0.00; immunized"
    flaw = {"answer": f"{flaw_ans}", "pitfall": "duration vs maturity matching",
            "reasoning_trace": (_assume([f"matching maturity"]) +
            f"Step 1. Matching nominal maturities does NOT immunize; duration {dur_assets:.2f} vs "
            f"{dur_liab:.2f} reveals a gap of {gap:.2f}. Duration (not maturity) drives parallel-shift sensitivity.")}
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
    tr = (_assume([f"VaR = V × σ × z", f"normal distribution"]) +
          f"Step 1. σ×z = {pct(sig,1)}×{z} = {sig*z:.4f}.\n"
          f"Step 2. VaR = {fmt(value)}×{sig*z:.4f} = {fmt(var)}.\n"
          f"Step 3. Interpretation: 95% confidence → 5% chance of losing more than {fmt(var)} in one day.\n"
          f"Step 4. Trap: using z=1.645 (90%) instead of {z} changes the confidence level; VaR scales with z.")
    flaw_z = 1.96 if z == 1.645 else 1.645
    flaw = {"answer": f"{fmt(value*sig*flaw_z)}", "pitfall": "confidence level",
            "reasoning_trace": (_assume([f"90% instead of 95%"]) +
            f"Step 1. Using z=1.645 for a 90% VaR gives {fmt(value)}×{pct(sig,1)}×1.645 = {fmt(value*sig*1.645)}, "
            f"a lower tail than the requested {fmt(value*sig*z)} at the {pct(1-0.05)} confidence level.")}
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
    tr = (_assume([f"total effect = allocation + selection"]) +
          f"Step 1. Total effect = {pct(port_ret,1)} − {pct(bench_ret,1)} = {pct(port_ret-bench_ret,1)}.\n"
          f"Step 2. Selection = {pct(port_ret-bench_ret,1)} − {pct(alloc_eff,1)} = {pct(select_eff)}.\n"
          f"Step 3. Trap: reporting total effect as selection conflates allocation and security selection.")
    flaw = {"answer": f"{pct(port_ret-bench_ret)}", "pitfall": "allocation vs selection",
            "reasoning_trace": (_assume([f"total effect as selection"]) +
            f"Step 1. Quoting total effect {pct(port_ret-bench_ret,1)} as pure selection ignores "
            f"allocation effect {pct(alloc_eff,1)}; selection = {pct(port_ret-bench_ret,1)} − "
            f"{pct(alloc_eff,1)} = {pct(select_eff)}.")}
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
    tr = (_assume([f"taxable-equivalent yield = tax-free / (1−tax)"]) +
          f"Step 1. After-tax retention = 1 − {tax:.2f} = {1-tax:.2f}.\n"
          f"Step 2. Taxable-equivalent = {pct(taxfree,1)}/{1-tax:.2f} = {pct(equiv)}.\n"
          f"Step 3. Trap: adding tax rate to the yield ({pct(taxfree+tax,1)}) misstates the equivalence.")
    flaw = {"answer": f"{pct(taxfree+tax)}", "pitfall": "tax equivalence formula",
            "reasoning_trace": (_assume([f"adding tax rate"]) +
            f"Step 1. Adding the tax rate to the yield: {pct(taxfree,1)} + {pct(tax,1)} = {pct(taxfree+tax,1)}. "
            f"Correct: tax-free / (1−tax) = {pct(taxfree,1)}/{1-tax:.2f} = {pct(equiv)}.")}
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
    tr = (_assume([f"spending = max(rate×AUM, prior×(1+π))"]) +
          f"Step 1. rate×AUM = {pct(spend_rate,1)}×{fmt(aum)} = {fmt(cur_spend)}.\n"
          f"Step 2. prior×(1+π) = {fmt(prev_spend)}×{1+infl:.3f} = {fmt(prev_spend*(1+infl))}.\n"
          f"Step 3. Spending = max of the two = {fmt(spending)}.\n"
          f"Step 4. Trap: using only the AUM-based amount ignores inflation smoothing of prior spending.")
    flaw = {"answer": f"{fmt(prev_spend*(1+infl))}", "pitfall": "inflation smoothing",
            "reasoning_trace": (_assume([f"AUM-only spending"]) +
            f"Step 1. Using only rate×AUM = {fmt(cur_spend)} drops the inflation-indexed floor "
            f"{fmt(prev_spend*(1+infl))}; the rule takes the higher to smooth purchasing power.")}
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
    tr = (_assume([f"rebalance if |actual − target| exceeds threshold"]) +
          f"Step 1. Deviation = {pct(current)} − {pct(target)} = {pct(current-target)}.\n"
          f"Step 2. Threshold 3%: {'rebalance' if need else 'no rebalance'} ({pct(abs(current-target))} deviation).\n"
          f"Step 3. Trap: ignoring drift gives no rebalance.")
    ans = "rebalance" if need else "no rebalance"
    flaw = {"answer": "no rebalance" if need else "rebalance", "pitfall": "drift magnitude",
            "reasoning_trace": (_assume([f"deviation ignored"]) +
            f"Step 1. Actual weight {pct(current)} ≈ target {pct(target)}; {ans}.")}
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
    tr = (_assume([f"covered call profit = premium + stock gain (capped at strike)"]) +
          f"Step 1. Stock P&L = {fmt(st-s0)}; premium = {prem}.\n"
          f"Step 2. Profit = {prem} + {fmt(min(st,k)-s0)} = {fmt(profit)}.\n"
          f"Step 3. Trap: adding the call payoff instead of premium gives {fmt(prem + max(0, st-k))}.")
    flaw = {"answer": f"{fmt(prem + max(0, st-k))}", "pitfall": "call payoff vs premium",
            "reasoning_trace": (_assume([f"call payoff added instead of premium"]) +
            f"Step 1. Profit = premium + call payoff = {prem} + {fmt(max(0, st-k))} = {fmt(prem+max(0, st-k))}.")}
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
    tr = (_assume([f"terminal value ~ log-normal", f"5th pctile: exp((μ−0.5σ²)t − 1.645σ√t)"]) +
          f"Step 1. Drift-adjusted μᵈ = {pct(mu,1)} − ½×{pct(sigma,1)}² = {mu - 0.5*sigma**2:.4f}.\n"
          f"Step 2. √t = {math.sqrt(t):.2f}; 1.645×σ×√t = {1.645*sigma*math.sqrt(t):.4f}.\n"
          f"Step 3. ln(V_T/V₀) = ({mu - 0.5*sigma**2:.4f})×{t} − {1.645*sigma*math.sqrt(t):.4f} = {log_return:.4f}.\n"
          f"Step 4. V_T(5th) = {fmt(v0)}×e^{log_return:.4f} ≈ {fmt(pctile_5)}.\n"
          f"Step 5. Trap: using the mean return (μ) without the ½σ² adjustment overstates the percentile.")
    wrong_pctile = v0 * math.exp(mu * t - 1.645 * sigma * math.sqrt(t))  # forgot log correction
    flaw = {"answer": f"{fmt(wrong_pctile)}", "pitfall": "log-normal drift adjustment",
            "reasoning_trace": (_assume([f"forgot ½σ² term"]) +
            f"Step 1. ln(V_T/V₀) = μt − 1.645σ√t = {mu*t:.4f} − {1.645*sigma*math.sqrt(t):.4f} = {mu*t - 1.645*sigma*math.sqrt(t):.4f}. "
            f"Correct: μᵈ = μ − ½σ² = {mu - 0.5*sigma**2:.4f}, giving V_T(5th) ≈ {fmt(pctile_5)}.")}
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
    tr = (_assume([f"steepening = short end stable, long end higher"]) +
          f"Step 1. Inflation expectations ↑ by {pct(fut_exp,1)} → central bank likely raises short rates or holds stance.\n"
          f"Step 2. Term premium narrowing from {pct(cur_spread,1)} to {pct(fut_spread,1)} makes long bonds *less* attractive than steepening.\n"
          f"Step 3. Optimal: short duration or barbell; long-duration bonds underperform in steepening as long rates rise more.\n"
          f"Step 4. Trap: extending duration in steepening assumes downward-sloping curve shift; steepening = long end outpaces short end rises.")
    flaw = {"answer": "extend duration for capital gains", "pitfall": "yield-curve steepening",
            "reasoning_trace": (_assume([f"steepening implies falling yields at all maturities"]) +
            f"Step 1. Assuming all yields fall → capital gains from duration extension. But steepening means long-end rises more than short end; extending duration is a loss in this scenario.")}
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
    tr = (_assume([f"gift tax exclusion reduces taxable estate", f"married couple ×2"]) +
          f"Step 1. Annual total exclusion = {fmt(annual_gift)} ×2 = {fmt(annual_gift*2)}.\n"
          f"Step 2. Total excluded over {yrs}y = {fmt(annual_gift*2)} × {yrs} = {fmt(tax_free)}.\n"
          f"Step 3. Taxable estate after exclusion ≈ {fmt(estate_val - tax_free)}.\n"
          f"Step 4. Tax without gifts = {fmt(estate_val)}×{pct(estate_tax_rate,1)} = {fmt(estate_val*estate_tax_rate)}.\n"
          f"Step 5. Tax after gifts ≈ {fmt(estate_val-tax_free)}×{pct(estate_tax_rate,1)} = {fmt((estate_val-tax_free)*estate_tax_rate)}.\n"
          f"Step 6. Savings ≈ {fmt(tax_saved)}.\n"
          f"Step 7. Trap: ignoring step-up in basis; assets received as inheritance get step-up, which gifts avoid.")
    flaw_ans = f"tax savings {fmt(estate_val * estate_tax_rate * 0.25)}"  # underestimates
    flaw = {"answer": flaw_ans, "pitfall": "gift tax exclusion math",
            "reasoning_trace": (_assume([f"forgot spouse exclusion"]) +
            f"Step 1. Only counting one spouse: excluded = {fmt(annual_gift)}×{yrs} = {fmt(annual_gift*yrs)}, "
            f"overstating taxable estate by {fmt(annual_gift*yrs)} and understating savings.")}
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
    tr = (_assume([f"spending must come from spending of total return net of expenses"]) +
          f"Step 1. After-expense return = {pct(total_ret,1)} − {pct(inv_exp,1)} = {pct(total_ret - inv_exp,1)}.\n"
          f"Step 2. Spending = {pct(spend_rate,1)}×{fmt(endowment)} = {fmt(spending)}.\n"
          f"Step 3. Income after expenses = {fmt(endowment)}×{pct(total_ret - inv_exp,1)} = {fmt(endowment*(total_ret-inv_exp))}.\n"
          f"Step 4. Surplus = {fmt(endowment*(total_ret-inv_exp))} − {fmt(spending)} = {fmt(surpl)}.\n"
          f"Step 5. {'Yes—spending is sustainable (positive surplus).' if surpl > 0 else 'No—spending exceeds total return net of expenses; corpus erodes.'}\n"
          f"Step 6. Trap: comparing spending rate to total return (ignoring expenses) is misleading.")
    flaw_ans = "sustainable" if surpl <= 0 else "not sustainable"
    flaw_ans2 = 'ignores expenses—compares {pct(spend_rate,1)} < {pct(total_ret,1)}'
    flaw = {"answer": flaw_ans, "pitfall": "ignoring investment expenses",
            "reasoning_trace": (_assume([f"comparing spend rate to gross return"]) +
            f"Step 1. Surplus = {fmt(surpl)}. Quoting '{flaw_ans}' ignores {pct(inv_exp,1)} expense drag, "
            f"so spend rate vs gross return is misleading.")}
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
    tr = (_assume([f"risk-budgeting aims to maximize active risk-adjusted performance"]) +
          f"Step 1. Active beta = {active_beta:.2f} indicates leverage of active bets.\n"
          f"Step 2. Metric = {pct(active_ret,1)} × {active_beta:.2f} = {active_ret*active_beta:.4f}.\n"
          f"Step 3. Trap: ignoring active beta treats all active return equally; beta > 1 amplifies active risk.\n"
          f"Step 4. If active beta > 1, the manager uses active concentration on higher-returning styles; benchmark: same.")
    flaw_ans = f"{pct(active_ret,1)}"
    flaw = {"answer": flaw_ans, "pitfall": "ignoring active beta",
            "reasoning_trace": (_assume([f"equal weight to all active return"]) +
            f"Step 1. Quoting {pct(active_ret,1)} without multiplying by active beta {active_beta:.2f} ignores active risk amplification; "
            f"correct metric = {pct(active_ret,1)} × {active_beta:.2f} = {active_ret*active_beta:.4f}.")}
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
    tr = (_assume([f"VaR at 5%: z = −1.645", f"φ(−1.645) ≈ 0.1031, Φ(−1.645) = 0.05"]) +
          f"Step 1. z(5%) = {z_5:.3f}.\n"
          f"Step 2. VaR(5%) = {pct(mean_ret,1)} + ({z_5:.3f}) × {pct(port_sig,1)} = {var5:.4f}.\n"
          f"Step 3. φ({z_5:.3f}) = {phi_z:.4f}; Φ({z_5:.3f}) = {cdf_z:.2f}.\n"
          f"Step 4. CVaR ≈ {pct(mean_ret,1)} − {pct(port_sig,1)} × {phi_z:.4f}/{cdf_z:.2f} = {cvar:.4f}.\n"
          f"Step 5. Interpretation: on the worst 5% of outcomes, average loss is {pct(cvar,1)}.\n"
          f"Step 6. Trap: using just VaR (ignoring the tail beyond VaR) underestimates expected loss.")
    flaw_ans = f"CVaR ≈ {var5:.4f}"  # confusing VaR = CVaR
    flaw = {"answer": flaw_ans, "pitfall": "VaR vs CVaR",
            "reasoning_trace": (_assume([f"equating VaR with CVaR"]) +
            f"Step 1. Stating CVaR = {var5:.4f} (just VaR) ignores that conditional expectation below VaR is pulled further down. "
            f"Correct CVaR ≈ {cvar:.4f} after the φ/Φ tail adjustment.")}
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
    tr = (_assume([f"We_q = E[R_q] / (δ · Var[R_q])", f"δ = risk aversion"]) +
          f"Step 1. Implied weight = {pct(eq_prem,1)} / ({risk_av} × {pct(cov_mat_diag,1)}) = {eq_prem:.4f} / ({risk_av * cov_mat_diag:.4f}) = {eq_weight:.4f}.\n"
          f"Step 2. Cap at 1.0 if exceeded.\n"
          f"Step 3. Trap: ignoring risk aversion (setting δ=1) overweights the asset.\n"
          f"Step 4. Trap: using standard deviation instead of variance gives {eq_prem / (risk_av * math.sqrt(cov_mat_diag)):.4f}.")
    flaw1 = eq_prem / math.sqrt(cov_mat_diag) * 0.5  # used std instead of var
    flaw = {"answer": f"{flaw1:.4f}", "pitfall": "std vs variance",
            "reasoning_trace": (_assume([f"used std instead of variance"]) +
            f"Step 1. Wrong: {pct(eq_prem,1)} / ({risk_av} × √{pct(cov_mat_diag,1)}) = {flaw1:.4f}. "
            f"Correct: divide by δ × variance = {risk_av * cov_mat_diag:.4f}, giving {eq_weight:.4f}.")}
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
    tr = (_assume([f"prospect theory reference-dependence"]) +
          f"Step 1. Current value {fmt(final_val)} vs reference {fmt(ref_pt)}.\n"
          f"Step 2. {'Outcome is above reference point: subjective value = (Final − Reference)^1 = ' if final_val >= ref_pt else f'Outcome is below reference point: subjective value = −2.25 × |{final_val} − {ref_pt}|^2.5 = '}")
    if final_val >= ref_pt:
        subj_val = utils_gain
        trace_detail = f"Step 3. v({final_val - ref_pt}) = ({final_val - ref_pt})^1 = {utils_gain:.2f}.\n"
        trace_detail += f"Step 4. Trap: using endowment effect only (ignoring reference point) gives value {fmt(final_val)}.\n"
    else:
        subj_val = utils_gain
        trace_detail = f"Step 3. v({final_val - ref_pt}) = −2.25 × {abs(final_val - ref_pt)}^1 = {utils_gain:.2f}.\n"
        trace_detail += f"Step 4. Trap: using absolute value {fmt(final_val)} ignores loss aversion.\n"
    trace_full = (_assume([f"prospect theory reference-dependence"]) +
                  f"Step 1. {fmt(final_val)} vs reference {fmt(ref_pt)}.\n"
                  f"Step 2. {'Gain' if final_val >= ref_pt else f'Loss of {fmt(abs(final_val-ref_pt))} from reference.'}\n"
                  f"{trace_detail}"
                  f"Step 5. Subjective value = {utils_gain:.2f}.\n"
                  f"Step 6. Key: loss aversion (λ = 2.25) makes losses feel ~2.25× larger than comparable gains.")
    flaw_ans = f"{fmt(final_val)}"  # using absolute value
    flaw = {"answer": flaw_ans, "pitfall": "absolute value vs reference point",
            "reasoning_trace": (_assume([f"reference-independent valuation"]) +
            f"Step 1. Using absolute value {fmt(final_val)} ignores the reference point {fmt(ref_pt)} and gives subjective value {utils_gain:.2f}. Prospect theory shows evaluation reference-dependent.")}
    return {"meta":{"topic":"Private Wealth","subtopic":"Behavioral Finance","difficulty":"L3_Medium",
                    "question_type":"Calculation","pitfalls":["reference dependence","loss aversion"]},
            "question":q, "answer":f"{utils_gain:.2f}",
            "distractors":[f"{fmt(final_val)}", f"{fmt(initial)}", f"{fmt(gain)}"],
            "reasoning_trace":trace_full, "flawed":flaw,
            "params":{"initial":initial,"gain":gain,"final_val":final_val,"ref_pt":ref_pt}}


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
}
