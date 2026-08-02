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

TEMPLATES = {
    "perf_twrr": perf_twrr, "perf_mwrr": perf_mwrr, "imm_duration_gap": imm_duration_gap,
    "risk_var": risk_var, "attr_carino": attr_carino, "tax_taxable_equiv": tax_taxable_equiv,
    "spend_rule": spend_rule,
    "alloc_rebalancing": alloc_rebalancing,
    "deriv_covered_call": deriv_covered_call,
}
