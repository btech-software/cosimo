"""
CFA Level II templates. Same contract as cfa_l1.
"""
import math
from pipelines.core import fmt, pct
from pipelines.templates.wrappers import wrap_mcq

PROG = "CFA_Level_II"

def _assume(a):
    return "ASSUMPTIONS: " + "; ".join(a) + ".\n"

def eq_fcff_dcf(rng, seq):
    fcff = rng.randint(8, 20) * 1000000
    wacc = rng.uniform(0.09, 0.12, 3)
    g = rng.uniform(0.02, 0.05, 3)
    if g >= wacc:
        g = wacc - rng.uniform(0.02, 0.04, 3)
    v = fcff*(1+g)/(wacc-g)
    q = (f"FCFF = ${fmt(fcff)}, WACC = {pct(wacc,1)}, long-run growth g = {pct(g,1)}. "
         f"Compute firm value via single-stage FCFF DCF: V = FCFF1/(WACC−g).")
    tr = (_assume([f"FCFF1 = FCFF0(1+g)", f"V = FCFF1/(WACC−g)", f"g < WACC"]) +
          f"Step 1. FCFF1 = {fmt(fcff)}×{1+g:.3f} = {fmt(fcff*(1+g))}.\n"
          f"Step 2. WACC−g = {wacc:.3f} − {g:.3f} = {wacc-g:.4f}.\n"
          f"Step 3. V = FCFF1/(WACC−g) = {fmt(fcff*(1+g))}/{wacc-g:.4f} = {fmt(v)}.\n"
          f"Step 4. Trap: discounting FCFF0 (no growth) understates value; use next-period FCFF1.")
    flaw = {"answer": f"{fmt(fcff/(wacc-g))}", "pitfall": "FCFF0 vs FCFF1",
            "reasoning_trace": (_assume([f"using FCFF0"]) +
            f"Step 1. Plugging FCFF0 = {fmt(fcff)} directly into {fmt(fcff)}/{wacc-g:.4f} = {fmt(fcff/(wacc-g))} "
            f"ignores the first growth step; single-stage DCF requires FCFF1 = FCFF0(1+g).")}
    return {"meta":{"topic":"Equity Valuation","subtopic":"FCFF Valuation","difficulty":"L2_Medium",
                    "question_type":"Calculation","pitfalls":["FCFF0 vs FCFF1","g<WACC"]},
            "question":q, "answer":f"{fmt(v)}", "distractors":[f"{fmt(fcff/(wacc-g))}", f"{fmt(v*1.5)}", f"{fmt(fcff/g)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"fcff":fcff,"wacc":wacc,"g":g}}

def eq_fcfe(rng, seq):
    ni = rng.randint(10, 30) * 1000000
    dep = rng.randint(2, 8) * 1000000
    capex = rng.randint(3, 9) * 1000000
    wc = rng.randint(1, 4) * 1000000
    net_borrow = rng.randint(1, 3) * 1000000
    fcfe = ni + dep - capex - wc + net_borrow
    q = (f"NI {fmt(ni)}, depreciation +{fmt(dep)}, capex −{fmt(capex)}, ΔNWC −{fmt(wc)}, "
         f"net borrowing +{fmt(net_borrow)}. Compute FCFE.")
    tr = (_assume([f"FCFE = NI + Dep − Capex − ΔNWC + Net Borrowing"]) +
          f"Step 1. NI + Dep = {fmt(ni)} + {fmt(dep)} = {fmt(ni+dep)}.\n"
          f"Step 2. Minus capex {fmt(capex)} and ΔNWC {fmt(wc)}: {fmt(ni+dep-capex-wc)}.\n"
          f"Step 3. Plus net borrowing {fmt(net_borrow)}: FCFE = {fmt(fcfe)}.\n"
          f"Step 4. Trap: FCFF excludes net borrowing; forgetting it understates FCFE for levered firms.")
    flaw = {"answer": f"{fmt(ni+dep-capex-wc)}", "pitfall": "FCFF vs FCFE",
            "reasoning_trace": (_assume([f"dropping net borrowing"]) +
            f"Step 1. NI+Dep−Capex−ΔNWC = {fmt(ni+dep-capex-wc)} is FCFE before financing flows. "
            f"FCFE adds net borrowing {fmt(net_borrow)}: {fmt(fcfe)}. FCFF excludes this debt flow.")}
    return {"meta":{"topic":"Equity Valuation","subtopic":"FCFE","difficulty":"L2_Easy",
                    "question_type":"Calculation","pitfalls":["FCFF vs FCFE","net borrowing"]},
            "question":q, "answer":f"{fmt(fcfe)}", "distractors":[f"{fmt(ni+dep-capex-wc)}", f"{fmt(ni+dep)}", f"{fmt(fcfe*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"ni":ni,"dep":dep,"capex":capex,"wc":wc,"net_borrow":net_borrow}}

def eq_residual_income(rng, seq):
    bv0 = rng.randint(20, 40) * 1000000
    ni = rng.randint(4, 8) * 1000000
    re = rng.uniform(0.10, 0.12, 3)
    ri = ni - re*bv0
    v = bv0 + ri/(1+re)
    q = (f"BV0 = {fmt(bv0)}, NI = {fmt(ni)}, required return on equity = {pct(re,1)}. "
         f"Compute residual income (RI = NI − r_e·BV0) and 1-period value V = BV0 + RI/(1+r_e).")
    tr = (_assume([f"RI = NI − r_e·BV0"]) +
          f"Step 1. r_e·BV0 = {re:.2f}×{fmt(bv0)} = {fmt(re*bv0)}.\n"
          f"Step 2. RI = {fmt(ni)} − {fmt(re*bv0)} = {fmt(ri)}.\n"
          f"Step 3. V = BV0 + RI/(1+r_e) = {fmt(bv0)} + {fmt(ri)}/{1+re:.3f} = {fmt(v)}.\n"
          f"Step 4. Trap: using total earnings (not residual) in the PV overstates value.")
    flaw = {"answer": f"{fmt(bv0 + ni/(1+re))}", "pitfall": "residual vs total earnings",
            "reasoning_trace": (_assume([f"discounting total NI"]) +
            f"Step 1. Adding total NI {fmt(ni)} (instead of residual {fmt(ri)}) to BV0: "
            f"{fmt(bv0)} + {fmt(ni)}/{1+re:.3f} = {fmt(bv0 + ni/(1+re))}. "
            f"RI valuation capitalizes only value CREATED above the required return on book value.")}
    return {"meta":{"topic":"Equity Valuation","subtopic":"Residual Income","difficulty":"L2_Medium",
                    "question_type":"Calculation","pitfalls":["residual vs total earnings","RI definition"]},
            "question":q, "answer":f"RI {fmt(ri)}; V {fmt(v)}", "distractors":[f"{fmt(bv0 + ni/(1+re))}", f"{fmt(ri)}", f"{fmt(v*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"bv0":bv0,"ni":ni,"re":re}}

def fi_spot_forward(rng, seq):
    s1 = rng.uniform(0.03, 0.05, 3)
    s2 = rng.uniform(0.04, 0.07, 3)
    fwd = ((1+s2)**2/(1+s1)) - 1
    # degenerate s1==s2 makes avg collide; use clearly-wrong fallback
    flawed = s2 if (s2-s1) >= 0.015 else s2 + 0.10
    q = (f"1-yr spot {pct(s1,1)}, 2-yr spot {pct(s2,1)}. Compute the 1-yr forward rate "
         f"1-yr ahead: (1+f) = (1+s2)²/(1+s1).")
    tr = (_assume([f"no-arbitrage: (1+f1,2) = (1+s2)²/(1+s1)"]) +
          f"Step 1. (1+s2)² = (1+{s2:.3f})² = {(1+s2)**2:.5f}.\n"
          f"Step 2. Divide by (1+s1) = {1+s1:.3f}: (1+f) = {(1+s2)**2/(1+s1):.5f}.\n"
          f"Step 3. f = {pct(fwd)}.\n"
          f"Step 4. Trap: averaging the two spot rates would give {pct((s1+s2)/2)}, violating no-arbitrage.")
    flaw = {"answer": f"{pct(flawed)}", "pitfall": "arbitrage vs averaging",
            "reasoning_trace": (_assume([f"simple average"]) +
            f"Step 1. Averaging spots {pct(s1,1)} and {pct(s2,1)} = {pct((s1+s2)/2)} ignores the "
            f"compounding structure; the forward must satisfy (1+f)(1+s1) = (1+s2)² = {pct(fwd)}.")}
    return {"meta":{"topic":"Fixed Income","subtopic":"Forward Rates","difficulty":"L2_Medium",
                    "question_type":"Calculation","pitfalls":["forward derivation","no-arbitrage"]},
            "question":q, "answer":f"{pct(fwd)}", "distractors":[f"{pct((s1+s2)/2)}", f"{pct(s2-s1)}", f"{pct(fwd*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"s1":s1,"s2":s2}}

def fi_bond_price(rng, seq):
    coupon = rng.randint(3, 6) * 10
    y = rng.uniform(0.04, 0.08, 3)
    n = rng.randint(3, 10)
    fv = 100
    price = sum(coupon/(1+y)**t for t in range(1, n+1)) + fv/(1+y)**n
    q = (f"An {n}-yr bond pays an annual coupon of ${coupon:.0f} (par 100), yield {pct(y,1)}. "
         f"Compute its price (sum of PVs).")
    tr = (_assume([f"price = Σ coupon/(1+y)^t + par/(1+y)^n"]) +
          f"Step 1. PV coupons: {fmt(price - fv/(1+y)**n)}.\n"
          f"Step 2. PV par = 100/(1+{y:.2f})^{n} = {fmt(fv/(1+y)**n)}.\n"
          f"Step 3. Price = {fmt(price)}.\n"
          f"Step 4. Trap: discounting coupons with the wrong exponent (no maturity scaling) misprices the bond.")
    flaw = {"answer": f"{fmt(coupon*n + fv)}", "pitfall": "discounting vs summing cash flows",
            "reasoning_trace": (_assume([f"undiscounted cash flows"]) +
            f"Step 1. Summing cash flows undiscounted: {coupon}×{n} + 100 = {fmt(coupon*n+fv)}. "
            f"Each cash flow must be PV-discounted by (1+y)^t; the true price is {fmt(price)}.")}
    return {"meta":{"topic":"Fixed Income","subtopic":"Bond Pricing","difficulty":"L2_Hard",
                    "question_type":"Calculation","pitfalls":["PV of each cash flow","maturity scaling"]},
            "question":q, "answer":f"{fmt(price)}", "distractors":[f"{fmt(coupon*n+fv)}", f"{fmt(price*1.1)}", f"{fmt(fv)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"coupon":coupon,"y":y,"n":n}}

def deriv_bsm_call(rng, seq):
    s = rng.randint(45, 60)
    k = rng.randint(45, 55)
    t = rng.uniform(0.5, 2.0, 3)
    r = rng.choice([0.04, 0.05, 0.06])
    sig = rng.uniform(0.20, 0.35, 3)
    import math as _m
    from math import erf, sqrt
    def nd(x): return 0.5*(1 + erf(x/sqrt(2)))
    d1 = (_m.log(s/k) + (r + sig**2/2)*t)/(sig*_m.sqrt(t))
    d2 = d1 - sig*_m.sqrt(t)
    call = s*nd(d1) - k*_m.exp(-r*t)*nd(d2)
    q = (f"BSM call: S={s}, K={k}, T={t:.1f} yrs, r={pct(r,1)}, σ={pct(sig,1)}. Compute the call price.")
    tr = (_assume([f"d1 = [ln(S/K)+(r+σ²/2)T]/σ√T", f"C = S·N(d1) − K·e^(−rT)·N(d2)"]) +
          f"Step 1. d1 = [ln({s}/{k}) + ({r:.3f}+{sig**2/2:.4f})×{t:.2f}]/({sig:.2f}×√{t:.2f}) = {d1:.3f}.\n"
          f"Step 2. d2 = d1 − σ√T = {d2:.3f}.\n"
          f"Step 3. C = {s}×N({d1:.3f}) − {k}×e^(−{r:.3f}×{t:.2f})×N({d2:.3f}) = {fmt(call)}.\n"
          f"Step 4. Trap: using intrinsic value max(S−K,0)={max(s-k,0)} ignores time value and vol; BSM is higher.")
    flaw = {"answer": f"{fmt(max(s-k,0))}", "pitfall": "intrinsic vs time value",
            "reasoning_trace": (_assume([f"intrinsic value only"]) +
            f"Step 1. Reporting intrinsic value max({s}−{k},0) = {max(s-k,0)} drops time value from "
            f"volatility {pct(sig,1)} and horizon {t:.1f} yrs; BSM price is {fmt(call)}.")}
    return {"meta":{"topic":"Derivatives","subtopic":"BSM","difficulty":"L2_Hard",
                    "question_type":"Calculation","pitfalls":["time value","BSM inputs"]},
            "question":q, "answer":f"{fmt(call)}", "distractors":[f"{fmt(max(s-k,0))}", f"{fmt(call*2)}", f"{fmt(call*0.5)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"s":s,"k":k,"t":t,"r":r,"sig":sig}}

def deriv_delta_gamma(rng, seq):
    delta = rng.uniform(0.3, 0.7, 3)
    gamma = rng.uniform(0.02, 0.06, 3)
    ds = rng.choice([0.5, 1.0, 2.0])
    dp = delta*ds + 0.5*gamma*ds**2
    q = (f"Option has Δ = {delta:.2f}, Γ = {gamma:.3f}. Spot rises by {ds:.1f}. "
         f"Estimate option price change: ΔP ≈ Δ·ΔS + ½Γ·ΔS².")
    tr = (_assume([f"ΔP ≈ Δ·ΔS + ½Γ·ΔS²"]) +
          f"Step 1. Δ·ΔS = {delta:.2f}×{ds:.1f} = {delta*ds:.3f}.\n"
          f"Step 2. ½Γ·ΔS² = ½×{gamma:.3f}×{ds:.1f}² = {0.5*gamma*ds**2:.3f}.\n"
          f"Step 3. ΔP = {delta*ds:.3f} + {0.5*gamma*ds**2:.3f} = {dp:.3f}.\n"
          f"Step 4. Trap: using only Δ (no convexity Γ) understates the gain for large moves.")
    flaw = {"answer": f"{fmt(delta*ds)}", "pitfall": "omitting gamma",
            "reasoning_trace": (_assume([f"delta-only approximation"]) +
            f"Step 1. Using only Δ·ΔS = {delta:.2f}×{ds:.1f} = {fmt(delta*ds)} drops the convexity "
            f"term ½ΓΔS² = {fmt(0.5*gamma*ds**2)}; gamma matters for large underlying moves.")}
    return {"meta":{"topic":"Derivatives","subtopic":"Greeks","difficulty":"L2_Medium",
                    "question_type":"Calculation","pitfalls":["gamma/convexity","Taylor approximation"]},
            "question":q, "answer":f"{dp:.3f}", "distractors":[f"{fmt(delta*ds)}", f"{fmt(dp*2)}", f"{fmt(-dp)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"delta":delta,"gamma":gamma,"ds":ds}}

def deriv_swap_value(rng, seq):
    notional = rng.randint(50, 150) * 1000000
    fixed = rng.choice([0.03, 0.04, 0.05])
    float_expected = rng.choice([0.03, 0.04, 0.05])
    years = rng.randint(2, 5)
    value = notional*(fixed - float_expected)*years
    q = (f"Receive-fixed swap: notional {fmt(notional)}, fixed rate {pct(fixed,1)}, "
         f"expected float {pct(float_expected,1)}, {years} yrs remaining. Approximate swap value = "
         f"notional × (fixed − float) × years.")
    tr = (_assume([f"value ≈ notional × (fixed − float) × years"]) +
          f"Step 1. Rate differential = {pct(fixed,1)} − {pct(float_expected,1)} = {pct(fixed-float_expected,1)}.\n"
          f"Step 2. Value = {fmt(notional)}×{pct(fixed-float_expected,1)}×{years} = {fmt(value)} "
          f"({'positive (receiving above-market fixed)' if value>0 else 'negative (fixed below market)'}).\n"
          f"Step 3. Trap: valuing from the wrong payer perspective flips the sign.")
    flaw = {"answer": f"{fmt(-value)}", "pitfall": "payer/receiver perspective",
            "reasoning_trace": (_assume([f"payer perspective"]) +
            f"Step 1. The payer-fixed counterparty sees value −{fmt(value)} because it pays {pct(fixed,1)} "
            f"and receives float {pct(float_expected,1)}. The receiver gains {fmt(value)}; sign depends on "
            f"which side you value.")}
    return {"meta":{"topic":"Derivatives","subtopic":"Swaps","difficulty":"L2_Medium",
                    "question_type":"Calculation","pitfalls":["payer vs receiver","swap sign"]},
            "question":q, "answer":f"{fmt(value)}", "distractors":[f"{fmt(-value)}", f"{fmt(value*2)}", f"{fmt(notional*(float_expected-fixed))}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"notional":notional,"fixed":fixed,"float_expected":float_expected,"years":years}}

def fsa_diluted_eps(rng, seq):
    ni = rng.randint(20, 40) * 1000000
    shares = rng.randint(5, 10) * 1000000
    conv_shares = rng.randint(1, 3) * 1000000
    interest = rng.randint(2, 5) * 100000
    tax = rng.choice([0.20, 0.25, 0.30])
    basic = ni/shares
    diluted = (ni + interest*(1-tax))/(shares + conv_shares)
    q = (f"NI {fmt(ni)}, common shares {fmt(shares)}, convertible bonds (interest {fmt(interest)}, "
         f"tax {pct(tax,1)}) convertible into {fmt(conv_shares)} shares. Compute basic and diluted EPS.")
    tr = (_assume([f"basic = NI/shares", f"diluted adds IF-converted interest (net of tax)"]) +
          f"Step 1. Basic EPS = {fmt(ni)}/{fmt(shares)} = {fmt(basic)}.\n"
          f"Step 2. If-converted interest, net of tax = {fmt(interest)}×(1−{tax:.2f}) = {fmt(interest*(1-tax))}.\n"
          f"Step 3. Diluted EPS = ({fmt(ni)}+{fmt(interest*(1-tax))})/({fmt(shares)}+{fmt(conv_shares)}) = {fmt(diluted)}.\n"
          f"Step 4. Trap: adding full (pre-tax) interest overstates diluted EPS numerator.")
    flaw = {"answer": f"{fmt((ni + interest)/(shares + conv_shares))}", "pitfall": "tax on if-converted interest",
            "reasoning_trace": (_assume([f"pre-tax interest"]) +
            f"Step 1. Adding gross interest {fmt(interest)} to NI: ({fmt(ni)}+{fmt(interest)})/"
            f"({fmt(shares)}+{fmt(conv_shares)}) = {fmt((ni+interest)/(shares+conv_shares))}. "
            f"Interest must be net of tax ({1-tax:.2f}×{fmt(interest)} = {fmt(interest*(1-tax))}).")}
    return {"meta":{"topic":"Financial Statement Analysis","subtopic":"EPS","difficulty":"L2_Hard",
                    "question_type":"Calculation","pitfalls":["if-converted","net of tax"]},
            "question":q, "answer":f"basic {fmt(basic)}; diluted {fmt(diluted)}",
            "distractors":[f"{fmt((ni+interest)/(shares+conv_shares))}", f"{fmt(basic*2)}", f"{fmt(diluted*0.5)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"ni":ni,"shares":shares,"conv_shares":conv_shares,"interest":interest,"tax":tax}}


def quant_multi_reg(rng, seq):
    b0 = rng.randint(2, 4)
    b1 = rng.randint(5, 9)
    b2 = rng.randint(2, 5)
    x1 = rng.randint(3, 6)
    x2 = rng.randint(4, 7)
    yhat = b0 + b1 * x1 + b2 * x2
    q = (f"Regression y = {b0} + {b1}·x1 + {b2}·x2. Given x1={x1}, x2={x2}, "
         f"compute the predicted y.")
    tr = (_assume([f"predicted value = intercept + sum of coefficient·predictor"]) +
          f"Step 1. ŷ = {b0} + {b1}×{x1} + {b2}×{x2} = {b0} + {b1*x1} + {b2*x2} = {yhat}.\n"
          f"Step 2. Trap: dropping the intercept gives {b1*x1 + b2*x2}.")
    flaw = {"answer": f"{b1*x1 + b2*x2}", "pitfall": "intercept omitted",
            "reasoning_trace": (_assume([f"intercept dropped"]) +
            f"Step 1. ŷ = {b1}×{x1} + {b2}×{x2} = {b1*x1} + {b2*x2} = {b1*x1+b2*x2}.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Multiple Regression","difficulty":"L2_Easy",
                     "question_type":"Calculation","pitfalls":["intercept omitted"]},
            "question":q, "answer":f"{yhat}",
            "distractors":[f"{b1*x1+b2*x2}", f"{b1*x1}", f"{b2*x2}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"b0":b0,"b1":b1,"b2":b2,"x1":x1,"x2":x2}}

def port_apt(rng, seq):
    rf = rng.uniform(0.02, 0.04, 3)
    lam = rng.uniform(0.04, 0.08, 3)
    f1 = rng.uniform(0.5, 1.2, 3)
    f2 = rng.uniform(0.3, 1.0, 3)
    exp = rf + lam * (f1 + f2)
    q = (f"APT: E[R] = r_f + λ(f1+f2), with r_f={pct(rf)}, λ={pct(lam)}, "
         f"factor sensitivities f1={f1:.2f}, f2={f2:.2f}. Compute expected return.")
    tr = (_assume([f"APT expected return = risk-free + λ·(sum of sensitivities)"]) +
          f"Step 1. E[R] = {pct(rf)} + {pct(lam)}×({f1:.2f}+{f2:.2f}) = {pct(rf)} + {pct(lam*(f1+f2))} = {pct(exp)}.\n"
          f"Step 2. Trap: using only f1 gives {pct(rf + lam*f1)}.")
    flaw = {"answer": f"{pct(rf + lam*f1)}", "pitfall": "both factor sensitivities required",
            "reasoning_trace": (_assume([f"only f1 sensitivity used"]) +
            f"Step 1. E[R] = {pct(rf)} + {pct(lam)}×{f1:.2f} = {pct(rf+lam*f1)}.")}
    return {"meta": {"topic":"Portfolio Management","subtopic":"Portfolio Concepts","difficulty":"L2_Medium",
                     "question_type":"Calculation","pitfalls":["both factor sensitivities required"]},
            "question":q, "answer":f"{pct(exp)}",
            "distractors":[f"{pct(rf+lam*f1)}", f"{pct(rf+lam*f2)}", f"{pct(lam*(f1+f2))}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"rf":rf,"lam":lam,"f1":f1,"f2":f2}}

m_quant_multi_reg = wrap_mcq(quant_multi_reg)

TEMPLATES = {
    "eq_fcff_dcf": eq_fcff_dcf, "eq_fcfe": eq_fcfe, "eq_residual_income": eq_residual_income,
    "fi_spot_forward": fi_spot_forward, "fi_bond_price": fi_bond_price,
    "deriv_bsm_call": deriv_bsm_call, "deriv_delta_gamma": deriv_delta_gamma,
    "deriv_swap_value": deriv_swap_value, "fsa_diluted_eps": fsa_diluted_eps,
    "quant_multi_reg": quant_multi_reg,
    "port_apt": port_apt,
    "m_quant_multi_reg": m_quant_multi_reg,
}
