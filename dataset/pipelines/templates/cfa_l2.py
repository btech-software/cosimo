"""
CFA Level II templates. Same contract as cfa_l1.
"""
import math
import math as _m
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


def v2_multiplier_models(rng, seq):
    noev_growth = rng.uniform(0.4, 0.8, 2)
    sales_growth = rng.uniform(0.06, 0.12, 3)
    roe_base = rng.uniform(0.14, 0.20, 3)
    dtr = rng.uniform(0.35, 0.55, 2)
    plowback = 1 - dtr
    sustainable = roe_base * plowback * noev_growth
    noev = roe_base * plowback
    if sales_growth >= sustainable:
        sales_growth = sustainable - rng.uniform(0.01, 0.04, 2)
    q = (f"Net equity multiplier = {noev_growth:.2f}, sales growth = {pct(sales_growth,2)}, "
         f"ROE = {pct(roe_base,1)}, dividend payout ratio = {pct(dtr,1)}%. "
         f"Sustainable growth rate from equity base: g_s = ROE x (1-D/P) x multiplier. "
         f"What is the maximum sustainable sales growth rate?")
    tr = (_assume([f"g_s = ROE x plowback ratio x equity multiplier"]) +
          f"Step 1. Plowback ratio = 1 - {dtr:.2f} = {plowback:.3f}.\n"
          f"Step 2. ROE x plowback = {roe_base:.3f} x {plowback:.3f} = {roe_base*plowback:.4f}.\n"
          f"Step 3. Multiply by net equity multiplier {noev_growth:.2f}: "
          f"g_s = {roe_base*plowback:.4f} x {noev_growth:.2f} = {sustainable:.4f} = {pct(sustainable)}.\n"
          f"Step 4. Trap: using plowback alone (without multiplier) gives "
          f"{pct(noev)}, which understates sustainable growth for levered firms.")
    flaw = {"answer": f"{pct(noev)}", "pitfall": "plowback without multiplier",
            "reasoning_trace": (_assume([f"omitted equity multiplier"]) +
                  f"Step 1. g = ROE x plowback = {roe_base:.3f} x {plowback:.3f} = {roe_base*plowback:.4f} = {pct(noev)}.\n"
                  f"Net equity multiplier {noev_growth:.2f} must be applied: g_s = {pct(sustainable)}.")
}
    return {"meta":{"topic":"Equity Valuation","subtopic":"Multiplier Models","difficulty":"v2_L2_Medium",
                    "question_type":"Calculation","pitfalls":["plowback without multiplier","g>ROE"]},
            "question":q, "answer":f"{pct(sustainable)}",
            "distractors":[f"{pct(roe_base*plowback)}", f"{pct(roe_base*plowback*noev_growth*1.5)}",
                           f"{pct(roe_base*sustainable)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"noev_growth":noev_growth,"sales_growth":sales_growth,"roe_base":roe_base,
                      "dtr":dtr,"plowback":plowback,"sustainable":sustainable}}


def v2_eco_efficiency(rng, seq):
    re_e = rng.uniform(0.12, 0.20, 3)
    roe_actual = rng.uniform(0.16, 0.26, 3)
    roe_economic = rng.uniform(0.12, 0.22, 3)
    if roe_economic >= roe_actual:
        roe_economic = roe_actual - rng.uniform(0.02, 0.08, 2)
    eco_premium = roe_actual - roe_economic
    q = (f"Reed Co has a required return on equity of {pct(re_e,1)}. Its actual ROE is "
         f"{pct(roe_actual,2)}, and its economic ROE (industry average) is {pct(roe_economic,2)}. "
         f"Economic premium value = (actual ROE - economic ROE) / required return. "
         f"What is the eco-premium value per dollar of book value?")
    tr = (_assume([f"Eco Premium = (actual ROE - economic ROE) / r_e"]) +
          f"Step 1. ROE spread = {roe_actual:.3f} - {roe_economic:.3f} = {eco_premium:.4f}.\n"
          f"Step 2. Eco Premium = {eco_premium:.4f} / {re_e:.3f} = {eco_premium/re_e:.4f}.\n"
          f"Step 3. Trap: reporting just the ROE spread {eco_premium:.4f} omits discounting by r_e, "
          f"overstating the present value of the premium.")
    flaw = {"answer": f"{fmt(eco_premium)}", "pitfall": "raw spread vs discounted value",
            "reasoning_trace": (_assume([f"omitted division by r_e"]) +
                  f"Step 1. Reporting ROE spread {eco_premium:.4f} without dividing by "
                  f"r_e = {re_e:.3f} gives the annual excess, not the present value. "
                  f"Eco premium = {eco_premium:.4f}/{re_e:.3f} = {eco_premium/re_e:.4f}.")
}
    return {"meta":{"topic":"Equity Valuation","subtopic":"Economic Value Added","difficulty":"v2_L2_Medium",
                    "question_type":"Calculation","pitfalls":["raw spread vs discounted","r_e denominator"]},
            "question":q, "answer":f"{fmt(eco_premium/re_e)}",
            "distractors":[f"{fmt(roe_actual - re_e)}", f"{fmt(eco_premium)}", f"{fmt(eco_premium*2)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"re_e":re_e,"roe_actual":roe_actual,"roe_economic":roe_economic,
                      "eco_premium":eco_premium}}


def v2_dividend_discount(rng, seq):
    d1 = rng.uniform(2.0, 5.0, 2)
    g = rng.uniform(0.03, 0.07, 3)
    re = rng.uniform(0.10, 0.15, 3)
    if g >= re:
        g = re - rng.uniform(0.03, 0.06, 3)
    v0 = d1 / (re - g)
    q = (f"Next dividend D1 = ${fmt(d1)}, growth rate g = {pct(g,2)}, "
         f"required return on equity = {pct(re,1)}. Compute stock price via DDM: "
         f"P0 = D1 / (re - g).")
    tr = (_assume([f"P0 = D1 / (re - g)", f"g < re"]) +
          f"Step 1. re - g = {re:.3f} - {g:.3f} = {re-g:.4f}.\n"
          f"Step 2. P0 = ${fmt(d1)} / {re-g:.4f} = ${fmt(v0)}.\n"
          f"Step 3. Trap: using D0 (not D1) in the numerator gives {fmt(d1/(1+g)/(re-g))}.")
    flaw = {"answer": f"{fmt(d1/(1+g)/(re-g))}", "pitfall": "D0 vs D1",
            "reasoning_trace": (_assume([f"D0 used instead of D1"]) +
                  f"Step 1. Using D0 = ${fmt(d1)}/(1+{g:.3f}) in the numerator: "
                  f"${fmt(d1/(1+g))}/{re-g:.4f} = {fmt(d1/(1+g)/(re-g))}. DDM requires next-period D1.")
}
    return {"meta":{"topic":"Equity Valuation","subtopic":"Dividend Discount Models","difficulty":"v2_L2_Medium",
                    "question_type":"Calculation","pitfalls":["D0 vs D1","g < re required"]},
            "question":q, "answer":f"${fmt(v0)}",
            "distractors":[f"${fmt(d1/(1+g)/(re-g))}", f"${fmt(d1/(re+g))}", f"${fmt(v0*1.5)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"d1":d1,"g":g,"re":re,"v0":v0}}


def v2_fi_credit_spread(rng, seq):
    t_maturity = rng.uniform(1.0, 4.0, 2)
    y_corp = rng.uniform(0.05, 0.09, 3)
    y_treasury = rng.choice([0.04, 0.045, 0.05, 0.055])
    spread = y_corp - y_treasury
    q = (f"A {t_maturity:.0f}-year corporate bond yields {pct(y_corp,2)}. "
         f"The comparable {int(t_maturity)}-year government bond yields {pct(y_treasury,2)}. "
         f"Compute the credit spread.")
    tr = (_assume([f"credit spread = corporate yield - risk-free yield"]) +
          f"Step 1. Spread = {pct(y_corp,2)} - {pct(y_treasury,2)} = {pct(spread)}.\n"
          f"Step 2. Trap: adding the yield and T-bond rate ({pct(y_corp+y_treasury,2)}) "
          f"does not isolate the credit component.")
    flaw = {"answer": f"{pct(y_corp+y_treasury)}", "pitfall": "adding instead of subtracting",
            "reasoning_trace": (_assume([f"spread = corporate + risk-free"]) +
                  f"Step 1. Adding: {pct(y_corp,2)} + {pct(y_treasury,2)} = {pct(y_corp+y_treasury,2)}. "
                  f"Spread is the DIFFERENCE: corporate - risk-free = {pct(spread)}.")
}
    return {"meta":{"topic":"Fixed Income","subtopic":"Credit Spreads","difficulty":"v2_L2_Hard",
                    "question_type":"Calculation","pitfalls":["adding vs subtracting","maturity match"]},
            "question":q, "answer":f"{pct(spread)}",
            "distractors":[f"{pct(y_corp+y_treasury)}", f"{pct(y_corp-y_treasury*1.1)}", f"{pct(y_corp*0.5)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"t_maturity":t_maturity,"y_corp":y_corp,"y_treasury":y_treasury,"spread":spread}}


def v2_der_binomial_multi(rng, seq):
    s0 = rng.randint(50, 80)
    u = rng.uniform(1.10, 1.25, 3)
    d = rng.uniform(0.75, 0.90, 3)
    r_per = rng.choice([0.04, 0.05, 0.06])
    n_steps = rng.choice([2, 3])
    k = rng.randint(int(s0*0.90), int(s0*1.05))
    pu = (1+r_per - d) / (u - d)
    pu = max(0.01, min(0.99, pu))
    d_r = round(d, 4)
    u_r = round(u, 4)
    r_rate = round(r_per, 4)
    def _payoff(n_up, n_down):
        su = s0 * (u_r ** n_up) * (d_r ** n_down)
        return max(su - k, 0)
    if n_steps == 2:
        p_up2 = pu * pu
        p_mid = 2 * pu * (1-pu)
        p_down2 = (1-pu) * (1-pu)
        call_val = (p_up2 * _payoff(2,0) + p_mid * _payoff(1,1) + p_down2 * _payoff(0,2)) / ((1+r_per)**2)
    else:
        p_up3 = pu**3
        p_2up1dn = 3*pu**2*(1-pu)
        p_1up2dn = 3*pu*(1-pu)**2
        p_dn3 = (1-pu)**3
        call_val = (p_up3*_payoff(3,0) + p_2up1dn*_payoff(2,1) + p_1up2dn*_payoff(1,2) + p_dn3*_payoff(0,3)) / ((1+r_per)**3)
    wrong_call = _payoff(n_steps, 0) / ((1+r_per)**n_steps) if n_steps == 2 else (_payoff(3,0) + _payoff(2,1) + _payoff(1,2) + _payoff(0,3)) / (4*((1+r_per)**3))
    q = (f"Binomial option pricing (n={n_steps} steps, S0={s0}, K={k}). "
         f"Up factor u={u_r:.3f}, down factor d={d_r}, per-step r={pct(r_rate,1)}. "
         f"Compute the European call price via the risk-neutral approach.")
    tr = (_assume([f"risk-neutral probability pu = (1+r-d)/(u-d)",
                  f"call price = discounted expected payoff"]) +
          f"Step 1. pu = (1+{r_rate} - {d_r}) / ({u_r} - {d_r}) = {pu:.4f}.\n"
          f"Step 2. Build binomial tree from S0={s0}: up-up = {s0*u_r:.1f}, "
          f"up-down = {s0*u_r*d_r:.1f}, down-down = {s0*d_r:.1f}.\n"
          f"Step 3. Payoffs at maturity: max(S-K, 0).\n"
          f"Step 4. Discount back with expected payoff: C = {fmt(call_val)}.\n"
          f"Step 5. Trap: using physical probability 50/50 gives {fmt(wrong_call)} - "
          f"must use the risk-neutral probability {pu:.4f} to discount.")
    flaw = {"answer": f"{fmt(wrong_call)}", "pitfall": "physical vs risk-neutral prob",
            "reasoning_trace": (_assume([f"50/50 physical probability"]) +
                  f"Step 1. Using equal probabilities for up/down (no arbitrage pricing): "
                  f"C = {fmt(wrong_call)}. The binomial model requires the risk-neutral "
                  f"probability pu = {pu:.4f} to prevent arbitrage.")
}
    return {"meta":{"topic":"Derivatives","subtopic":"Binomial Option Pricing","difficulty":"v2_L2_Hard",
                    "question_type":"Calculation","pitfalls":["physical vs risk-neutral prob","arbitrage-free"]},
            "question":q, "answer":f"{fmt(call_val)}",
            "distractors":[f"{fmt(wrong_call)}", f"{fmt(max(s0-k,0))}", f"{fmt(call_val*2)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"s0":s0,"u":u_r,"d":d_r,"r":r_rate,"n_steps":n_steps,"k":k,"pu":pu}}


def v2_op_active_passive(rng, seq):
    active_ret = rng.uniform(0.10, 0.18, 3)
    bench_ret = rng.uniform(0.07, 0.13, 3)
    if bench_ret >= active_ret:
        bench_ret = active_ret - rng.uniform(0.02, 0.08, 2)
    active_beta = rng.uniform(0.85, 1.15, 3)
    bench_vol = rng.uniform(0.12, 0.20, 3)
    active_vol = rng.uniform(0.14, 0.25, 3)
    rf = rng.uniform(0.02, 0.04, 3)
    jensen_alpha = active_ret - (rf + active_beta * (bench_ret - rf))
    treynor = (active_ret - rf) / active_beta
    sharpe = (active_ret - rf) / active_vol
    bench_sharpe = (bench_ret - rf) / bench_vol
    q = (f"A fund earned {pct(active_ret,1)} (beta {active_beta:.2f}). "
         f"The benchmark returned {pct(bench_ret,2)}, and the risk-free rate is {pct(rf,1)}. "
         f"Fund vol = {pct(active_vol,1)}, bench vol = {pct(bench_vol,2)}. "
         f"Compute Jensen alpha, Treynor ratio, and Sharpe ratio, then evaluate performance.")
    tr = (_assume([f"Jensen alpha = R_p - [r_f + b(R_m - r_f)]",
                  f"Treynor = (R_p - r_f) / b_p",
                  f"Sharpe = (R_p - r_f) / s_p"]) +
          f"Step 1. Jensen alpha = {pct(active_ret,1)} - [{pct(rf,1)} + {active_beta:.2f} x ({pct(bench_ret,2)} - {pct(rf,1)})] = {pct(jensen_alpha)}.\n"
          f"Step 2. Treynor = ({pct(active_ret,1)} - {pct(rf,1)}) / {active_beta:.2f} = {fmt(treynor)}.\n"
          f"Step 3. Sharpe = ({pct(active_ret,1)} - {pct(rf,1)}) / {pct(active_vol,1)} = {fmt(sharpe)}.\n"
          f"Step 4. Bench Sharpe = ({pct(bench_ret,2)} - {pct(rf,1)}) / {pct(bench_vol,2)} = {fmt(bench_sharpe)}.\n"
          f"Step 5. Trap: comparing alpha directly to benchmark return ignores the risk premium component.")
    flaw = {"answer": f"{fmt(active_ret-bench_ret)}", "pitfall": "raw return difference vs risk-adjusted",
            "reasoning_trace": (_assume([f"raw difference"]) +
                  f"Step 1. Reporting {pct(active_ret,1)} - {pct(bench_ret,2)} = {fmt(active_ret-bench_ret)} ignores "
                  f"market risk premium and beta adjustment. Jensen alpha = {pct(jensen_alpha)}, which accounts "
                  f"for systematic risk exposure.")
}
    return {"meta":{"topic":"Active Portfolio Management","subtopic":"Active/Passive Evaluation","difficulty":"v2_L2_Hard",
                    "question_type":"Calculation","pitfalls":["raw return vs risk-adjusted","jensen alpha formula"]},
            "question":q, "answer":f"alpha {pct(jensen_alpha)}; Treynor {fmt(treynor)}; Sharpe {fmt(sharpe)}",
            "distractors":[f"{fmt(active_ret-bench_ret)}", f"{pct(active_ret)}", f"{fmt(treynor*2)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"active_ret":active_ret,"bench_ret":bench_ret,"active_beta":active_beta,
                      "bench_vol":bench_vol,"active_vol":active_vol,"rf":rf,"jensen_alpha":jensen_alpha,
                      "treynor":treynor,"sharpe":sharpe,"bench_sharpe":bench_sharpe}}


def v2_fs_ashenanigans(rng, seq):
    rev0 = rng.randint(100, 300) * 1000000
    rev1 = rev0 * (1 + rng.uniform(0.05, 0.15, 3))
    rev2 = rev1 * (1 + rng.uniform(0.05, 0.15, 3))
    rev3 = rev2 * (1 + rng.uniform(0.02, 0.08, 3))
    rev4 = rev3 * (1 - rng.uniform(0.02, 0.08, 3))
    cogs_pct = rng.uniform(0.55, 0.70, 2)
    cogs0 = rev0 * cogs_pct
    gross0 = rev0 - cogs0
    gross_margin = gross0 / rev0
    q = (f"Revenue (5 yrs): ${fmt(rev0)}, ${fmt(rev1)}, ${fmt(rev2)}, ${fmt(rev3)}, ${fmt(rev4)}. "
         f"COGS = {pct(cogs_pct*100,1)}% of revenue. Gross margin in Year 1 = {pct(gross_margin,1)}%. "
         f"Rev growth accelerates from Y2->Y3 then declines Y3->Y4. Which Shenanigans signals might be present?")
    g23 = rev3/rev2 - 1 if rev2 > 0 else 0
    g34 = rev4/rev3 - 1 if rev3 > 0 else 0
    q2 = (f"Revenue (5 yrs): ${fmt(rev0)}, ${fmt(rev1)}, ${fmt(rev2)}, ${fmt(rev3)}, ${fmt(rev4)}. "
          f"COGS = {pct(cogs_pct*100,1)}% of revenue. Gross margin in Year 1 = {pct(gross_margin,1)}%. "
          f"Rev growth accelerates from Y2 to Y3 then declines Y3 to Y5. Which Shenanigans signals might be present?")
    tr = (_assume([f"Revenue growth accelerates Y2->Y3, then decelerates Y3->Y4"]) +
          f"Step 1. Revenue CAGR Y0->Y4 = {(rev4/rev0)**(1/4)-1:.3f} = {pct((rev4/rev0)**(1/4)-1)}.\n"
          f"Step 2. Y2->Y3 growth = {rev3/rev2-1:.3f} = {pct(g23)},"
          f"Y3->Y4 growth = {pct(g34)} -> Y4 declined from Y3 peak.\n"
          f"Step 3. Gross margin: {pct(gross_margin,1)}% (stable). "
          f"Trap: stable margins with accelerating then declining revenue suggests channel-stuffing.")
    flaw = {"answer": "No issue", "pitfall": "missing shenanigans pattern",
            "reasoning_trace": (_assume([f"stable margins"]) +
                  f"Step 1. Stable gross margin at {pct(gross_margin,1)}% masks revenue growth "
                  f"{pct(g23)} from Y2->Y3 followed by decline {pct(g34)} at Y3->Y4. "
                  f"This pattern (accelerate then reverse) is a classic big-bath indicator. "
                  f"Check receivables growth vs revenue.")
}
    return {"meta":{"topic":"Financial Statement Analysis","subtopic":"Financial Statement Shenanigans","difficulty":"v2_L2_Hard",
                    "question_type":"Calculation","pitfalls":["revenue recognition timing","big bath"]},
            "question":q2, "answer":"Revenue acceleration then deceleration pattern",
            "distractors":["No issue","Margin expansion","Cost overruns","Currency effects"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"rev0":rev0,"rev1":rev1,"rev2":rev2,"rev3":rev3,"rev4":rev4,
                      "cogs_pct":cogs_pct,"gross_margin":gross_margin}}


def v2_port_fundamental_risky(rng, seq):
    mu = rng.uniform(0.08, 0.14, 3)
    sigma = rng.uniform(0.15, 0.30, 3)
    rf = rng.uniform(0.02, 0.04, 3)
    t_horizon = rng.choice([1, 2, 3])
    z = (mu - rf) * (t_horizon ** 0.5) / sigma
    from math import erf, sqrt as _sqrt
    prob_under_rf = 0.5 * (1 + erf(-z / _sqrt(2)))
    evr = mu - sigma * _m.exp(0.5 * sigma**2 * t_horizon) * (1 - 2*prob_under_rf)
    q = (f"Fundamentally risky asset: expected return mu = {pct(mu,1)}, s = {pct(sigma,1)}. "
         f"Risk-free rate = {pct(rf,1)}. Time horizon = {t_horizon} yr. "
         f"Estimate the probability of underperforming the risk-free rate.")
    tr = (_assume([f"normal distribution: r ~ N(mu, s*sqrt(T))"]) +
          f"Step 1. Z-score for rf: z = (mu-rf)*sqrt(T)/s = "
          f"({mu:.4f}-{rf:.4f})*{t_horizon**0.5:.3f}/{sigma:.3f} = {z:.3f}.\n"
          f"Step 2. P(r < rf) = N(-z) = {prob_under_rf:.4f} = {pct(prob_under_rf)}.\n"
          f"Step 3. EVR = mu - s*exp(0.5*s^2*T)*(1-2P) = {fmt(evr)}.\n"
          f"Step 4. Trap: treating s as absolute risk (vs tracking error) mislabels Fundamental Risk.")
    flaw = {"answer": f"{pct(prob_under_rf*2)}", "pitfall": "confusing total vol with tracking error",
            "reasoning_trace": (_assume([f"total volatility"]) +
                  f"Step 1. Using total vol s = {pct(sigma,1)} instead of tracking error (e.g., 5%) "
                  f"doubles the probability to {pct(prob_under_rf*2)}. Fundamental Risk requires "
                  f"tracking error relative to the benchmark, not the absolute vol of the portfolio.")
}
    return {"meta":{"topic":"Portfolio Management","subtopic":"Fundamentally Risky Portfolios","difficulty":"v2_L2_Hard",
                    "question_type":"Calculation","pitfalls":["total vol vs tracking error","EVR definition"]},
            "question":q, "answer":f"{pct(prob_under_rf)}",
            "distractors":[f"{pct(prob_under_rf*2)}", f"{pct((mu-rf)/100)}", f"{pct((mu*sigma)/100)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"mu":mu,"sigma":sigma,"rf":rf,"t_horizon":t_horizon,
                      "prob_under_rf":prob_under_rf,"z":z}}


TEMPLATES = {
    "eq_fcff_dcf": eq_fcff_dcf, "eq_fcfe": eq_fcfe, "eq_residual_income": eq_residual_income,
    "fi_spot_forward": fi_spot_forward, "fi_bond_price": fi_bond_price,
    "deriv_bsm_call": deriv_bsm_call, "deriv_delta_gamma": deriv_delta_gamma,
    "deriv_swap_value": deriv_swap_value, "fsa_diluted_eps": fsa_diluted_eps,
    "quant_multi_reg": quant_multi_reg,
    "port_apt": port_apt,
    "m_quant_multi_reg": m_quant_multi_reg,
    "v2_multiplier_models": v2_multiplier_models,
    "v2_eco_efficiency": v2_eco_efficiency,
    "v2_dividend_discount": v2_dividend_discount,
    "v2_fi_credit_spread": v2_fi_credit_spread,
    "v2_der_binomial_multi": v2_der_binomial_multi,
    "v2_op_active_passive": v2_op_active_passive,
    "v2_fs_ashenanigans": v2_fs_ashenanigans,
    "v2_port_fundamental_risky": v2_port_fundamental_risky,
}
