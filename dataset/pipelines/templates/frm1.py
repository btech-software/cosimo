"""
FRM Part 1 templates (market risk, VaR, Greeks, valuation).
"""
import math
from pipelines.core import fmt, pct
from pipelines.templates.wrappers import wrap_mcq

PROG = "FRM_Part_1"

def _assume(a):
    return "ASSUMPTIONS: " + "; ".join(a) + ".\n"

def mkt_param_var(rng, seq):
    value = rng.randint(50, 200) * 1000000
    sig_d = rng.uniform(0.005, 0.02, 4)
    z = rng.choice([1.645, 1.96, 2.576])
    var = value*sig_d*z
    q = (f"Portfolio {fmt(value)}, daily σ {pct(sig_d,2)}, parametric {pct(1-z*0,1)} VaR with z={z}. "
         f"Compute one-day VaR = V·σ·z.")
    tr = (_assume([f"parametric VaR = V × σ × z", f"normal returns"]) +
          f"Step 1. σ×z = {pct(sig_d,2)}×{z} = {sig_d*z:.5f}.\n"
          f"Step 2. VaR = {fmt(value)}×{sig_d*z:.5f} = {fmt(var)}.\n"
          f"Step 3. Interpretation: {pct(1-(0.5 if z==1.645 else 0.05 if z==1.96 else 0.01))} confidence.\n"
          f"Step 4. Trap: using annual σ without the √daily scaling overstates VaR.")
    flaw = {"answer": f"{fmt(value*sig_d*z*math.sqrt(250))}", "pitfall": "time aggregation",
            "reasoning_trace": (_assume([f"annualizing daily VaR"]) +
            f"Step 1. Multiplying daily VaR by √250 = {fmt(math.sqrt(250))} scales to a 1-year horizon "
            f"{fmt(value*sig_d*z*math.sqrt(250))}; the question asks one-day VaR = {fmt(var)}. "
            f"Daily VaR uses daily σ, not annualized.")}
    return {"meta":{"topic":"Market Risk","subtopic":"Parametric VaR","difficulty":"FRM1_Medium",
                    "question_type":"Calculation","pitfalls":["time aggregation","confidence level"]},
            "question":q, "answer":f"{fmt(var)}", "distractors":[f"{fmt(value*sig_d*z*math.sqrt(250))}", f"{fmt(value*sig_d)}", f"{fmt(var*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"value":value,"sig_d":sig_d,"z":z}}

def mkt_cvar(rng, seq):
    var = rng.randint(20, 60) * 1000000
    loss_avg = rng.uniform(1.2, 1.8, 3)  # avg loss multiple of VaR
    cvar = var*loss_avg
    q = (f"Daily 95% VaR = {fmt(var)}. Expected loss given that VaR is breached = {loss_avg:.2f}×VaR. "
         f"Compute CVaR/ES.")
    tr = (_assume([f"CVaR = E[loss | loss > VaR]"]) +
          f"Step 1. CVaR = VaR × loss-multiple = {fmt(var)}×{loss_avg:.2f} = {fmt(cvar)}.\n"
          f"Step 2. CVaR ≥ VaR always; CVaR captures tail severity, VaR only the threshold.\n"
          f"Step 3. Trap: reporting VaR as the expected tail loss ignores the magnitude beyond the threshold.")
    flaw = {"answer": f"{fmt(var)}", "pitfall": "CVaR vs VaR",
            "reasoning_trace": (_assume([f"using VaR as tail loss"]) +
            f"Step 1. Quoting VaR {fmt(var)} as the expected tail loss ignores that breaches average "
            f"{loss_avg:.2f}× VaR; CVaR = {fmt(cvar)}. VaR is a quantile, not a conditional expectation.")}
    return {"meta":{"topic":"Market Risk","subtopic":"CVaR/ES","difficulty":"FRM1_Medium",
                    "question_type":"Calculation","pitfalls":["CVaR vs VaR","tail severity"]},
            "question":q, "answer":f"{fmt(cvar)}", "distractors":[f"{fmt(var)}", f"{fmt(cvar*2)}", f"{fmt(var*loss_avg*0.5)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"var":var,"loss_avg":loss_avg}}

def greek_delta_hedge(rng, seq):
    delta = rng.uniform(0.4, 0.7, 3)
    units = rng.randint(1000, 5000) * 100
    hedge_shares = delta*units
    q = (f"Call delta {delta:.2f} on {units} call options (each on 1 share). "
         f"Compute shares needed for a delta-neutral hedge.")
    tr = (_assume([f"delta-neutral: shares = Δ × units"]) +
          f"Step 1. Shares = {delta:.2f}×{units} = {fmt(hedge_shares)}.\n"
          f"Step 2. This offsets the short-call position's Δ exposure.\n"
          f"Step 3. Trap: forgetting Δ (using 1 share per option) over-hedges and leaves residual risk.")
    flaw = {"answer": f"{fmt(units)}", "pitfall": "ignoring delta",
            "reasoning_trace": (_assume([f"1:1 hedge"]) +
            f"Step 1. Hedging 1 share per option ({fmt(units)}) ignores the option delta {delta:.2f}; "
            f"correct hedge is Δ×units = {delta:.2f}×{fmt(units)} = {fmt(hedge_shares)}.")}
    return {"meta":{"topic":"Options","subtopic":"Delta Hedging","difficulty":"FRM1_Easy",
                    "question_type":"Calculation","pitfalls":["delta weighting","hedge ratio"]},
            "question":q, "answer":f"{fmt(hedge_shares)}", "distractors":[f"{fmt(units)}", f"{fmt(hedge_shares*2)}", f"{fmt(delta)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"delta":delta,"units":units}}

def bond_var_duration(rng, seq):
    value = rng.randint(50, 150) * 1000000
    mod = rng.uniform(3, 7, 3)
    dy = rng.uniform(0.001, 0.005, 4)
    pchange = -mod*dy
    var = value*pchange
    q = (f"Bond portfolio {fmt(value)}, modified duration {mod:.2f}. Yield rises by {pct(dy,2)}. "
         f"Estimate price change and VaR: ΔV ≈ −ModDur × Δy × V.")
    tr = (_assume([f"ΔP/P ≈ −ModDur × Δy", f"ΔV ≈ ΔP/P × V"]) +
          f"Step 1. ΔP/P = −{mod:.2f}×{pct(dy,2)} = {pchange:.5f} = {pct(pchange)}.\n"
          f"Step 2. ΔV = {pct(pchange)}×{fmt(value)} = {fmt(var)} (loss).\n"
          f"Step 3. Trap: ignoring the negative sign reports a gain; yields rising ⇒ price falls.")
    flaw = {"answer": f"{fmt(-var)}", "pitfall": "duration sign",
            "reasoning_trace": (_assume([f"sign error"]) +
            f"Step 1. Writing ΔV = +ModDur×Δy×V = {fmt(-var)} treats a yield increase as a gain. "
            f"Duration measures price sensitivity with a NEGATIVE sign for yields: ΔV = −{mod:.2f}×{pct(dy,2)}×{fmt(value)} = {fmt(var)} (loss)." )}
    return {"meta":{"topic":"Market Risk","subtopic":"Bond VaR","difficulty":"FRM1_Hard",
                    "question_type":"Calculation","pitfalls":["duration sign","duration approximation"]},
            "question":q, "answer":f"{fmt(var)}", "distractors":[f"{fmt(-var)}", f"{fmt(value*pchange*2)}", f"{fmt(value)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"value":value,"mod":mod,"dy":dy}}

def reg_beta_corr(rng, seq):
    rho = rng.uniform(0.3, 0.8, 3)
    sig_a = rng.uniform(0.15, 0.30, 3)
    sig_m = rng.uniform(0.10, 0.20, 3)
    beta = rho*sig_a/sig_m
    q = (f"ρ(asset, market) = {rho:.2f}, σ_asset {pct(sig_a,1)}, σ_market {pct(sig_m,1)}. "
         f"Compute asset beta = ρ·σ_a/σ_m.")
    tr = (_assume([f"β = ρ × σ_a / σ_m"]) +
          f"Step 1. ρ×σ_a = {rho:.2f}×{pct(sig_a,1)} = {rho*sig_a:.4f}.\n"
          f"Step 2. Divide by σ_m = {pct(sig_m,1)}: β = {beta:.2f}.\n"
          f"Step 3. Trap: using correlation alone ({rho:.2f}) as beta ignores the volatility ratio.")
    flaw = {"answer": f"{fmt(rho*sig_a)}", "pitfall": "beta vs correlation",
            "reasoning_trace": (_assume([f"using ρ as β"]) +
            f"Step 1. Reporting correlation {rho:.2f} as beta ignores σ_a/σ_m = {pct(sig_a,1)}/{pct(sig_m,1)} "
            f"= {sig_a/sig_m:.2f}; β = ρ×σ_a/σ_m = {beta:.2f}.")}
    return {"meta":{"topic":"Quantitative Methods","subtopic":"Regression & Beta","difficulty":"FRM1_Medium",
                    "question_type":"Calculation","pitfalls":["beta vs correlation","volatility ratio"]},
            "question":q, "answer":f"β = {beta:.2f}", "distractors":[f"β = {fmt(rho)}", f"β = {fmt(rho*sig_a)}", f"β = {fmt(beta*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"rho":rho,"sig_a":sig_a,"sig_m":sig_m}}

def bsm_put_parity(rng, seq):
    s = rng.randint(45, 60)
    k = rng.randint(45, 55)
    r = rng.choice([0.04, 0.05, 0.06])
    t = rng.uniform(0.5, 2.0, 3)
    call = rng.uniform(3, 8, 3)
    put = call + k*math.exp(-r*t) - s
    q = (f"Put-call parity: S={s}, K={k}, r={pct(r,1)}, T={t:.1f}, call price {call:.2f}. "
         f"Compute the put price: P = C + K·e^(−rT) − S.")
    tr = (_assume([f"put-call parity: C − P = S − PV(K)"]) +
          f"Step 1. PV(K) = {k}×e^(−{r:.2f}×{t:.1f}) = {k*math.exp(-r*t):.3f}.\n"
          f"Step 2. P = {call:.2f} + {k*math.exp(-r*t):.3f} − {s} = {fmt(put)}.\n"
          f"Step 3. Trap: using K undiscounted ({k}) overstates the put by the discount factor.")
    flaw = {"answer": f"{fmt(call + k - s)}", "pitfall": "undiscounted strike",
            "reasoning_trace": (_assume([f"undiscounted strike"]) +
            f"Step 1. P = C + K − S = {call:.2f} + {k} − {s} = {fmt(call+k-s)}. "
            f"Put-call parity uses the PRESENT VALUE of the strike K·e^(−rT) = {k*math.exp(-r*t):.3f}, "
            f"giving P = {fmt(put)}.")}
    return {"meta":{"topic":"Options","subtopic":"Put-Call Parity","difficulty":"FRM1_Medium",
                    "question_type":"Calculation","pitfalls":["PV of strike","parity formula"]},
            "question":q, "answer":f"{fmt(put)}", "distractors":[f"{fmt(call+k-s)}", f"{fmt(call)}", f"{fmt(put*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"s":s,"k":k,"r":r,"t":t,"call":call}}


def mkt_historical_var(rng, seq):
    pv = rng.randint(100, 200) * 1000
    sigma = rng.uniform(0.01, 0.02, 4)
    n = rng.choice([10, 20, 250])
    var95 = pv * sigma * 1.645
    q = (f"Portfolio value {fmt(pv)}, daily volatility {pct(sigma)}. "
         f"Compute 1-day 95% VaR.")
    tr = (_assume([f"VaR(95%) = V × σ × 1.645"]) +
          f"Step 1. VaR = {fmt(pv)} × {pct(sigma)} × 1.645 = {fmt(var95)}.\n"
          f"Step 2. Trap: using 99% z-score 2.326 gives {fmt(pv*sigma*2.326)}.")
    flaw = {"answer": f"{fmt(pv*sigma*2.326)}", "pitfall": "confidence z-score",
            "reasoning_trace": (_assume([f"99% z-score used"]) +
            f"Step 1. VaR = {fmt(pv)} × {pct(sigma)} × 2.326 = {fmt(pv*sigma*2.326)}.")}
    return {"meta": {"topic":"Valuation and Risk Models","subtopic":"VaR Models","difficulty":"FRM1_Medium",
                     "question_type":"Calculation","pitfalls":["confidence z-score"]},
            "question":q, "answer":f"{fmt(var95)}",
            "distractors":[f"{fmt(pv*sigma*2.326)}", f"{fmt(pv*sigma)}", f"{fmt(pv)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"pv":pv,"sigma":sigma,"n":n}}

def fut_forward_price(rng, seq):
    s0 = rng.randint(80, 120) * 10
    r = rng.uniform(0.03, 0.06, 4)
    t = rng.choice([0.25, 0.5, 1.0])
    f0 = s0 * (1 + r) ** t
    sim = s0 * (1 + r * t)
    if abs(sim - f0) < 1e-9:
        sim = s0 * (1 + 2 * r * t)
    q = (f"Spot {fmt(s0)}, risk-free rate {pct(r)}, maturity {t} yr. "
         f"Compute the no-arbitrage forward price.")
    tr = (_assume([f"F_0 = S_0 × (1+r)^T"]) +
          f"Step 1. F_0 = {fmt(s0)} × (1+{pct(r)})^{t:.2f} = {fmt(f0)}.\n"
          f"Step 2. Trap: using simple r×T without compounding gives {fmt(s0*(1+r*t))}.")
    flaw = {"answer": f"{fmt(s0*(1+r*t))}", "pitfall": "compounding over T",
            "reasoning_trace": (_assume([f"simple interest used"]) +
            f"Step 1. F_0 = {fmt(s0)} × (1+{pct(r)}×{t:.2f}) = {fmt(s0*(1+r*t))}.")}
    return {"meta": {"topic":"Financial Markets and Products","subtopic":"Forward and Futures","difficulty":"FRM1_Easy",
                     "question_type":"Calculation","pitfalls":["compounding over T"]},
            "question":q, "answer":f"{fmt(f0)}",
            "distractors":[f"{fmt(sim)}", f"{fmt(s0)}", f"{fmt(s0*r*t)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"s0":s0,"r":r,"t":t}}

m_mkt_historical_var = wrap_mcq(mkt_historical_var)
m_fut_forward_price = wrap_mcq(fut_forward_price)

TEMPLATES = {
    "mkt_param_var": mkt_param_var, "mkt_cvar": mkt_cvar, "greek_delta_hedge": greek_delta_hedge,
    "bond_var_duration": bond_var_duration, "reg_beta_corr": reg_beta_corr, "bsm_put_parity": bsm_put_parity,
    "mkt_historical_var": mkt_historical_var,
    "fut_forward_price": fut_forward_price,
    "m_mkt_historical_var": m_mkt_historical_var,
    "m_fut_forward_price": m_fut_forward_price,
}
