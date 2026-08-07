"""
FRM Part 1 templates (market risk, VaR, Greeks, valuation).
"""
import math
from pipelines.core import fmt, pct, render_trace
from pipelines.templates.wrappers import wrap_cr, wrap_mcq, wrap_vignette

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
    steps = [
        ("Step 1.", 'σ×z = {pct(sig_d,2)}×{z} = {sig_d*z:.5f}.\\n'),
        ("Step 2.", 'VaR = {fmt(value)}×{sig_d*z:.5f} = {fmt(var)}.\\n'),
        ("Step 3.", 'Interpretation: {pct(1-(0.5 if z==1.645 else 0.05 if z==1.96 else 0.01))} confidence.\\n'),
        ("Step 4.", 'Trap: using annual σ without the √daily scaling overstates VaR'),
    ]
    tr = render_trace(rng, ['parametric VaR = V × σ × z', 'normal returns'], steps, conclusion='Trap: using annual σ without the √daily scaling overstates VaR')
    flaw = {"answer": f"{fmt(value*sig_d*z*math.sqrt(250))}", "pitfall": "time aggregation",
            "reasoning_trace": render_trace(rng, ['annualizing daily VaR'], [
                ("Step 1.", 'Multiplying daily VaR by √250 = {fmt(math.sqrt(250))} scales to a 1-year horizon \n{fmt(value*sig_d*z*math.sqrt(250))}; the question asks one-day VaR = {fmt(var)}. \nDaily VaR uses daily σ, not annualized'),
            ]),
                   }
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
    steps = [
        ("Step 1.", 'CVaR = VaR × loss-multiple = {fmt(var)}×{loss_avg:.2f} = {fmt(cvar)}.\\n'),
        ("Step 2.", 'CVaR ≥ VaR always; CVaR captures tail severity, VaR only the threshold.\\n'),
        ("Step 3.", 'Trap: reporting VaR as the expected tail loss ignores the magnitude beyond the threshold'),
    ]
    tr = render_trace(rng, ['CVaR = E[loss | loss > VaR]'], steps, conclusion='Trap: reporting VaR as the expected tail loss ignores the magnitude beyond the threshold')
    flaw = {"answer": f"{fmt(var)}", "pitfall": "CVaR vs VaR",
            "reasoning_trace": render_trace(rng, ['using VaR as tail loss'], [
                ("Step 1.", 'Quoting VaR {fmt(var)} as the expected tail loss ignores that breaches average \n{loss_avg:.2f}× VaR; CVaR = {fmt(cvar)}. VaR is a quantile, not a conditional expectation'),
            ]),
                   }
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
    steps = [
        ("Step 1.", 'Shares = {delta:.2f}×{units} = {fmt(hedge_shares)}.\\n'),
        ("Step 2.", "This offsets the short-call position's Δ exposure.\\n"),
        ("Step 3.", 'Trap: forgetting Δ (using 1 share per option) over-hedges and leaves residual risk'),
    ]
    tr = render_trace(rng, ['delta-neutral: shares = Δ × units'], steps, conclusion='Trap: forgetting Δ (using 1 share per option) over-hedges and leaves residual risk')
    flaw = {"answer": f"{fmt(units)}", "pitfall": "ignoring delta",
            "reasoning_trace": render_trace(rng, ['1:1 hedge'], [
                ("Step 1.", 'Hedging 1 share per option ({fmt(units)}) ignores the option delta {delta:.2f}; \ncorrect hedge is Δ×units = {delta:.2f}×{fmt(units)} = {fmt(hedge_shares)}'),
            ]),
                   }
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
    steps = [
        ("Step 1.", 'ΔP/P = −{mod:.2f}×{pct(dy,2)} = {pchange:.5f} = {pct(pchange)}.\\n'),
        ("Step 2.", 'ΔV = {pct(pchange)}×{fmt(value)} = {fmt(var)} (loss).\\n'),
        ("Step 3.", 'Trap: ignoring the negative sign reports a gain; yields rising ⇒ price falls'),
    ]
    tr = render_trace(rng, ['ΔP/P ≈ −ModDur × Δy', 'ΔV ≈ ΔP/P × V'], steps, conclusion='Trap: ignoring the negative sign reports a gain; yields rising ⇒ price falls')
    flaw = {"answer": f"{fmt(-var)}", "pitfall": "duration sign",
            "reasoning_trace": render_trace(rng, ['sign error'], [
                ("Step 1.", 'Writing ΔV = +ModDur×Δy×V = {fmt(-var)} treats a yield increase as a gain. \nDuration measures price sensitivity with a NEGATIVE sign for yields: ΔV = −{mod:.2f}×{pct(dy,2)}×{fmt(value)} = {fmt(var)} (loss)'),
            ]),
                   }
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
    steps = [
        ("Step 1.", 'ρ×σ_a = {rho:.2f}×{pct(sig_a,1)} = {rho*sig_a:.4f}.\\n'),
        ("Step 2.", 'Divide by σ_m = {pct(sig_m,1)}: β = {beta:.2f}.\\n'),
        ("Step 3.", 'Trap: using correlation alone ({rho:.2f}) as beta ignores the volatility ratio'),
    ]
    tr = render_trace(rng, ['β = ρ × σ_a / σ_m'], steps, conclusion='Trap: using correlation alone ({rho:.2f}) as beta ignores the volatility ratio')
    flaw = {"answer": f"{fmt(rho*sig_a)}", "pitfall": "beta vs correlation",
            "reasoning_trace": render_trace(rng, ['using ρ as β'], [
                ("Step 1.", 'Reporting correlation {rho:.2f} as beta ignores σ_a/σ_m = {pct(sig_a,1)}/{pct(sig_m,1)} \n= {sig_a/sig_m:.2f}; β = ρ×σ_a/σ_m = {beta:.2f}'),
            ]),
                   }
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
    steps = [
        ("Step 1.", 'PV(K) = {k}×e^(−{r:.2f}×{t:.1f}) = {k*math.exp(-r*t):.3f}.\\n'),
        ("Step 2.", 'P = {call:.2f} + {k*math.exp(-r*t):.3f} − {s} = {fmt(put)}.\\n'),
        ("Step 3.", 'Trap: using K undiscounted ({k}) overstates the put by the discount factor'),
    ]
    tr = render_trace(rng, ['put-call parity: C − P = S − PV(K)'], steps, conclusion='Trap: using K undiscounted ({k}) overstates the put by the discount factor')
    flaw = {"answer": f"{fmt(call + k - s)}", "pitfall": "undiscounted strike",
            "reasoning_trace": render_trace(rng, ['undiscounted strike'], [
                ("Step 1.", 'P = C + K − S = {call:.2f} + {k} − {s} = {fmt(call+k-s)}. \nPut-call parity uses the PRESENT VALUE of the strike K·e^(−rT) = {k*math.exp(-r*t):.3f}, \ngiving P = {fmt(put)}'),
            ]),
                   }
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
    steps = [
        ("Step 1.", 'VaR = {fmt(pv)} × {pct(sigma)} × 1.645 = {fmt(var95)}.\\n'),
        ("Step 2.", 'Trap: using 99% z-score 2.326 gives {fmt(pv*sigma*2.326)}'),
    ]
    tr = render_trace(rng, ['VaR(95%) = V × σ × 1.645'], steps, conclusion='Trap: using 99% z-score 2.326 gives {fmt(pv*sigma*2.326)}')
    flaw = {"answer": f"{fmt(pv*sigma*2.326)}", "pitfall": "confidence z-score",
            "reasoning_trace": render_trace(rng, ['99% z-score used'], [
                ("Step 1.", 'VaR = {fmt(pv)} × {pct(sigma)} × 2.326 = {fmt(pv*sigma*2.326)}'),
            ]),
                   }
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
    steps = [
        ("Step 1.", 'F_0 = {fmt(s0)} × (1+{pct(r)})^{t:.2f} = {fmt(f0)}.\\n'),
        ("Step 2.", 'Trap: using simple r×T without compounding gives {fmt(s0*(1+r*t))}'),
    ]
    tr = render_trace(rng, ['F_0 = S_0 × (1+r)^T'], steps, conclusion='Trap: using simple r×T without compounding gives {fmt(s0*(1+r*t))}')
    flaw = {"answer": f"{fmt(s0*(1+r*t))}", "pitfall": "compounding over T",
            "reasoning_trace": render_trace(rng, ['simple interest used'], [
                ("Step 1.", 'F_0 = {fmt(s0)} × (1+{pct(r)}×{t:.2f}) = {fmt(s0*(1+r*t))}'),
            ]),
                   }
    return {"meta": {"topic":"Financial Markets and Products","subtopic":"Forward and Futures","difficulty":"FRM1_Easy",
                      "question_type":"Calculation","pitfalls":["compounding over T"]},
            "question":q, "answer":f"{fmt(f0)}",
            "distractors":[f"{fmt(sim)}", f"{fmt(s0)}", f"{fmt(s0*r*t)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"s0":s0,"r":r,"t":t}}

def v1_qa_time_series(rng, seq):
    # AR(1) model: Xt = φ·Xt-1 + εt
    phi = rng.uniform(0.2, 0.85, 3)
    xt_1 = rng.uniform(2.0, 8.0, 3)
    # shock from a pre-defined set so answer is a concrete number
    eps_values = [0.1, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0]
    eps = rng.choice(eps_values)
    xt = phi * xt_1 + eps
    # also compute 1-step ahead forecast and steady-state variance
    forecast_next = phi * xt  # E[Xt+1|Ft]
    ss_var = 1.0 / (1.0 - phi**2)  # var(ε) = 1

    q = (f"An AR(1) process: Xt = φ·Xt-1 + εt with φ = {phi:.3f}. "
          f"The previous observation Xt-1 = {xt_1:.3f} and current shock εt = {eps:.1f}. "
          f"Compute the current value Xt of the series.")
    tr = (_assume([f"AR(1) Xt = φ·Xt-1 + εt", f"εt independent N(0,1)",
                   f"WSS if |φ| < 1"]) +
          f"Step 1. Xt = {phi:.3f} × {xt_1:.3f} + {eps:.1f} = {phi*xt_1:.3f} + {eps:.1f} = {xt:.3f}.\n"
          f"Step 2. One-step forecast E[Xt+1 | Ft] = φ×Xt = {phi:.3f}×{xt:.3f} = {forecast_next:.3f}.\n"
          f"Step 3. Stationarity check: |φ|={phi:.3f} < 1 → process is stationary. "
          f"Steady-state variance Var(Xt) = σ²_ε/(1−φ²) = {ss_var:.3f}.\n"
          f"Step 4. Trap: using Xt = φ + εt ignores the lagged value {xt_1:.3f}.")
    flaw = {"answer": f"{fmt(phi + eps)}", "pitfall": "ignoring lagged value",
            "reasoning_trace": render_trace(rng, ['Xt = φ + εt'], [
                ("Step 1.", '{phi:.3f} + {eps:.1f} = {phi + eps:.3f} — this ignores the lagged value Xt-1 = {xt_1:.3f}. \nCorrect: Xt = φ×Xt-1 + εt = {phi:.3f}×{xt_1:.3f} + {eps:.1f} = {xt:.3f}'),
            ]),
                   }
    return {"meta":{"topic":"Quantitative Methods","subtopic":"Time Series Analysis","difficulty":"v1_FRM1_Hard",
                    "question_type":"Calculation","pitfalls":["lagged value","stationarity check"]},
            "question":q, "answer":f"{xt:.3f}",
            "distractors":[f"{fmt(phi+eps)}", f"{fmt(phi*xt_1)}", f"{fmt(ss_var)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"phi":phi,"xt_1":xt_1,"eps":eps,"xt":xt}}

def v1_qa_simulation(rng, seq):
    # Monte Carlo: compute PV of an option payoff with 3 simulated paths
    s0 = rng.randint(90, 115)
    k = rng.randint(95, 105)  # often at-the-money or slightly out
    r = rng.uniform(0.03, 0.06, 4)
    sigma = rng.uniform(0.15, 0.30, 3)
    t = 1.0  # 1 year
    n_sim = 3  # small for deterministic trace (always 3 paths)

    # Generate 3 random paths using fixed uniform seeds
    u_values = []
    for _ in range(n_sim):
        u_values.append(rng.uniform(-2.5, 2.5))  # z-scores for normal

    # GBM price at T: S_T = S0 * exp((r - 0.5*σ²)*T + σ*√T*Z)
    dt = t
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)

    payoffs = []
    path_vals = []
    for z in u_values:
        st = s0 * math.exp(drift + diffusion * z)
        path_vals.append(st)
        payoff = max(st - k, 0)  # call payoff
        payoffs.append(payoff)

    avg_payoff = sum(payoffs) / n_sim
    pv = avg_payoff * math.exp(-r * t)

    q = (f"Monte Carlo pricing: a European call, S0 = {fmt(s0)}, K = {fmt(k)}, "
          f"r = {pct(r)}, σ = {pct(sigma)}, T = 1 yr. "
          f"3 simulated terminal prices: {[f'{v:.2f}' for v in path_vals]}. "
          f"Compute the option PV (average discounted payoff).")
    trace_parts = (_assume([f"GBM: ST = S0 × exp((r−0.5σ²)T + σ√T·Z)",
                            f"call payoff = max(ST−K, 0)"]) +
                   f"Step 1. Drift = (r − ½σ²)×T = ({r:.4f} − 0.5×{sigma:.2f}²)×1 = {drift:.4f}.\n"
                   f"Step 2. Diffusion = σ×√T = {sigma:.4f}×1 = {diffusion:.4f}.\n")
    for i, (z, st, pay) in enumerate(zip(u_values, path_vals, payoffs)):
        trace_parts += f"  Path {i+1}: Z={z:.3f} → ST = {st:.2f} → payoff = {pay:.2f}.\n"
    trace_parts += (f"Step 3. Avg payoff = {avg_payoff:.2f}.\n"
                    f"Step 4. PV = {avg_payoff:.2f}×e^(−{r:.4f}×1) = {fmt(pv)}.\n"
                    f"Step 5. Trap: averaging terminal prices instead of discounting each payoff, "
                    f"or using the forward price E[ST] directly ignores the optionality.")
    flaw = {"answer": f"{fmt(s0 - k if s0 > k else 0)}", "pitfall": "intrinsic value only",
            "reasoning_trace": render_trace(rng, ['intrinsic value'], [
                ("Step 1.", 'Quoting the intrinsic value max(S0−K, 0) = {fmt(s0 - k if s0 > k else 0)} ignores both \ntime value (σ = {pct(sigma)}) and simulation-based discounting. \nMonte Carlo PV = {fmt(pv)}, which includes the random payoff distribution'),
            ]),
                   }
    return {"meta":{"topic":"Quantitative Methods","subtopic":"Monte Carlo Simulation","difficulty":"v1_FRM1_Hard",
                    "question_type":"Calculation","pitfalls":["optionality","discount each payoff"]},
            "question":q, "answer":f"{fmt(pv)}",
            "distractors":[f"{fmt(s0-k)}", f"{fmt(avg_payoff)}", f"{fmt(pv*2)}"],
            "reasoning_trace":trace_parts, "flawed":flaw,
            "params":{"s0":s0,"k":k,"r":r,"sigma":sigma,"path_vals":path_vals,"payoffs":payoffs}}

def v1_qa_prob_stats(rng, seq):
    # Probability: binomial CDF - P(X <= k) for specific params
    n = rng.choice([8, 9, 10, 11])
    p = rng.choice([0.3, 0.35, 0.4])
    k_target = rng.randint(3, 6)
    # compute P(X <= k) = sum C(n, j) * p^j * (1-p)^(n-j) for j=0..k
    def comb_fn(a, b):
        if b < 0 or b > a:
            return 0
        r = 1
        for i in range(min(b, a - b)):
            r = r * (a - i) // (i + 1)
        return r

    cdf_val = 0
    individual_probs = {}
    for j in range(k_target + 1):
        pj = comb_fn(n, j) * (p ** j) * ((1 - p) ** (n - j))
        cdf_val += pj
        individual_probs[j] = pj
    mean_n = n * p
    sd_n = math.sqrt(n * p * (1 - p))

    q = (f"Binomial distribution: n = {n} trials, P(success) = {pct(p)}. "
          f"What is P(X <= {k_target})?")

    # Build step 1 string pre-joined to avoid f-string inline expression issues
    step1_base = f"Step 1. Compute CDF P(X <= {k_target}) = "
    step1_formula = f"sum_j=0^{k_target} C({n},j)*p^j*(1-p)^({n}-j).\\n"
    term_lines = ""
    for j, pj in individual_probs.items():
        term_lines += f"    P(X={j}) = {comb_fn(n, j)} * {pct(p)}^j * {(1-p):.4f}^{n}-{j} = {pj:.6f}\n"

    tr = (_assume([f"C(n,k) = n! / (k!*(n-k)!)"]) +
          step1_base + "\n" +
          term_lines +
          f"Step 2. Sum = {cdf_val:.6f}.\n"
          f"Step 3. Mean = n*p = {mean_n:.1f}, StdDev = sqrt(n*p*(1-p)) = {sd_n:.3f}.\n"
          f"Step 4. Trap: using P(X = {k_target}) = {individual_probs.get(k_target, 0):.6f} instead of "
          f"the cumulative sum up to {k_target}.")
    flawed_ans = individual_probs.get(k_target, 0)
    flaw = {"answer": f"{flawed_ans:.6f}", "pitfall": "PMF instead of CDF",
            "reasoning_trace": render_trace(rng, ['P(X = k) only'], [
                ("Step 1.", 'Computing P(X = {k_target}) = {flawed_ans:.6f} ignores the cumulative \nnature of P(X <= {k_target}). CDF sums P(X=0) through P(X={k_target}) = {cdf_val:.6f}'),
            ]),
                   }
    return {"meta":{"topic":"Quantitative Methods","subtopic":"Probability Distributions","difficulty":"v1_FRM1_Medium",
                    "question_type":"Calculation","pitfalls":["CDF vs PMF","binomial expansion"]},
            "question":q, "answer":f"P(X <= {k_target}) = {cdf_val:.6f}",
            "distractors":[f"P(X <= {k_target}) = {flawed_ans:.6f}", f"{cdf_val * 0.5:.6f}", f"{1 - cdf_val:.6f}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"n":n,"p":p,"k_target":k_target,"cdf_val":cdf_val}}

def v1_fm_corp_bond(rng, seq):
    # Corporate bond pricing: price as sum of discounted cash flows with credit spread
    face = 100  # per 100 par
    coupon_rate = rng.choice([0.03, 0.035, 0.04, 0.045, 0.05])
    years = rng.choice([3, 5, 7, 10])
    spot_rates = {
        1: rng.uniform(0.03, 0.045, 4),
        2: rng.uniform(0.035, 0.050, 4),
        3: rng.uniform(0.04, 0.055, 4),
        5: rng.uniform(0.042, 0.060, 4),
    }
    # If needed term > available, interpolate
    credit_spread = rng.uniform(0.005, 0.02, 4)
    annual_coupon = face * coupon_rate

    # Build spot rates for all terms needed (interpolate linearly)
    available_terms = sorted(spot_rates.keys())

    def get_spot(y):
        if y in spot_rates:
            return spot_rates[y]
        # interpolate between closest available terms
        low = high = available_terms[0]
        for t in available_terms:
            if t < y:
                low = t
            if t > y:
                high = t
                break
        if high == low:
            return spot_rates[low]
        frac = (y - low) / (high - low)
        return spot_rates[low] * (1 - frac) + spot_rates[high] * frac

    price = 0
    cf_list = []
    for t in range(1, years + 1):
        sr = get_spot(t)
        ytm_total = sr + credit_spread  # spread-adjusted rate
        if t < years:
            cf = annual_coupon
            df = 1.0 / ((1 + ytm_total) ** t)
        else:
            cf = annual_coupon + face
            df = 1.0 / ((1 + ytm_total) ** t)
        price += cf * df
        cf_list.append((t, cf, df, cf * df))

    ytm_approx = coupon_rate + (face - price) / (years * (face + price) / 2)

    q = (f"A {years}-year corporate bond, 100 par, coupon {pct(coupon_rate,1)}/yr. "
          f"Spot rates provided with credit spread {pct(credit_spread,1)} on top. "
          f"Compute the bond price by discounting each cash flow.")
    trace_parts = (_assume([f"Price = sum_CF_t / (1+YTM)^t where YTM = spot + spread"]) +
                   f"Cash flow analysis:\n")
    for t, cf, df, pv_t in cf_list:
        sr = get_spot(t)
        rate = sr + credit_spread
        trace_parts += f"  t={t}: CF = {fmt(cf)}, spot+spread = {pct(rate)}, " \
                       f"DF = {df:.6f}, PV = {fmt(pv_t)}\n"
    trace_parts += (f"Step 1. Sum of PVs = {fmt(price)}.\n"
                    f"Step 2. Approximate YTM = {pct(ytm_approx)}.\n"
                    f"Step 3. Trap: using a single yield without the spread overstates price; "
                    f"ignoring the credit spread prices the bond at risk-free levels.")

    # Simpler flawed: price without spread (use just spot rates)
    price_rf = 0
    for t in range(1, years + 1):
        sr = get_spot(t)
        if t < years:
            cf = annual_coupon
            df = 1.0 / ((1 + sr) ** t)
        else:
            cf = annual_coupon + face
            df = 1.0 / ((1 + sr) ** t)
        price_rf += cf * df

    flaw = {"answer": f"{fmt(price_rf)}", "pitfall": "ignoring credit spread",
            "reasoning_trace": render_trace(rng, ['risk-free discounting'], [
                ("Step 1.", 'Pricing at the risk-free spot rates alone = {fmt(price_rf)} ignores the \ncredit spread {pct(credit_spread,1)}. A corporate bond must be discounted at \nspot + credit spread, giving price = {fmt(price)}. \nThe spread accounts for default risk'),
            ]),
                   }
    return {"meta":{"topic":"Fixed Income","subtopic":"Corporate Bonds","difficulty":"v1_FRM1_Hard",
                    "question_type":"Calculation","pitfalls":["credit spread","risk-free rate"]},
            "question":q, "answer":f"{fmt(price)}",
            "distractors":[f"{fmt(price_rf)}", f"{fmt(face)}", f"{fmt(price + credit_spread)}"],
            "reasoning_trace":trace_parts, "flawed":flaw,
            "params":{"face":face,"coupon_rate":coupon_rate,"years":years,
                      "credit_spread":credit_spread,"price":price}}

def v1_fm_mortgage(rng, seq):
    # MBS / mortgage amortization: monthly payment on fixed-rate mortgage
    principal = rng.randint(200, 500) * 10000  # $2M to $5M
    annual_rate = rng.choice([0.035, 0.04, 0.0425, 0.045, 0.0475])
    years = rng.choice([15, 20, 30])
    months = years * 12
    monthly_rate = annual_rate / 12

    # Monthly payment: M = P * r * (1+r)^n / ((1+r)^n - 1)
    factor = (1 + monthly_rate) ** months
    monthly_payment = principal * monthly_rate * factor / (factor - 1)

    # First month: interest vs principal breakdown
    first_interest = principal * monthly_rate
    first_principal = monthly_payment - first_interest

    # Total interest over the life
    total_payments = monthly_payment * months
    total_interest = total_payments - principal

    # MBS passthrough rate (assume 0.25% servicing fee)
    pass_through_rate = annual_rate * 0.0025  # servicing fee as pct
    monthly_pt_payment = monthly_payment * (1 - pass_through_rate)
    servicing_fee_monthly = monthly_payment * pass_through_rate

    q = (f"A {fmt(principal)} mortgage, annual rate {pct(annual_rate,2)}, {years}-year term. "
          f"Compute the fixed monthly payment.")
    tr = (_assume([f"M = P × r × (1+r)^n / ((1+r)^n − 1)",
                   f"r = annual rate / 12, n = years × 12"]) +
          f"Step 1. Monthly rate r = {pct(annual_rate,2)} / 12 = {pct(monthly_rate,6)}.\n"
          f"Step 2. Number of payments n = {years}×12 = {months}.\n"
          f"Step 3. (1+r)^n = ({1+monthly_rate:.6f})^{months} = {factor:.2f}.\n"
          f"Step 4. M = {fmt(principal)}×{pct(monthly_rate,6)}×{factor:.2f} / ({factor:.2f}−1) = {fmt(monthly_payment)}.\n"
          f"Step 5. First month: interest = {fmt(principal)}×{pct(monthly_rate,6)} = {fmt(first_interest)}, "
          f"principal = {fmt(monthly_payment)}−{fmt(first_interest)} = {fmt(first_principal)}.\n"
          f"Step 6. Total interest over {years} yrs = {fmt(total_interest)}.\n"
          f"Step 7. Trap: using annual rate directly as monthly rate overstates the payment by factor of 12.")
    flaw = {"answer": f"{fmt(principal * annual_rate / 12)}", "pitfall": "annual rate used as monthly",
            "reasoning_trace": render_trace(rng, ['monthly rate = annual rate'], [
                ("Step 1.", '{fmt(principal * annual_rate / 12)} uses annual rate {pct(annual_rate)} \nas if it were the monthly rate. The monthly rate is r = {pct(annual_rate,2)} / 12 = {pct(monthly_rate)}, \nyielding M = {fmt(monthly_payment)}'),
            ]),
                   }
    return {"meta":{"topic":"Fixed Income","subtopic":"Mortgages","difficulty":"v1_FRM1_Hard",
                    "question_type":"Calculation","pitfalls":["monthly vs annual rate","amortization"]},
            "question":q, "answer":f"{fmt(monthly_payment)}",
            "distractors":[f"{fmt(principal * annual_rate / 12)}", f"{fmt(principal * annual_rate)}",
                           f"{fmt(monthly_payment * months)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"principal":principal,"annual_rate":annual_rate,"months":months,
                      "monthly_payment":monthly_payment,"first_interest":first_interest,
                      "total_interest":total_interest}}

def v1_fm_swap(rng, seq):
    # Plain vanilla interest rate swap: fixed rate calculation (par swap rate)
    notional = rng.randint(100, 300) * 1000000  # $100M to $300M
    fixed_rate_quarterly = rng.choice([0.0175, 0.02, 0.02125, 0.0225])  # quarterly rate
    fixed_rate_annual = fixed_rate_quarterly * 4
    # Simulate discount factors from a simple yield curve
    n_periods = rng.choice([4, 8])  # 1 year or 2 years
    # Simple flat yield curve for discounting
    y = rng.uniform(0.04, 0.065, 4)

    # Discount factors: DF_t = 1 / (1 + y/4)^t for quarterly
    df_list = []
    for t in range(1, n_periods + 1):
        df = 1 / ((1 + y/4) ** t)
        df_list.append(df)

    # Par swap rate: s = (1 - DF_n) / ΣDF_i
    sum_df = sum(df_list)
    last_df = df_list[-1]
    par_rate_quarterly = (1 - last_df) / sum_df
    swap_price_fv = 0  # At inception, swap value = 0 by construction

    q = (f"Quoted: 4-quarterly-pay swap, flat yield {pct(y)}. "
          f"Notional {fmt(notional)}. "
          f"What fixed quarterly rate makes the swap value zero at inception? "
          f"Compute the par swap rate.")
    trace_parts = (_assume([f"Par rate: s = (1 − DF_n) / Σ(DF_i)",
                            f"Discount quarterly: DF_t = (1+y/4)^(-t)"]) +
                   f"Discount factors for each period:\n")
    for i, df in enumerate(df_list):
        trace_parts += f"  Period {i+1}: DF = {df:.6f}\n"
    trace_parts += (f"Step 1. ΣDF = {sum_df:.6f}.\n"
                    f"Step 2. DF_final = {last_df:.6f}.\n"
                    f"Step 3. Par quarterly rate = (1 − {last_df:.6f}) / {sum_df:.6f} = {par_rate_quarterly:.6f} = {pct(par_rate_quarterly)}.\n"
                    f"Step 4. Annualized par rate = {pct(par_rate_quarterly)} × 4 = {pct(par_rate_quarterly * 4)}.\n"
                    f"Step 5. The swap value at inception = {fmt(swap_price_fv)} (zero by construction).\n"
                    f"Step 6. Trap: quoting the zero-coupon rate {pct(y)} as the swap rate ignores the annuity factor "
                    f"(the sum of discount factors).")
    flaw = {"answer": f"{pct(y)}", "pitfall": "yield rate vs swap rate",
            "reasoning_trace": render_trace(rng, ['swap rate = yield'], [
                ("Step 1.", 'The flat yield {pct(y)} is the zero-coupon rate, NOT the par swap rate. \nThe par swap rate accounts for the annuity of payments: \ns = (1 − DF_n)/ΣDF_i = {pct(par_rate_quarterly)} quarterly = {pct(par_rate_quarterly * 4)} annualized. \nSwaps pay a stream of fixed flows, so the rate must be spread across all periods'),
            ]),
                   }
    return {"meta":{"topic":"Financial Markets and Products","subtopic":"Swaps","difficulty":"v1_FRM1_Hard",
                    "question_type":"Calculation","pitfalls":["par rate vs yield","annuity factor"]},
            "question":q, "answer":f"par rate = {pct(par_rate_quarterly * 4)} p.a.",
            "distractors":[f"par rate = {pct(y)} p.a.", f"par rate = {pct(fixed_rate_annual)} p.a.",
                           f"par rate = {pct(y/2)} p.a."],
            "reasoning_trace":trace_parts, "flawed":flaw,
            "params":{"notional":notional,"n_periods":n_periods,"y":y,
                      "par_rate_quarterly":par_rate_quarterly,"fixed_rate_annual":fixed_rate_annual}}

def v1_qa_ethics_time_series(rng, seq):
    # Conceptual: time series model-selection bias
    data_type = rng.choice(["stationary", "non-stationary"])
    test_used = rng.choice(["unit root test", "graphical analysis", "AIC"])
    if data_type == "stationary":
        true_model = "AR(1) with φ = 0.3"
        problem = "The analyst found no unit root and fitted an AR(1) model. The series was indeed stationary, so the choice was reasonable, but she did not check model residuals for autocorrelation."
    else:
        true_model = "No meaningful AR structure (pure random walk)"
        problem = "Fitting an AR model to a non-stationary random walk is a unit root fallacy."

    q = (f"An analyst is examining a univariate time series for a risk forecasting model. "
          f"She uses only a {test_used} and fits the resulting {data_type}_model, "
          f"which indicates {true_model}. "
          f"What is the primary concern with this approach?")
    tr = ("A structured approach to evaluating model-selection risk:\n"
          f"1. Model identification: A {test_used} alone is insufficient to pin down the correct dynamics.\n"
          f"2. Residual diagnostics: Even if a model fits the data, residuals must be checked "
          f"for remaining autocorrelation, heteroskedasticity, or non-normality.\n"
          f"3. Overfitting vs underfitting: Fewer lags may miss structure; more lags waste degrees of freedom.\n"
          f"4. Out-of-sample validation: The model should be tested on a held-out period to confirm "
          f"forecasting power beyond the sample.\n"
          f"5. The key pitfall is treating model-selection as a one-step process and skipping "
          f"residual validation and out-of-sample testing.")

    flawed = {"answer": "No concern — the unit root test is sufficient",
              "pitfall": "single-test validation",
              "reasoning_trace": ("The unit root test or AIC selects the model order correctly.\n"
                                  "No further validation is needed before using the model.\n"
                                  "This is wrong because model selection is only step one "
                                  "— residual diagnostics and out-of-sample validation "
                                  "are essential to guard against misspecification.")}

    return {"meta": {"topic": "Quantitative Methods", "subtopic": "Time Series",
                     "difficulty": "v1_FRM1_Medium",
                     "question_type": "Constructed Response", "pitfalls": [
                         "model selection bias", "residual diagnostics", "overfitting"]},
            "question": q, "answer": "Model-selection bias: the approach skips residual diagnostics, "
            "out-of-sample validation, and overfitting checks. A single criterion (unit root test / AIC / "
            "graphical analysis) is necessary but not sufficient for correct time-series specification.",
            "distractors": [],
            "reasoning_trace": tr, "flawed": flawed,
            "params": {"data_type": data_type, "test_used": test_used}}

m_v1_qa_ethics_time_series = wrap_cr(v1_qa_ethics_time_series)

def v1_qa_concept_swap_valuation(rng, seq):
    # Conceptual: understanding swap valuation from holder's perspective
    scenario = rng.choice(["floating falls", "floating rises", "credit deterioration"])
    if scenario == "floating falls":
        direction = "The fixed-rate payer benefits"
        mechanism = "When floating rates fall after the swap is initiated, the fixed-rate payer pays the fixed rate and receives the now-lower floating rate, generating positive valuation."
    elif scenario == "floating rises":
        direction = "The fixed-rate payer suffers a mark-to-market loss"
        mechanism = "When floating rates rise, the fixed-rate payer still owes the fixed rate but receives a higher floating payment, making the fixed-rate leg relatively more expensive."
    else:
        direction = "The swap's credit risk increases for the fixed-rate receiver"
        mechanism = "Counterparty credit deterioration raises the probability that the floating-rate payer defaults when the swap is in-the-money to the fixed-rate receiver."

    q = (f"For a plain vanilla fixed-for-float interest rate swap, a major "
          f"{scenario} event has occurred since inception. "
          f"What is the primary valuation/credit concern?")
    tr = ("Step-by-step analysis:\n"
          f"1. Swap valuation: A swap’s value equals the sum of discounted differences between fixed and floating cash flows. "
          f"The value is zero at inception by construction.\n"
          f"2. {direction} — {mechanism}\n"
          f"3. Credit exposure: For the fixed-rate payer, mark-to-market value = -(fixed leg) + (floating leg). "
          f"When floating rates move, the positive side is uncertain (counterparty may not pay).\n"
          f"4. Framework: PFR (Potential Future Exposure) at each horizon uses simulated rate paths "
          f"to estimate E[positive MTM], not just the current MTM.\n"
          f"5. Trap: treating swap exposure as static (spot MTM only) ignores that exposure can spike "
          f"far from inception — PFR is a forward-looking measure.")

    flawed = {"answer": "None — swap contracts are symmetric and have zero value at all times",
              "pitfall": "static zero value",
              "reasoning_trace": ("The swap was zero at inception, so it remains zero forever.\n"
                                  "This is wrong because swap MTM changes as market rates move — "
                                  "the fixed leg is a bond (fixed prices), while the floating leg resets. "
                                  "The asymmetry creates positive or negative value over time.")}

    return {"meta": {"topic": "Financial Markets and Products", "subtopic": "Swaps",
                     "difficulty": "v1_FRM1_Medium",
                     "question_type": "Constructed Response", "pitfalls": [
                         "MTM vs inception value", "credit exposure measurement"]},
            "question": q, "answer": direction + ". " + mechanism[0].upper() + mechanism[1:],
            "distractors": [],
            "reasoning_trace": tr, "flawed": flawed,
            "params": {"scenario": scenario, "direction": direction}}

m_v1_qa_concept_swap_valuation = wrap_vignette(v1_qa_concept_swap_valuation)

m_mkt_historical_var = wrap_mcq(mkt_historical_var)
m_fut_forward_price = wrap_mcq(fut_forward_price)

m_v1_qa_time_series = wrap_mcq(v1_qa_time_series)
m_v1_qa_simulation = wrap_mcq(v1_qa_simulation)
m_v1_qa_prob_stats = wrap_mcq(v1_qa_prob_stats)
m_v1_fm_corp_bond = wrap_mcq(v1_fm_corp_bond)
m_v1_fm_mortgage = wrap_mcq(v1_fm_mortgage)
m_v1_fm_swap = wrap_mcq(v1_fm_swap)


def v1_mkt_backtesting(rng, seq):
    n_days = rng.choice([250, 500])
    exceptions = rng.randint(5, 18)
    exception_rate = exceptions / n_days
    if exceptions <= 3:
        zone, signal = "green", "acceptable - continue monitoring"
    elif exceptions <= 10:
        zone, signal = "yellow", "potentially elevated - consider model audit"
    else:
        zone, signal = "red", "unacceptable - capital surcharge; recalibration required"

    q = (f"Backtesting over {n_days} days: {exceptions} VaR exceedances observed. "
         f"Exception rate = {exception_rate:.3%}. "
         f"Under Basel backtesting, classify the zone and describe the response.")

    steps = [
        ("Step 1.", 'Expected exceptions at 99% VaR in {n_days} days = {n_days * 0.01:.0f}. \nObserved = {exceptions}.\\n'),
        ("Step 2.", 'Exception rate = {exceptions}/{n_days} = {exception_rate:.3%}.\\n'),
        ("Step 3.", 'Zone = {zone.upper()} -> {signal}.\\n'),
        ("Step 4.", 'Capital surcharge: green = minimum, yellow = 0.40x, red = 1.00x.\\n'),
        ("Step 5.", 'Trap: using total count without adjusting for window length'),
    ]
    tr = render_trace(rng, ['Green: 0-3 exceptions / Yellow: 4-10 / Red: 11+ for 99% VaR'], steps, conclusion='Trap: using total count without adjusting for window length')

    flaw = {"answer": "0.5% exception rate is acceptable", "pitfall": "ignoring zone classification",
            "reasoning_trace": (f"Step 1. The rate {exception_rate:.3%} must be mapped to a zone - "
                                f"the count {exceptions} drives the Basel surcharge. "
                                f"Zone {zone.upper()} means {signal}.")}

    return {"meta": {"topic": "Market Risk", "subtopic": "Backtesting",
                     "difficulty": "FRM1_Medium", "question_type": "Calculation",
                     "pitfalls": ["zone classification", "capital surcharge", "count vs rate"]},
            "question": q,
            "answer": f"Zone {zone.upper()} - {signal} ({exceptions}/{n_days})",
            "distractors": ["Zone green - rate is within bounds",
                            "Zone red - model is permanently invalid",
                            "Zone yellow - no supervisory action needed"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"n_days": n_days, "exceptions": exceptions, "zone": zone}}

m_v1_mkt_backtesting = wrap_mcq(v1_mkt_backtesting)




def v1_qa_var_basics(rng, seq):
    var_dollar = rng.randint(5, 50) * 1000000
    conf = rng.choice([95, 99])
    horizon = rng.choice([1, 10, 21])
    tail = 100.0 - conf

    q = (f"A financial institution reports a {conf}% {horizon}-day VaR of "
         f"${fmt(var_dollar, 0)}. What is the correct interpretation?")

    tr = (_assume(["VaR is a quantile at the given confidence level",
                   "VaR is not subadditive for non-elliptical distributions"]) +
          f"Step 1. {conf}% {horizon}-day VaR = ${fmt(var_dollar, 0)} means:\n"
          f"  - There is a {tail}% probability that loss over {horizon} days EXCEEDS VaR.\n"
          f"  - Equivalently, {conf}% of {horizon}-day losses are at or below VaR.\n"
          f"Step 2. VaR does NOT measure average loss given VaR is breached (that is CVaR/ES).\n"
          f"Step 3. VaR is not a coherent risk measure: VaR_port >= Sum(VaR_i) possible.\n"
          f"Step 4. Trap: interpreting VaR as a maximum loss. It is a statistical quantile.\n"
          f"    Tail losses can and do significantly exceed the VaR threshold.")

    flaw = {"answer": f"Maximum possible loss is ${fmt(var_dollar, 0)}",
            "pitfall": "VaR as maximum loss bound",
            "reasoning_trace": (f"Step 1. VaR at {conf}% confidence means a {tail}% chance of EXCEEDING VaR.\n"
                                f"CVaR/ES is needed to capture the tail severity beyond VaR.")}

    return {"meta": {"topic": "Market Risk", "subtopic": "VaR Basics",
                     "difficulty": "FRM1_Easy", "question_type": "MCQ",
                     "pitfalls": ["quantile vs max error", "CVaR/ES distinction"]},
            "question": q,
            "answer": (f"A {conf}% {horizon}-day VaR of ${fmt(var_dollar, 0)} means "
                       f"a {tail}% probability loss EXCEEDS this amount over {horizon} days."),
            "distractors": [f"Maximum loss = ${fmt(var_dollar, 0)}",
                            f"Average loss = ${fmt(var_dollar, 0)}",
                            f"{conf}% probability loss equals ${fmt(var_dollar, 0)}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"var_dollar": var_dollar, "conf": conf, "horizon": horizon}}

m_v1_qa_var_basics = wrap_mcq(v1_qa_var_basics)




def v1_qa_risk_types(rng, seq):
    scenarios = [
        ("A sudden 40% collapse in equity indices over three trading days",
         "market risk", "price volatility of traded assets"),
        ("A sovereign nation suspends foreign debt payments",
         "credit risk", "default of a counterparty or issuer"),
        ("A ransomware attack encrypts the bank's trading systems for 48 hours",
         "operational risk", "technology failure from an external cyber event"),
        ("A bank cannot roll over short-term repo funding overnight",
         "liquidity risk", "inability to fund or roll over liability maturities"),
    ]
    scenario, risk_type, explanation = rng.choice(scenarios)

    q = (f"Classify the risk type:\n"
         f"Scenario: {scenario}\n\n"
         f"What primary category of risk does this represent?")

    steps = [
        ("Step 1.", 'This is {risk_type}: {explanation}.\\n'),
        ("Step 2.", 'Distinguishing categories:\\n\n  Market risk: losses from price movements in rates, FX, equities, commodities.\\n\n  Credit risk: losses from default or downgrade of counterparties/instruments.\\n\n  Operational risk: losses from people, processes, systems, or external events.\\n\n  Liquidity risk: inability to fund liabilities (funding) or sell at fair value (market).\\n'),
        ("Step 3.", 'Interdependence: a credit event can trigger market risk and liquidity risk.\\n\n    Risk managers must map cross-category contagion channels'),
    ]
    tr = render_trace(rng, ['Basel risk taxonomy: Market, Credit, Operational, Liquidity'], steps)

    flaw = {"answer": "market risk - any loss is a market risk event",
            "pitfall": "not distinguishing risk categories",
            "reasoning_trace": (f"Step 1. Not all losses are market risk. Ransomware = operational, "
                                f"default = credit, funding freeze = liquidity. "
                                f"Basel framework has four distinct categories with specific capital charges.")}

    return {"meta": {"topic": "Market Risk", "subtopic": "Risk Types (Taxonomy)",
                     "difficulty": "FRM1_Easy", "question_type": "MCQ",
                     "pitfalls": ["risk taxonomy", "Basel categories"]},
            "question": q,
            "answer": f"{risk_type.capitalize()} - {explanation}",
            "distractors": ["Market risk - any loss is market risk",
                            "Operational risk - all unexpected losses",
                            "Systemic risk - the event affects the entire system"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"risk_type": risk_type, "scenario": scenario, "explanation": explanation}}

m_v1_qa_risk_types = wrap_mcq(v1_qa_risk_types)




def v1_fm_financial_crises(rng, seq):
    crises = [
        ("2008 Global Financial Crisis",
         "Subprime mortgage securitization unwind -> counterparty contagion -> interbank freeze -> global liquidity crisis.",
         "credit/market/liquidity contagion via securitization and leverage"),
        ("2010 Eurozone Debt Crisis",
         "Sovereign doubts about Greece -> Irish bank bailout -> sovereign-bank doom loop spreads.",
         "sovereign-bank feedback loop"),
        ("2020 COVID-19 Market Crash",
         "Pandemic -> 30% decline in 4 days -> extraordinary central bank intervention -> V-shaped recovery.",
         "exogenous shock with massive policy response"),
    ]
    era, desc, mech = rng.choice(crises)

    q = (f"Review the {era}: {desc}\n\n"
         f"What were the primary risk factors and risk-management failures involved?")

    steps = [
        ("Step 1.", 'Risk accumulation:\\n\n  - Asset quality deteriorated; leverage ratios increased.\\n\n  - Risk models on calm-period data underestimated tail correlation.\\n\n  - Long-dated assets funded by short-term wholesale funding.\\n'),
        ("Step 2.", 'Amplification channels:\\n\n  - Financial contagion via CDS, REPO, derivatives.\\n\n  - Margin cascades: MTM losses -> margin calls -> forced liquidations.\\n\n  - Procyclicality: VaR falls in calm periods -> more leverage -> fragility.\\n'),
        ("Step 3.", 'Risk-management failures:\\n\n  - VaR calibrated on aberrant calm data.\\n\n  - Stress tests not validated against crisis scenarios.\\n\n  - No countercyclical capital buffers (Basel pro-cyclicality).\\n\n  - Inadequate liquidity stress testing for withdrawal.\\n'),
        ("Step 4.", 'Regulatory response: Dodd-Frank, Basel III countercyclical buffers.\\n'),
        ("Step 5.", 'Key lesson: risk models must stress for known-unknowns. Tail is not pure Black Swan'),
    ]
    tr = render_trace(rng, ['Crisis framework: risk accumulation -> trigger -> amplification -> response'], steps)

    flaw = {"answer": ("The crisis was a pure Black Swan - "
                       "no risk model could have predicted it"),
            "pitfall": "Black Swan fallacy",
            "reasoning_trace": (f"Step 1. This ignores visible risk signals: leverage ratios, "
                                f"maturity mismatches, credit quality deterioration.\n"
                                f"Risk management's purpose is to stress for plausible-but-unlikely events.")}

    return {"meta": {"topic": "Market Risk", "subtopic": "Financial Crises",
                     "difficulty": "FRM1_Medium", "question_type": "Constructed Response",
                     "pitfalls": ["Black Swan fallacy", "procyclicality", "contagion channels"]},
            "question": q,
            "answer": ("Risk factors: deteriorating asset quality, leverage, maturity mismatch, "
                       "tail correlation underestimation. Failures: VaR miscalibration, no countercyclical "
                       "buffers, inadequate stress tests. Amplification via contagion and margin cascades."),
            "distractors": [],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"era": era, "desc": desc, "mech": mech}}

m_v1_fm_financial_crises = wrap_vignette(v1_fm_financial_crises)




def v1_qa_vega_greek(rng, seq):
    s = rng.randint(90, 115)
    k = rng.randint(90, 120)
    t = rng.choice([0.25, 0.5, 1.0])
    r = rng.uniform(0.03, 0.06, 4)
    sigma = rng.uniform(0.20, 0.40, 3)

    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    n_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vega_val = s * math.sqrt(t) * n_d1

    q = (f"European option: S = {s}, K = {k}, T = {t:.2f} yr, r = {pct(r,1)}, "
         f"sigma = {pct(sigma,0)}. Compute the option's vega - "
         f"dollar sensitivity to a 100 bps change in implied vol.")

    tr = (_assume(["Vega approx S * sqrt(T) * N'(d1)",
                   "N'(x) = (1/sqrt(2*pi)) * e^(-0.5 * x^2)"]) +
          f"Step 1. d1 = (ln({s}/{k}) + ({pct(r)} + 0.5 * {pct(sigma)}^2) * {t}) / "
          f"({pct(sigma)} * sqrt({t})) = {d1:.3f}.\n"
          f"Step 2. N'(d1) = (1/sqrt(2*pi)) * e^(-0.5 * {d1:.3f}^2) = {n_d1:.4f}.\n"
          f"Step 3. Vega (per unit vol) = {s} * sqrt({t}) * {n_d1:.4f} = {vega_val:.2f}.\n"
          f"Step 4. A 100 bps move in sigma changes option price by ~${vega_val:.2f}.\n"
          f"Step 5. Trap: using delta (direction risk) instead of vega (volatility risk).")

    flaw = {"answer": f"${s * r * t:.2f}", "pitfall": "confusing vega with time value S*r*T",
            "reasoning_trace": (f"Step 1. S * r * T = {fmt(s * r * t)} is rough time value, NOT vega.\n"
                                f"Vega = {vega_val:.2f} measures price sensitivity to sigma.\n"
                                f"Delta measures sensitivity to S. They hedge different risk dimensions.")}

    return {"meta": {"topic": "Options", "subtopic": "Greeks - Vega",
                     "difficulty": "FRM1_Medium", "question_type": "Calculation",
                     "pitfalls": ["d1 computation", "Gaussian PDF", "delta vs vega"]},
            "question": q,
            "answer": f"Vega = ${vega_val:.2f} per 100 bps vol move",
            "distractors": [f"Vega = ${vega_val * 100:.2f}",
                            f"Vega = ${s * r * t:.2f}",
                            f"Vega = ${s * sigma:.2f}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"s": s, "k": k, "t": t, "sigma": sigma, "d1": round(d1, 4),
                       "vega": round(vega_val, 2)}}

m_v1_qa_vega_greek = wrap_mcq(v1_qa_vega_greek)




def v1_qa_probability_stats(rng, seq):
    n_days = rng.choice([250, 500])
    alpha = rng.choice([0.05, 0.01])
    exceptions_obs = rng.randint(10, 30)
    obs_rate = exceptions_obs / n_days
    se = math.sqrt(0.01 * 0.99 / n_days)
    z_stat = (obs_rate - 0.01) / se if se > 0 else 0.0
    crit_val = 1.96 if alpha == 0.05 else 2.576
    reject = abs(z_stat) > crit_val

    q = (f"Risk manager tests 99% VaR model over {n_days} days. "
         f"{exceptions_obs} exceedances. Test H0: model correctly specified (alpha = {alpha:.0%}).")

    steps = [
        ("Step 1.", 'H0: true failure rate = 1%. H1: miscalibrated.\\n'),
        ("Step 2.", 'Observed rate = {exceptions_obs}/{n_days} = {obs_rate:.4f}.\\n'),
        ("Step 3.", 'SE = sqrt(0.01 * 0.99 / {n_days}) = {se:.4f}.\\n'),
        ("Step 4.", 'z = ({obs_rate:.4f} - 0.01) / {se:.4f} = {z_stat:.2f}.\\n'),
        ("Step 5.", 'Critical value (alpha={alpha:.0%}) = {crit_val:.2f}.\\n'),
        ("Step 6.", "|z|={abs(z_stat):.2f} {'>' if reject else '<='} {crit_val:.2f} -> \n{'Reject H0: miscalibrated.' if reject else 'Fail to reject H0: adequate.'}.\\n"),
        ("Step 7.", "Trap: treating 'fail to reject' as H0 is proven. Type II error risk persists"),
    ]
    tr = render_trace(rng, ['Two-sided test: reject H0 if |z_obs| > z_crit'], steps, conclusion="Trap: treating 'fail to reject' as H0 is proven. Type II error risk persists")

    flaw = {"answer": "Fail to reject H0: the model is confirmed correct",
            "pitfall": "treating fail-to-reject as proof",
            "reasoning_trace": (f"Step 1. Fail to reject != H0 is true. It means insufficient evidence against H0.\n"
                                f"The Type II error risk (mispriced model) remains significant.")}

    return {"meta": {"topic": "Quantitative Methods", "subtopic": "Probability and Statistics",
                     "difficulty": "FRM1_Medium", "question_type": "Calculation",
                     "pitfalls": ["Type II error", "p-value vs significance", "fail to reject error"]},
            "question": q,
            "answer": (f"z = {z_stat:.2f}, crit = {crit_val:.2f}. "
                       f"{'Reject H0: miscalibrated at ' if reject else 'Fail to reject: adequate at '}alpha={alpha}."),
            "distractors": ["Accept H0: model confirmed correct",
                            f"z = {z_stat * 2:.2f}",
                            "p-value always equals alpha"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"n_days": n_days, "exceptions": exceptions_obs,
                       "z_stat": round(z_stat, 2), "alpha": alpha, "reject": reject}}

m_v1_qa_probability_stats = wrap_mcq(v1_qa_probability_stats)


TEMPLATES = {

    "mkt_param_var": mkt_param_var, "mkt_cvar": mkt_cvar, "greek_delta_hedge": greek_delta_hedge,
    "bond_var_duration": bond_var_duration, "reg_beta_corr": reg_beta_corr, "bsm_put_parity": bsm_put_parity,
    "mkt_historical_var": mkt_historical_var,
    "fut_forward_price": fut_forward_price,
    "m_mkt_historical_var": m_mkt_historical_var,
    "m_fut_forward_price": m_fut_forward_price,
    # NEW: time series, simulation, probability, corporate bonds, mortgages, swaps
    "v1_qa_time_series": v1_qa_time_series, "v1_qa_simulation": v1_qa_simulation,
    "v1_qa_prob_stats": v1_qa_prob_stats,
    "v1_fm_corp_bond": v1_fm_corp_bond, "v1_fm_mortgage": v1_fm_mortgage, "v1_fm_swap": v1_fm_swap,
    "m_v1_qa_time_series": m_v1_qa_time_series, "m_v1_qa_simulation": m_v1_qa_simulation,
    "m_v1_qa_prob_stats": m_v1_qa_prob_stats,
    "m_v1_fm_corp_bond": m_v1_fm_corp_bond, "m_v1_fm_mortgage": m_v1_fm_mortgage,
    "m_v1_fm_swap": m_v1_fm_swap,
    "v1_qa_ethics_time_series": v1_qa_ethics_time_series,
    "m_v1_qa_ethics_time_series": m_v1_qa_ethics_time_series,
    "v1_qa_concept_swap_valuation": v1_qa_concept_swap_valuation,
    "m_v1_qa_concept_swap_valuation": m_v1_qa_concept_swap_valuation,

    "new_v1_mkt_backtesting": v1_mkt_backtesting,
    "new_v1_qa_var_basics": v1_qa_var_basics,
    "new_v1_qa_risk_types": v1_qa_risk_types,
    "new_v1_fm_financial_crises": v1_fm_financial_crises,
    "new_v1_qa_vega_greek": v1_qa_vega_greek,
    "new_v1_qa_probability_stats": v1_qa_probability_stats,


}
