"""
FRM Part 2 templates (credit, operational, liquidity, Basel).
"""
import math
from pipelines.core import fmt, pct, render_trace
from pipelines.templates.wrappers import wrap_mcq, wrap_cr, wrap_vignette

PROG = "FRM_Part_2"

def _assume(a):
    return "ASSUMPTIONS: " + "; ".join(a) + ".\n"

def credit_el(rng, seq):
    pd = rng.uniform(0.01, 0.05, 4)
    lgd = rng.uniform(0.30, 0.60, 3)
    ead = rng.randint(50, 200) * 1000000
    el = pd*lgd*ead
    q = (f"Credit exposure (EAD) {fmt(ead)}, PD {pct(pd,1)}, LGD {pct(lgd,1)}. "
         f"Compute expected loss = PD × LGD × EAD.")
    steps = [
        ("Step 1.", 'PD×LGD = {pct(pd,1)}×{pct(lgd,1)} = {pct(pd*lgd,1)}.\\n'),
        ("Step 2.", 'EL = {pct(pd*lgd,1)}×{fmt(ead)} = {fmt(el)}.\\n'),
        ("Step 3.", 'Trap: using EAD without LGD ({fmt(pd*ead)}) overstates expected loss'),
    ]
    tr = render_trace(rng, ['EL = PD × LGD × EAD'], steps, conclusion='Trap: using EAD without LGD ({fmt(pd*ead)}) overstates expected loss')
    flaw = {"answer": f"{fmt(pd*ead)}", "pitfall": "omitting LGD",
            "reasoning_trace": render_trace(rng, ['EL = PD × EAD'], [
                ("Step 1.", 'Multiplying PD {pct(pd,1)} by EAD {fmt(ead)} = {fmt(pd*ead)} skips the loss-given-\ndefault {pct(lgd,1)}; EL = PD×LGD×EAD = {pct(pd,1)}×{pct(lgd,1)}×{fmt(ead)} = {fmt(el)}'),
            ]),
                   }
    return {"meta":{"topic":"Credit Risk","subtopic":"Expected Loss","difficulty":"FRM2_Medium",
                    "question_type":"Calculation","pitfalls":["PD, LGD, EAD","loss given default"]},
            "question":q, "answer":f"{fmt(el)}", "distractors":[f"{fmt(pd*ead)}", f"{fmt(ead*lgd)}", f"{fmt(el*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"pd":pd,"lgd":lgd,"ead":ead}}

def credit_var_vasicek(rng, seq):
    # Vasicek one-factor: simple conditional default prob
    rho = rng.uniform(0.10, 0.30, 3)
    pd = rng.uniform(0.02, 0.06, 4)
    import math as _m
    # threshold
    th = _m.erf((2*pd-1)/_m.sqrt(2))  # approx N^-1 via erf
    q = (f"Vasicek: asset correlation ρ = {rho:.2f}, unconditional PD = {pct(pd,1)}. "
         f"Explain how systemic factor raises tail default; give conditional PD for a −2σ systemic factor "
         f"(approx).")
    # conditional PD approx via Vasicek formula (simplified)
    cpd = _m.erf((th - rho*2)/(_m.sqrt(1-rho**2)*_m.sqrt(2)))*0.5+0.5
    steps = [
        ("Step 1.", 'N⁻¹(PD) ≈ {th:.3f} (from {pct(pd,1)}).\\n'),
        ("Step 2.", 'Systemic factor Z = −2 (tail stress): numerator {th:.3f} − {rho:.2f}×(−2) = {th + 2*rho:.3f}.\\n'),
        ("Step 3.", 'Denominator √(1−ρ²) = √(1−{rho:.2f}²) = {_m.sqrt(1-rho**2):.3f}.\\n'),
        ("Step 4.", 'Conditional PD = {pct(cpd)} vs unconditional {pct(pd,1)} — the systemic tail multiplies default probability.\\n'),
        ("Step 5.", 'Trap: quoting unconditional PD ignores the correlation-driven systemic tail'),
    ]
    tr = render_trace(rng, ['Vasicek conditional PD = N[(N⁻¹(PD) − ρ·Z)/√(1−ρ²)]'], steps, conclusion='Trap: quoting unconditional PD ignores the correlation-driven systemic tail')
    flaw = {"answer": f"{pct(pd)}", "pitfall": "unconditional vs conditional PD",
            "reasoning_trace": render_trace(rng, ['using unconditional PD'], [
                ("Step 1.", 'Reporting unconditional PD {pct(pd,1)} in a −2σ systemic stress ignores asset \ncorrelation {rho:.2f}; Vasicek conditional PD rises to {pct(cpd)}'),
            ]),
                   }
    return {"meta":{"topic":"Credit Risk","subtopic":"Vasicek Model","difficulty":"FRM2_Hard",
                    "question_type":"Calculation","pitfalls":["unconditional vs conditional","correlation"]},
            "question":q, "answer":f"conditional PD {pct(cpd)}", "distractors":[f"{pct(pd)}", f"{pct(cpd*2)}", f"{pct(1-cpd)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"rho":rho,"pd":pd}}

def op_risk_basel(rng, seq):
    alpha = rng.choice([0.10, 0.12, 0.15])
    gross_income = rng.randint(20, 40) * 1000000
    op_charge = alpha*gross_income*12  # annualized
    q = (f"Basel II Basic Indicator Approach: alpha = {alpha:.2f}, average gross income {fmt(gross_income)} "
         f"over 3 yrs. Compute the operational risk capital charge = α × gross income.")
    steps = [
        ("Step 1.", 'Charge = {alpha:.2f}×{fmt(gross_income)} = {fmt(op_charge)} per year.\\n'),
        ("Step 2.", 'Trap: using total income over the 3-yr window (not the average) overstates the charge'),
    ]
    tr = render_trace(rng, ['BIA charge = α × gross income (annualized)'], steps, conclusion='Trap: using total income over the 3-yr window (not the average) overstates the charge')
    flaw = {"answer": f"{fmt(alpha*gross_income*3)}", "pitfall": "average vs total income",
            "reasoning_trace": render_trace(rng, ['total 3-yr income'], [
                ("Step 1.", 'Multiplying 3-yr total income × alpha = {fmt(alpha*gross_income*3)} instead of \nthe AVERAGE gross income × alpha = {fmt(op_charge)}. The BIA uses average gross income over 3 years'),
            ]),
                   }
    return {"meta":{"topic":"Operational Risk","subtopic":"Basel Approaches","difficulty":"FRM2_Medium",
                    "question_type":"Calculation","pitfalls":["average gross income","BIA formula"]},
            "question":q, "answer":f"{fmt(op_charge)}", "distractors":[f"{fmt(alpha*gross_income*3)}", f"{fmt(gross_income)}", f"{fmt(op_charge*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"alpha":alpha,"gross_income":gross_income}}

def liq_lcad(rng, seq):
    # liquidity coverage ratio
    hqla = rng.randint(100, 200) * 1000000
    outflow = rng.randint(80, 150) * 1000000
    inflow = rng.randint(20, 40) * 1000000
    n_outflow = outflow - inflow
    lcr = hqla/n_outflow
    q = (f"LCR: HQLA {fmt(hqla)}, 30-day gross outflows {fmt(outflow)}, inflows {fmt(inflow)}. "
         f"Compute LCR = HQLA / (outflows − inflows).")
    steps = [
        ("Step 1.", 'Net outflows = {fmt(outflow)} − {fmt(inflow)} = {fmt(n_outflow)}.\\n'),
        ("Step 2.", 'LCR = {fmt(hqla)}/{fmt(n_outflow)} = {lcr:.2f} ({pct(lcr)}).\\n'),
        ("Step 3.", 'Trap: dividing by gross outflows (no inflow offset) understates the ratio'),
    ]
    tr = render_trace(rng, ['LCR = HQLA / net cash outflows'], steps, conclusion='Trap: dividing by gross outflows (no inflow offset) understates the ratio')
    flaw = {"answer": f"{fmt(hqla/outflow)}", "pitfall": "net vs gross outflows",
            "reasoning_trace": render_trace(rng, ['gross outflows'], [
                ("Step 1.", 'LCR = HQLA/gross outflows = {fmt(hqla)}/{fmt(outflow)} = {fmt(hqla/outflow)} \ninstead of HQLA/net outflows = {fmt(hqla)}/{fmt(n_outflow)} = {lcr:.2f}. \nBasel nets inflows against outflows in the denominator'),
            ]),
                   }
    return {"meta":{"topic":"Liquidity Risk","subtopic":"LCR","difficulty":"FRM2_Medium",
                    "question_type":"Calculation","pitfalls":["net vs gross outflows","LCR definition"]},
            "question":q, "answer":f"LCR {lcr:.2f}", "distractors":[f"{fmt(hqla/outflow)}", f"{fmt(lcr*2)}", f"{fmt(hqla)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"hqla":hqla,"outflow":outflow,"inflow":inflow}}

def liq_ws_avg(rng, seq):
    # weighted average life / cost of funds
    amt1 = rng.randint(10, 30) * 1000000
    rate1 = rng.choice([0.02, 0.03, 0.04])
    amt2 = rng.randint(20, 50) * 1000000
    rate2 = rng.choice([0.04, 0.05, 0.06])
    total = amt1+amt2
    avg_cost = (amt1*rate1 + amt2*rate2)/total
    q = (f"Funding: {fmt(amt1)} at {pct(rate1,1)} and {fmt(amt2)} at {pct(rate2,1)}. "
         f"Compute the weighted average cost of funds.")
    steps = [
        ("Step 1.", 'Total = {fmt(amt1)} + {fmt(amt2)} = {fmt(total)}.\\n'),
        ("Step 2.", 'Weighted cost = ({fmt(amt1)}×{pct(rate1,1)} + {fmt(amt2)}×{pct(rate2,1)})/{fmt(total)} = {pct(avg_cost)}.\\n'),
        ("Step 3.", 'Trap: unweighted average ({pct((rate1+rate2)/2)}) misstates funding cost when amounts differ'),
    ]
    tr = render_trace(rng, ['WACF = Σ(amount×rate)/Σ amount'], steps, conclusion='Trap: unweighted average ({pct((rate1+rate2)/2)}) misstates funding cost when amounts differ')
    flaw = {"answer": f"{fmt(amt1*rate1+amt2*rate2)}", "pitfall": "weighted vs simple average",
            "reasoning_trace": render_trace(rng, ['simple average'], [
                ("Step 1.", 'Averaging rates {pct(rate1,1)} and {pct(rate2,1)} = {pct((rate1+rate2)/2)} \nignores the amounts {fmt(amt1)} vs {fmt(amt2)}; weighted cost = {pct(avg_cost)}'),
            ]),
                   }
    return {"meta":{"topic":"Liquidity Risk","subtopic":"Funding Cost","difficulty":"FRM2_Easy",
                    "question_type":"Calculation","pitfalls":["weighted average","amount weighting"]},
            "question":q, "answer":f"{pct(avg_cost)}", "distractors":[f"{pct((rate1+rate2)/2)}", f"{pct(rate1)}", f"{pct(avg_cost*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"amt1":amt1,"rate1":rate1,"amt2":amt2,"rate2":rate2}}


def credit_var(rng, seq):
    pv = rng.randint(100, 200) * 1000
    pd = rng.uniform(0.01, 0.03, 4)
    lgd = rng.choice([0.5, 0.6, 0.7])
    var99 = pv * pd * lgd * 2.326
    q = (f"Loan {fmt(pv)}, default probability {pct(pd)}, loss-given-default {lgd:.0%}. "
         f"Compute 1-day 99% credit VaR.")
    steps = [
        ("Step 1.", 'CVaR = {fmt(pv)} × {pct(pd)} × {lgd:.0%} × 2.326 = {fmt(var99)}.\\n'),
        ("Step 2.", 'Trap: dropping LGD gives {fmt(pv*pd*2.326)}'),
    ]
    tr = render_trace(rng, ['credit VaR ≈ V × PD × LGD × z(99%)'], steps, conclusion='Trap: dropping LGD gives {fmt(pv*pd*2.326)}')
    flaw = {"answer": f"{fmt(pv*pd*2.326)}", "pitfall": "LGD omitted",
            "reasoning_trace": render_trace(rng, ['LGD omitted'], [
                ("Step 1.", 'CVaR = {fmt(pv)} × {pct(pd)} × 2.326 = {fmt(pv*pd*2.326)}'),
            ]),
                   }
    return {"meta": {"topic":"Credit Risk","subtopic":"Credit VaR","difficulty":"FRM2_Medium",
                     "question_type":"Calculation","pitfalls":["LGD omitted"]},
            "question":q, "answer":f"{fmt(var99)}",
            "distractors":[f"{fmt(pv*pd*2.326)}", f"{fmt(pv*pd*lgd)}", f"{fmt(pv*pd)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"pv":pv,"pd":pd,"lgd":lgd}}

m_credit_var = wrap_mcq(credit_var)


def v2_cr_counterparty(rng, seq):
    exp = rng.randint(100, 300) * 1000000
    pd_cp = rng.uniform(0.01, 0.04, 3)
    lgd_cp = rng.uniform(0.40, 0.60, 3)
    r = rng.uniform(0.02, 0.04, 4)
    t = rng.choice([3.0, 5.0, 7.0])
    cva = exp * (1 - lgd_cp) * pd_cp * math.exp(-r * t)
    steps = [
        ("Step 4.", 'DVA is the symmetric amount when we are the counterparty: DVA = our_PD × our_LGD × E × discount.\\n'),
        ("Step 5.", 'Wrong-way risk: correlation between E and PD pushes actual CVA above the calculated value'),
    ]
    tr = render_trace(rng, ['CVA = E × (1 − R) × PD × exp(−rT)'], steps)
    flaw = {"answer": f"{fmt(exp * pd_cp)}",
            "pitfall": "omitting discount factor and (1−R)",
            "reasoning_trace": (_assume(["CVA = E × PD"]) +
                                f"CVA = {fmt(exp)} × {pct(pd_cp)} = {fmt(exp * pd_cp)} "
                                f"omits loss-given-default (1 − {pct(lgd_cp)}) and the discount factor {math.exp(-r * t):.4f}.\n"
                                f"Correct: CVA = {fmt(cva)}.")
            }
    return {"meta": {"topic": "Credit Risk", "subtopic": "CVA / DVA", "difficulty": "v2_FRM2_Hard",
                     "question_type": "Calculation",
                     "pitfalls": ["CVA formula", "discount factor", "wrong-way risk"]},
            "question": (f"A derivative has expected exposure ({100 * lgd_cp:.0f}% LGD applied) of {fmt(exp)}, "
                         f"counterparty PD = {pct(pd_cp)}, risk-free rate = {pct(r)}, maturity = {t} years. "
                         f"Compute CVA (continuous discounting) and briefly state how DVA differs."),
            "answer": f"CVA {fmt(cva)}",
            "distractors": [f"{fmt(exp * pd_cp)}", f"{fmt(exp * (1 - lgd_cp) * pd_cp)}",
                            f"{fmt(cva * 1.5)}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"exp": exp, "pd": pd_cp, "lgd": lgd_cp, "r": r, "t": t}
            }


def v2_cr_securitization(rng, seq):
    # Three-notch CMO waterfall: compute interest + principal distribution
    a_princ = rng.randint(40, 70) * 1000000
    b_princ = rng.randint(30, 50) * 1000000
    c_princ = rng.randint(20, 40) * 1000000
    a_cpn = rng.choice([0.03, 0.035, 0.04])
    b_cpn = rng.choice([0.05, 0.055, 0.06])
    c_cpn = rng.choice([0.08, 0.09, 0.10])
    avail_pool = rng.randint(20, 30) * 1000000

    a_int = a_princ * a_cpn
    b_int = b_princ * b_cpn
    c_int = c_princ * c_cpn
    total_int = a_int + b_int + c_int
    avail_principal = avail_pool - total_int

    # Waterfall distribution
    if avail_principal <= a_int * 0:  # no principal if interest not fully covered
        a_paid = b_paid = c_paid = 0
    else:
        pool = avail_principal
        a_paid = min(a_princ, pool) if avail_principal > 0 else 0
        if avail_principal > 0:
            pool -= a_paid
            b_paid = min(b_princ, pool) if avail_principal > 0 else 0
            if avail_principal > 0:
                pool -= b_paid
                c_paid = min(c_princ, pool) if avail_principal > 0 else 0

    wt_yld = (a_princ * a_cpn + b_princ * b_cpn + c_princ * c_cpn) / (a_princ + b_princ + c_princ)
    steps = [
        ("Step 1.", 'Compute interest per tranche:\\n\n  A_int = {fmt(a_princ)} × {pct(a_cpn)} = {fmt(a_int)}\\n\n  B_int = {fmt(b_princ)} × {pct(b_cpn)} = {fmt(b_int)}\\n\n  C_int = {fmt(c_princ)} × {pct(c_cpn)} = {fmt(c_int)}\\n\n  Total interest = {fmt(total_int)}\\n\\n'),
        ("Step 2.", 'Available for principal = {fmt(avail_pool)} − {fmt(total_int)} = {fmt(avail_principal)}\\n\\n'),
        ("Step 3.", 'Waterfall: A gets {fmt(a_paid)}, B gets {fmt(b_paid)}, C gets {fmt(c_paid)}\\n\\n'),
        ("Step 4.", 'The portfolio weighted-yield = {pct(wt_yld)}.\\n'),
        ("Step 5.", 'Trap: allocating principal pro-rata across tranches ignores the legal waterfall'),
    ]
    tr = render_trace(rng, ['Waterfall: interest first, then principal down the seniority chain'], steps, conclusion='Trap: allocating principal pro-rata across tranches ignores the legal waterfall')
    flaw = {"answer": f"A:{fmt(b_paid)}, B:{fmt(c_paid)}, C:{fmt(avail_principal)}",
            "pitfall": "pro-rata principal allocation",
            "reasoning_trace": (_assume(["pro-rata distribution"]) +
                                f"Allocating the available principal {fmt(avail_principal)} pro-rata by tranche size ignores "
                                f"the legal waterfall that requires interest to be paid in full to all tranches before any principal "
                                f"flows down. Correct: A={fmt(a_paid)}, B={fmt(b_paid)}, C={fmt(c_paid)}.")
            }
    return {"meta": {"topic": "Credit Risk", "subtopic": "Securitization", "difficulty": "v2_FRM2_Hard",
                     "question_type": "Calculation",
                     "pitfalls": ["waterfall structure", "seniority", "pro-rata trap"]},
            "question": (f"A CMO has three sequential tranches: A ({fmt(a_princ)}, {pct(a_cpn)}), "
                         f"B ({fmt(b_princ)}, {pct(b_cpn)}), C ({fmt(c_princ)}, {pct(c_cpn)}). "
                         f"In a given period the pool provides {fmt(avail_pool)}. "
                         f"Compute each tranche's interest and principal payment under the waterfall."),
            "answer": f"A_int:{fmt(a_int)} A_principal:{fmt(a_paid)} B_int:{fmt(b_int)} B_principal:{fmt(b_paid)} C_int:{fmt(c_int)} C_principal:{fmt(c_paid)}",
            "distractors": [f"A:{fmt(b_paid)} B:{fmt(c_paid)} C:{fmt(avail_principal)}",
                            f"A_total:{fmt(a_int + a_paid)} B_total:{fmt(b_int + b_paid)} C_total:{fmt(c_int + c_paid)}",
                            f"pro-rata A:{fmt(avail_principal * a_princ / (a_princ + b_princ + c_princ))}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"a": a_princ, "b": b_princ, "c": c_princ, "avail": avail_pool}
            }


def v2_op_loss_dist(rng, seq):
    # Poisson(λ) frequency × lognormal(μ,σ) severity
    lam = rng.uniform(5, 12, 1)
    mu_s = rng.uniform(3.5, 4.5, 2)  # ln(mean severity)
    sigma_s = rng.uniform(0.5, 1.2, 2)
    # Expected loss
    exp_loss = lam * math.exp(mu_s + 0.5 * sigma_s**2)
    std_sev = math.sqrt((math.exp(sigma_s**2) - 1) * math.exp(2 * mu_s + sigma_s**2))
    std_loss = math.sqrt(lam) * std_sev
    # Gaussian approx: 99.5th percentile = exp_loss + 2.326 * std_loss
    var995 = exp_loss + 2.326 * std_loss
    eco_cap = var995 - exp_loss
    steps = [
        ("Step 1.", 'Expected loss = λ × E[X] = {lam:.1f} × {fmt(math.exp(mu_s + 0.5 * sigma_s**2))} = {fmt(exp_loss)}\\n'),
        ("Step 2.", 'Std loss = √λ × σ = {math.sqrt(lam):.2f} × {fmt(std_sev)} = {fmt(std_loss)}\\n'),
        ("Step 3.", '99.5th-percentile ≈ {fmt(exp_loss)} + 2.326 × {fmt(std_loss)} = {fmt(var995)}\\n'),
        ("Step 4.", 'Economic capital = {fmt(var995)} − {fmt(exp_loss)} = {fmt(eco_cap)}\\n\\n'),
        ("Step 5.", 'Trap: quoting 99.5th percentile VaR itself as economic capital. \nEC = VaR − EL; VaR already includes expected loss'),
    ]
    tr = render_trace(rng, ['Severity ~ lognormal(μ,σ); Loss L = Σ Xi ≈ Normal(λ·m, λ·s²)'], steps, conclusion='Trap: quoting 99.5th percentile VaR itself as economic capital. \nEC = VaR − EL; VaR already includes expected loss')
    flaw = {"answer": f"{fmt(var995)}",
            "pitfall": "quoting VaR instead of EC (VaR − EL)",
            "reasoning_trace": render_trace(rng, ['EC = 99.5th %ile'], [
                ("Step 1.", 'Quoting the 99.5% VaR = {fmt(var995)} as economic capital is incorrect. \nEconomic capital = VaR − EL = {fmt(var995)} − {fmt(exp_loss)} = {fmt(eco_cap)}. \nExpected loss is reserved for; EC covers unexpected loss only'),
            ]),
                   }
    return {"meta": {"topic": "Operational Risk", "subtopic": "Loss Distribution",
                     "difficulty": "v2_FRM2_Hard",
                     "question_type": "Calculation",
                     "pitfalls": ["EC = VaR − EL", "lognormal moments", "Poisson compound"]},
            "question": (f"Operational losses follow Poisson(λ = {lam:.1f}) frequency × "
                         f"lognormal(μ = {mu_s:.2f}, σ = {sigma_s:.2f}) severity. "
                         f"Compute EL, economic capital at 99.5 %ile (Gaussian approx)."),
            "answer": f"EL = {fmt(exp_loss)}; EC = {fmt(eco_cap)}",
            "distractors": [f"EL = {fmt(exp_loss)}; EC = {fmt(var995)}",
                            f"EL = {fmt(lam * math.exp(mu_s))}; EC = {fmt(var995 * 0.5)}",
                            f"EL = {fmt(std_loss)}; EC = {fmt(std_loss * 2.326)}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"lambda": lam, "mu": mu_s, "sigma": sigma_s}
            }


def v2_op_model_risk(rng, seq):
    # Conceptual / CR format
    model_type = rng.choice([
        "A volatility-forecasting model systematically underestimates realized volatility during crisis periods with short (60-day) lookback windows.",
        "An internal credit-rating model yields significantly higher ratings than external agency ratings, especially for BBB-grade issuers.",
        "An AML screening model produces excessive false-positive alerts, causing operational inefficiency while missing true matches in certain non-English name variations."
    ])
    issues = {
        "A volatility-forecasting model systematically underestimates realized volatility during crisis periods with short (60-day) lookback windows.":
            ["Conceptual model risk: short-window GARCH or EWMA ignores volatility clustering and long-memory effects.\n"
             "Implementation risk: the lookback was calibrated on normal-market data; parameters were never stress-tested.\n"
             "Outcome risk: backtesting shows systematic under-prediction; VaR coverage ratio falls below the 95 % benchmark.\n"
             "\nMitigation: widen lookback window; use GARCH-family or stochastic-volatility model; implement backtesting regime; document model limitations in the model inventory."],
        "An internal credit-rating model yields significantly higher ratings than external agency ratings, especially for BBB-grade issuers.":
            ["Conceptual model risk: training data is biased toward internal portfolio that may not reflect economy-wide cyclicality.\n"
             "Implementation risk: missing macroeconomic variables in the scorecard; no calibration-to-default data.\n"
             "Outcome risk: rating migration model fails IFRS 9 stage-transfer tests; expected credit losses understated.\n"
             "\nMitigation: rebalance the training sample; incorporate external default data; validate back-to-back against agency grades on a hold-out set."],
        "An AML screening model produces excessive false-positive alerts, causing operational inefficiency while missing true matches in certain non-English name variations.":
            ["Conceptual model risk: string-matching or simple fuzzy-algorithm cannot capture phonetic variations across languages.\n"
             "Implementation risk: no continuous retraining on newly-sanctioned entity lists; outdated name dictionaries.\n"
             "Outcome risk: SAR filing delay increases regulatory enforcement risk; false-positive volume exhausts analyst capacity.\n"
             "\nMitigation: upgrade to NLP-based name matching; implement automated feedback loops; establish periodic model validation against OFAC/UN lists."]
    }
    q = (f"{_assume([_s for _s in [model_type]])} \n"
         f"A quantitative analyst discovers: {model_type}\n\n"
         f"Q) Identify at least three model-risk types present. "
         f"What would your model-validation process look like?")
    tr = (f"Issue observed:\n  {model_type}\n\n"
          "Model-validation process:\n"
          f"  1. Independent validation — challenge the model developer; document assumptions.\n"
          "  2. Benchmarking — compare to industry-standard or external model results.\n"
          "  3. Backtesting — check predictive accuracy using out-of-sample data.\n"
          "  4. Decision-independence — ensure validator has veto authority.\n"
          "  5. Documentation — feed findings back into the model inventory and remediation plan.")
    flaw = {"answer": "Only conceptual and implementation risk; outcome risk is not a model-risk.",
            "pitfall": "missing outcome risk as a distinct model risk type",
            "reasoning_trace": (_assume(["only 2 model risk types"]) +
                                "Model risk includes: (I) conceptual risk, (II) implementation risk, "
                                "(III) outcome risk (validated by backtesting, benchmarking, and P&L attribution). "
                                "Outcome risk is confirmed when actual model outputs deviate from observed values.")
                       }
    return {"meta": {"topic": "Operational Risk", "subtopic": "Model Risk", "difficulty": "v2_FRM2_Medium",
                     "question_type": "Constructed Response",
                     "pitfalls": ["conceptual model risk", "implementation risk", "outcome risk"]},
            "question": q, "answer": "conceptual risk; implementation risk; outcome risk",
            "distractors": [],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"model_type": model_type}
            }


def v2_liq_nsfr(rng, seq):
    # Net Stable Funding Ratio
    eq_capital = rng.randint(40, 80) * 1000000
    long_debt = rng.randint(40, 70) * 1000000

    total_assets = rng.randint(250, 400) * 1000000
    cash_reserves = rng.randint(30, 60) * 1000000
    high_quality_bonds = rng.randint(20, 40) * 1000000
    corporate_loans_5y = rng.randint(40, 80) * 1000000
    illiquid_investments = rng.randint(15, 30) * 1000000
    real_estate = rng.randint(20, 40) * 1000000
    other_assets = total_assets - cash_reserves - high_quality_bonds - corporate_loans_5y - illiquid_investments - real_estate

    # NSF requirements by asset: [cash=0%, HQLA=5%, corporate=50%, illiquid=85%, RE=65%, other=90%]
    asr = (cash_reserves * 0.0 +
           high_quality_bonds * 0.05 +
           corporate_loans_5y * 0.50 +
           illiquid_investments * 0.85 +
           real_estate * 0.65 +
           other_assets * 0.90)

    # NSF sources: equity 95%, long debt (M>1y) 90%, short wholesale 70%, customer deposit 90%
    customer_deposits = int(total_assets * 0.45)
    short_wholesale = int(total_assets * 0.10)
    other_liabilities = eq_capital + long_debt + customer_deposits + short_wholesale - total_assets

    asf = (eq_capital * 0.95 +
           long_debt * 0.90 +
           customer_deposits * 0.90 +
           short_wholesale * 0.70 +
           other_liabilities * 0.50)

    nsfr = asf / asr

    tr = (_assume(["NSFR = ASN / ASR, must be ≥ 1.0"]) +
          f"NSF Sources (ASN):\n"
          f"  Equity       ({fmt(eq_capital)} × 0.95)   = {fmt(eq_capital * 0.95)}\n"
          f"  Long debt    ({fmt(long_debt)} × 0.90)   = {fmt(long_debt * 0.90)}\n"
          f"  Customer dep. ({fmt(customer_deposits)} × 0.90) = {fmt(customer_deposits * 0.90)}\n"
          f"  Short whsl.  ({fmt(short_wholesale)} × 0.70)   = {fmt(short_wholesale * 0.70)}\n"
          f"  Other liab.  ({fmt(other_liabilities)} × 0.50) = {fmt(other_liabilities * 0.50)}\n"
          f"  TOTAL ASN = {fmt(asf)}\n\n"
          f"NSF Requirements (ASR):\n"
          f"  Cash reserves          ({fmt(cash_reserves)} × 0.00) = {fmt(cash_reserves * 0.00)}\n"
          f"  High-quality bonds     ({fmt(high_quality_bonds)} × 0.05) = {fmt(high_quality_bonds * 0.05)}\n"
          f"  Corporate loans (5y)   ({fmt(corporate_loans_5y)} × 0.50) = {fmt(corporate_loans_5y * 0.50)}\n"
          f"  Illiquid investments   ({fmt(illiquid_investments)} × 0.85) = {fmt(illiquid_investments * 0.85)}\n"
f"  Real estate            ({fmt(real_estate)} × 0.65) = {fmt(real_estate * 0.65)}\n"
           f"  Other assets           ({fmt(other_assets)} × 0.90) = {fmt(other_assets * 0.90)}\n"
          f"  TOTAL ASR = {fmt(asr)}\n\n"
          f"NSFR = ASN / ASR = {fmt(asf)} / {fmt(asr)} = {nsfr:.2f}\n"
          f"For compliance, NSFR must be ≥ 1.00 — the bank is {'compliant' if nsfr >= 1 else 'non-compliant'}.")
    flaw = {"answer": f"NSFR {fmt(asr / asf)}",
            "pitfall": "NSFR = ASR / ASN (reversed ratio)",
            "reasoning_trace": (_assume(["NSFR = ASR / ASN"]) +
                                f"NSFR = ASN / ASR, NOT ASR / ASN. "
                                f"The ratio {fmt(asr / asf)} inverts numerator and denominator.\n"
                                f"Correct: NSFR = {fmt(asf)} / {fmt(asr)} = {nsfr:.2f}.")
                       }
    return {"meta": {"topic": "Liquidity Risk", "subtopic": "NSFR", "difficulty": "v2_FRM2_Hard",
                     "question_type": "Calculation",
                     "pitfalls": ["NSFR = ASN/ASR", "ASF/ASR weights", "compliance threshold"]},
            "question": (f"Based on the balance sheet below, compute the Net Stable Funding Ratio (NSFR).\n\n"
                         f"Assets:\nCash: {fmt(cash_reserves)}  HQLA bonds: {fmt(high_quality_bonds)}  "
                         f"Corporatel 5y: {fmt(corporate_loans_5y)}  Illiquid: {fmt(illiquid_investments)}  "
                         f"Real estate: {fmt(real_estate)}  Other: {fmt(other_assets)}\n\n"
                         f"Liabilities & Equity:\n"
                         f"Equity: {fmt(eq_capital)}  Long debt (>1y): {fmt(long_debt)}  "
                         f"Customer deposits: {fmt(customer_deposits)}  "
                         f"Short wholesale: {fmt(short_wholesale)}  Other: {fmt(other_liabilities)}"),
            "answer": f"NSFR {nsfr:.2f}",
            "distractors": [f"NSFR {fmt(asr / asf)}", f"NSFR {fmt(asf / asr)}", f"NSFR {fmt(asf * 2 / asr)}"],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"assets": asr, "liabilities": asf, "nsfr": nsfr}
            }


def v2_ci_climate(rng, seq):
    # Climate transition risk: carbon pricing impact (CR format)
    carbon_intensity = rng.choice(["1.2 MtCO₂ (steel/iron)", "0.8 MtCO₂ (cement)", "0.5 MtCO₂ (power generation)"])
    carbon_price = rng.randint(30, 60)
    annual_emissions_mtc = None
    if "steel" in carbon_intensity:
        annual_emissions_mtc = round(1.2e6 * rng.uniform(0.9, 1.1, 2), 6)
    elif "cement" in carbon_intensity:
        annual_emissions_mtc = round(0.8e6 * rng.uniform(0.9, 1.1, 2), 6)
    else:
        annual_emissions_mtc = round(0.5e6 * rng.uniform(0.9, 1.1, 2), 6)

    annual_cost = annual_emissions_mtc * carbon_price
    npv_cost_annual = annual_cost / 0.03 if rng.randint(0, 1) else annual_cost / 0.05

    tr = (_assume(["Carbon price: $X/ton CO₂; Emissions: Y tons/year"]) +
          f"Emissions: {annual_emissions_mtc:,.0f} tCO₂/yr\n"
          f"Carbon price: ${carbon_price}/tCO₂\n\n"
          f"Annual compliance cost = {annual_emissions_mtc:,.0f} × {carbon_price} = {fmt(annual_cost)}\n\n"
          "Transition channels:\n"
          "  1. Cost channel — carbon price increases operating costs.\n"
          "  2. Demand channel — consumer preference shifts away from high-carbon products.\n"
          "  3. Valuation channel — stranded-asset risk: discounting the perpetual compliance cost "
          f"(PV at {3 if annual_emissions_mtc > 5e5 else 5} %)."
          f"\n  NPV of annual cost = {fmt(annual_cost)} / {3 if annual_emissions_mtc > 5e5 else 5} % = {fmt(npv_cost_annual)}.\n\n"
          "Mitigation: invest in low-carbon technology; take carbon offsets; hedges under EU ETS.")
    flaw = {"answer": f"Annual cost ${fmt(annual_cost * 2)}",
            "pitfall": "doubling the carbon price impact (applying twice)",
            "reasoning_trace": (_assume(["applying carbon tax twice"]) +
                                f"Cost = emissions × carbon price = {annual_emissions_mtc:,.0f} × {carbon_price} = ${fmt(annual_cost)}. "
                                f"Using ${fmt(annual_cost * 2)} double-counts the carbon price impact.")
                       }
    q = (f"{_assume(['Industry: ' + carbon_intensity])}\n"
         f"A company's annual carbon emissions are {annual_emissions_mtc:,.0f} tons CO2. "
         f"A government announces a carbon price of ${carbon_price}/ton, phased in over 5 years.\n\n"
         "Q: (a) Compute the annual compliance cost. "
         "(b) Name three climate transition-risk channels affecting the company. "
         "(c) Estimate the NPV of the perpetual compliance cost (assume 3–5 % discount).")
    return {"meta": {"topic": "Climate Risk", "subtopic": "Carbon Pricing", "difficulty": "v2_FRM2_Medium",
                     "question_type": "Constructed Response",
                     "pitfalls": ["carbon price application", "transition risk channels", "stranded assets"]},
            "question": q,
            "answer": f"Annual cost: ${fmt(annual_cost)}; Transition channels: cost, demand, valuation",
            "distractors": [],
            "reasoning_trace": tr, "flawed": flaw,
            "params": {"carbon": carbon_price, "emissions": annual_emissions_mtc}
            }


m_v2_cr_counterparty = wrap_mcq(v2_cr_counterparty)
m_v2_cr_securitization = wrap_mcq(v2_cr_securitization)
m_v2_liq_nsfr = wrap_mcq(v2_liq_nsfr)
m_v2_ci_climate = wrap_vignette(v2_ci_climate)

TEMPLATES = {
    "credit_el": credit_el, "credit_var_vasicek": credit_var_vasicek,
    "op_risk_basel": op_risk_basel, "liq_lcad": liq_lcad, "liq_ws_avg": liq_ws_avg,
    "credit_var": credit_var,
    "m_credit_var": m_credit_var,

    # -- NEW FRM_Part_2 v2 stems (6) --
    "v2_cr_counterparty": v2_cr_counterparty,
    "v2_cr_securitization": v2_cr_securitization,
    "v2_op_loss_dist": v2_op_loss_dist,
    "v2_op_model_risk": v2_op_model_risk,
    "v2_liq_nsfr": v2_liq_nsfr,
    "v2_ci_climate": v2_ci_climate,

    # -- MCQ / Vignette wrappers --
    "m_v2_cr_counterparty": m_v2_cr_counterparty,
    "m_v2_cr_securitization": m_v2_cr_securitization,
    "m_v2_liq_nsfr": m_v2_liq_nsfr,
    "m_v2_ci_climate": m_v2_ci_climate,
}
