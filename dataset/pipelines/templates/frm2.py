"""
FRM Part 2 templates (credit, operational, liquidity, Basel).
"""
import math
from pipelines.core import fmt, pct
from pipelines.templates.wrappers import wrap_mcq

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
    tr = (_assume([f"EL = PD × LGD × EAD"]) +
          f"Step 1. PD×LGD = {pct(pd,1)}×{pct(lgd,1)} = {pct(pd*lgd,1)}.\n"
          f"Step 2. EL = {pct(pd*lgd,1)}×{fmt(ead)} = {fmt(el)}.\n"
          f"Step 3. Trap: using EAD without LGD ({fmt(pd*ead)}) overstates expected loss.")
    flaw = {"answer": f"{fmt(pd*ead)}", "pitfall": "omitting LGD",
            "reasoning_trace": (_assume([f"EL = PD × EAD"]) +
            f"Step 1. Multiplying PD {pct(pd,1)} by EAD {fmt(ead)} = {fmt(pd*ead)} skips the loss-given-"
            f"default {pct(lgd,1)}; EL = PD×LGD×EAD = {pct(pd,1)}×{pct(lgd,1)}×{fmt(ead)} = {fmt(el)}.")}
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
    tr = (_assume([f"Vasicek conditional PD = N[(N⁻¹(PD) − ρ·Z)/√(1−ρ²)]"]) +
          f"Step 1. N⁻¹(PD) ≈ {th:.3f} (from {pct(pd,1)}).\n"
          f"Step 2. Systemic factor Z = −2 (tail stress): numerator {th:.3f} − {rho:.2f}×(−2) = {th + 2*rho:.3f}.\n"
          f"Step 3. Denominator √(1−ρ²) = √(1−{rho:.2f}²) = {_m.sqrt(1-rho**2):.3f}.\n"
          f"Step 4. Conditional PD = {pct(cpd)} vs unconditional {pct(pd,1)} — the systemic tail multiplies default probability.\n"
          f"Step 5. Trap: quoting unconditional PD ignores the correlation-driven systemic tail.")
    flaw = {"answer": f"{pct(pd)}", "pitfall": "unconditional vs conditional PD",
            "reasoning_trace": (_assume([f"using unconditional PD"]) +
            f"Step 1. Reporting unconditional PD {pct(pd,1)} in a −2σ systemic stress ignores asset "
            f"correlation {rho:.2f}; Vasicek conditional PD rises to {pct(cpd)}.")}
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
    tr = (_assume([f"BIA charge = α × gross income (annualized)"]) +
          f"Step 1. Charge = {alpha:.2f}×{fmt(gross_income)} = {fmt(op_charge)} per year.\n"
          f"Step 2. Trap: using total income over the 3-yr window (not the average) overstates the charge.")
    flaw = {"answer": f"{fmt(alpha*gross_income*3)}", "pitfall": "average vs total income",
            "reasoning_trace": (_assume([f"total 3-yr income"]) +
            f"Step 1. Multiplying 3-yr total income × alpha = {fmt(alpha*gross_income*3)} instead of "
            f"the AVERAGE gross income × alpha = {fmt(op_charge)}. The BIA uses average gross income over 3 years.")}
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
    tr = (_assume([f"LCR = HQLA / net cash outflows"]) +
          f"Step 1. Net outflows = {fmt(outflow)} − {fmt(inflow)} = {fmt(n_outflow)}.\n"
          f"Step 2. LCR = {fmt(hqla)}/{fmt(n_outflow)} = {lcr:.2f} ({pct(lcr)}).\n"
          f"Step 3. Trap: dividing by gross outflows (no inflow offset) understates the ratio.")
    flaw = {"answer": f"{fmt(hqla/outflow)}", "pitfall": "net vs gross outflows",
            "reasoning_trace": (_assume([f"gross outflows"]) +
            f"Step 1. LCR = HQLA/gross outflows = {fmt(hqla)}/{fmt(outflow)} = {fmt(hqla/outflow)} "
            f"instead of HQLA/net outflows = {fmt(hqla)}/{fmt(n_outflow)} = {lcr:.2f}. "
            f"Basel nets inflows against outflows in the denominator.")}
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
    tr = (_assume([f"WACF = Σ(amount×rate)/Σ amount"]) +
          f"Step 1. Total = {fmt(amt1)} + {fmt(amt2)} = {fmt(total)}.\n"
          f"Step 2. Weighted cost = ({fmt(amt1)}×{pct(rate1,1)} + {fmt(amt2)}×{pct(rate2,1)})/{fmt(total)} = {pct(avg_cost)}.\n"
          f"Step 3. Trap: unweighted average ({pct((rate1+rate2)/2)}) misstates funding cost when amounts differ.")
    flaw = {"answer": f"{fmt(amt1*rate1+amt2*rate2)}", "pitfall": "weighted vs simple average",
            "reasoning_trace": (_assume([f"simple average"]) +
            f"Step 1. Averaging rates {pct(rate1,1)} and {pct(rate2,1)} = {pct((rate1+rate2)/2)} "
            f"ignores the amounts {fmt(amt1)} vs {fmt(amt2)}; weighted cost = {pct(avg_cost)}.")}
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
    tr = (_assume([f"credit VaR ≈ V × PD × LGD × z(99%)"]) +
          f"Step 1. CVaR = {fmt(pv)} × {pct(pd)} × {lgd:.0%} × 2.326 = {fmt(var99)}.\n"
          f"Step 2. Trap: dropping LGD gives {fmt(pv*pd*2.326)}.")
    flaw = {"answer": f"{fmt(pv*pd*2.326)}", "pitfall": "LGD omitted",
            "reasoning_trace": (_assume([f"LGD omitted"]) +
            f"Step 1. CVaR = {fmt(pv)} × {pct(pd)} × 2.326 = {fmt(pv*pd*2.326)}.")}
    return {"meta": {"topic":"Credit Risk","subtopic":"Credit VaR","difficulty":"FRM2_Medium",
                     "question_type":"Calculation","pitfalls":["LGD omitted"]},
            "question":q, "answer":f"{fmt(var99)}",
            "distractors":[f"{fmt(pv*pd*2.326)}", f"{fmt(pv*pd*lgd)}", f"{fmt(pv*pd)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"pv":pv,"pd":pd,"lgd":lgd}}

m_credit_var = wrap_mcq(credit_var)

TEMPLATES = {
    "credit_el": credit_el, "credit_var_vasicek": credit_var_vasicek,
    "op_risk_basel": op_risk_basel, "liq_lcad": liq_lcad, "liq_ws_avg": liq_ws_avg,
    "credit_var": credit_var,
    "m_credit_var": m_credit_var,
}
