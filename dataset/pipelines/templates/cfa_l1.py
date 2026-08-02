"""
CFA Level I templates. Each template fn(rng, seq) -> rich dict:

  {meta:{topic,subtopic,difficulty,question_type,pitfalls},
   question, answer, distractors, reasoning_trace,
   flawed:{answer,reasoning_trace,pitfall} | None, params}

Flawed variants are numerically-grounded: they use a wrong-formula branch and
land on a concrete wrong number, internally consistent with their own arithmetic.
"""
import math
from pipelines.core import fmt, pct
from pipelines.templates.wrappers import wrap_vignette, wrap_cr, wrap_mcq

PROG = "CFA_Level_I"

def _assume(assumptions):
    return "ASSUMPTIONS: " + "; ".join(assumptions) + ".\n"

# ---------------- 1. Time Value of Money ----------------
def tvm_annuity_fv(rng, seq):
    pmt = rng.randint(100, 900) * 100
    n = rng.randint(3, 10) * 12
    r_ann = rng.choice([0.04, 0.05, 0.06, 0.07, 0.08])
    r = r_ann / 12
    fv_ord = pmt * ((1 + r) ** n - 1) / r
    fv_due = fv_ord * (1 + r)
    q = (f"A client deposits ${pmt:,} at the END of each month into an account paying "
         f"{r_ann*100:.0f}% compounded monthly. Compute the future value after {n} months.")
    tr = (_assume([f"ordinary annuity (end-of-month)", f"periodic rate {r_ann*100:.0f}%/12 = {r:.4f}",
                   f"{n} periods"]) +
          f"Step 1. Periodic rate r = {r_ann}/12 = {r:.4f}.\n"
          f"Step 2. Ordinary-annuity FV factor = ((1+r)^n - 1)/r = {((1+r)**n-1)/r:.4f}.\n"
          f"Step 3. FV = PMT × factor = {pmt:,} × {((1+r)**n-1)/r:.4f} = {fmt(fv_ord)}.\n"
          f"Step 4. Deposits are END-of-month → ordinary annuity stands. "
          f"Distractor {fmt(fv_due)} is the annuity-DUE value (deposits at start), wrong timing.")
    flaw = {"answer": f"{fmt(fv_due)}", "pitfall": "annuity due vs ordinary",
            "reasoning_trace": (_assume([f"annuity DUE (start-of-month)", f"rate {r:.4f}", f"{n} periods"]) +
            f"Step 1. r = {r_ann}/12 = {r:.4f}.\n"
            f"Step 2. Ordinary factor = {((1+r)**n-1)/r:.4f}.\n"
            f"Step 3. Treating deposits as START-of-month multiplies by (1+r): "
            f"{fmt(fv_ord)} × {1+r:.4f} = {fmt(fv_due)}. This wrongly shifts every cash flow one period earlier.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Time Value of Money",
                     "difficulty":"L1_Easy","question_type":"Calculation","pitfalls":["annuity due vs ordinary","compounding frequency"]},
            "question":q, "answer":f"{fmt(fv_ord)}", "distractors":[f"{fmt(fv_due)}", f"{fmt(pmt*n)}", f"{fmt(fv_ord*0.9)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"pmt":pmt,"n":n,"r_ann":r_ann,"kind":"ordinary"}}

def tvm_pv_lump(rng, seq):
    fv = rng.randint(5, 40) * 100000
    n = rng.randint(3, 10)
    r = rng.choice([0.06, 0.08, 0.10, 0.12])
    pv = fv / (1 + r) ** n
    pv_simple = fv / (1 + r * n)
    q = (f"An amount of ${fv:,} is received in {n} years. The discount rate is {r*100:.0f}% "
         f"per year (annual compounding). Compute its present value.")
    tr = (_assume([f"annual compounding", f"rate {r*100:.0f}%", f"{n}-year horizon"]) +
          f"Step 1. PV = FV / (1+r)^n = {fv:,} / (1+{r})^{n}.\n"
          f"Step 2. (1+{r})^{n} = {(1+r)**n:.4f}; PV = {fv:,} / {(1+r)**n:.4f} = {fmt(pv)}.\n"
          f"Step 3. Trap: simple-interest divisor (1+r·n) would give {fmt(pv_simple)}; discounting must compound.")
    flaw = {"answer": f"{fmt(pv_simple)}", "pitfall": "simple vs compound discounting",
            "reasoning_trace": (_assume([f"simple-interest discounting"]) +
            f"Step 1. Using divisor (1 + r·n) = 1 + {r:.2f}×{n} = {1+r*n:.2f}.\n"
            f"Step 2. PV = {fv:,} / {1+r*n:.2f} = {fmt(pv_simple)}. This understates true PV because "
            f"interest must be compounded, not applied linearly.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Time Value of Money",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["simple vs compound discounting"]},
            "question":q, "answer":f"{fmt(pv)}", "distractors":[f"{fmt(pv_simple)}", f"{fmt(fv)}", f"{fmt(fv/(1+r)**(n+1))}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"fv":fv,"n":n,"r":r}}

def tvm_eay(rng, seq):
    nominal = rng.choice([0.05, 0.06, 0.08, 0.10, 0.12])
    m = rng.choice([2, 4, 12, 52])
    eay = (1 + nominal / m) ** m - 1
    q = (f"A nominal annual rate of {nominal*100:.0f}% is compounded {m} times per year. "
         f"Compute the effective annual rate (EAR).")
    tr = (_assume([f"nominal {nominal*100:.0f}%", f"{m} compounding periods per year"]) +
          f"Step 1. EAR = (1 + nominal/m)^m − 1 = (1 + {nominal}/{m})^{m} − 1.\n"
          f"Step 2. (1 + {nominal/m:.5f})^{m} = {(1+nominal/m)**m:.6f}; EAR = {pct(eay)}.\n"
          f"Step 3. The nominal rate {pct(nominal)} is NOT the effective rate; compounding raises it.")
    flaw = {"answer": f"{pct(nominal)}", "pitfall": "nominal vs effective rate",
            "reasoning_trace": (_assume([f"treating nominal as effective"]) +
            f"Step 1. Quoting the nominal rate {pct(nominal)} directly as the effective rate ignores "
            f"{m} compounding sub-periods per year, which raise the true annual return.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Time Value of Money",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["nominal vs effective rate","compounding frequency"]},
            "question":q, "answer":f"{pct(eay)}", "distractors":[f"{pct(nominal)}", f"{pct(eay+0.01)}", f"{pct((1+nominal/m)**m)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"nominal":nominal,"m":m}}

def tvm_npv_irr(rng, seq):
    inv = rng.randint(40, 90) * 1000
    cf = rng.randint(20, 40) * 1000
    cf2 = rng.randint(30, 50) * 1000
    r = rng.choice([0.08, 0.10, 0.12, 0.15])
    npv = -inv + cf/(1+r) + cf2/(1+r)**2
    lo, hi = 0.01, 0.30
    for _ in range(40):
        mid = (lo+hi)/2
        v = -inv + cf/(1+mid) + cf2/(1+mid)**2
        if v > 0: lo = mid
        else: hi = mid
    irr = (lo+hi)/2
    q = (f"A project costs ${inv:,} and returns ${cf:,} in year 1 and ${cf2:,} in year 2. "
         f"Cost of capital is {r*100:.0f}%. Compute NPV; state whether IRR > cost of capital.")
    tr = (_assume([f"cost of capital {r*100:.0f}%", f"CF1 ${cf:,}", f"CF2 ${cf2:,}"]) +
          f"Step 1. NPV = −{inv:,} + {cf:,}/(1+{r}) + {cf2:,}/(1+{r})².\n"
          f"Step 2. PV1 = {fmt(cf/(1+r))}; PV2 = {fmt(cf2/(1+r)**2)}.\n"
          f"Step 3. NPV = {fmt(npv)}. " +
          (f"NPV>0 → accept; IRR ≈ {pct(irr)} > {pct(r)}, consistent." if npv>0 else
           f"NPV<0 → reject; IRR ≈ {pct(irr)} < {pct(r)}."))
    flaw = {"answer": f"{fmt(inv+cf/(1+r)+cf2/(1+r)**2)}", "pitfall": "NPV sign",
            "reasoning_trace": (_assume([f"sign error"]) +
            f"Step 1. If the initial outlay is added instead of subtracted: NPV = +{inv:,} + {fmt(cf/(1+r))} + {fmt(cf2/(1+r)**2)} "
            f"= {fmt(inv+cf/(1+r)+cf2/(1+r)**2)}. A cash outflow must be discounted and subtracted.")}
    return {"meta": {"topic":"Corporate Issuers","subtopic":"Capital Budgeting",
                     "difficulty":"L1_Hard","question_type":"Calculation","pitfalls":["NPV sign","discounting cash flows"]},
            "question":q, "answer":f"{fmt(npv)}", "distractors":[f"{fmt(npv*1.2)}", f"{fmt(npv*0.8)}", f"{fmt(cf+cf2-inv)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"inv":inv,"cf":cf,"cf2":cf2,"r":r}}

# ---------------- 2. Statistics & Probability ----------------
def stats_var_sd(rng, seq):
    vals = [rng.randint(10, 60) for _ in range(5)]
    n = len(vals)
    mean = sum(vals)/n
    var_s = sum((v-mean)**2 for v in vals)/(n-1)
    sd_s = math.sqrt(var_s)
    var_p = sum((v-mean)**2 for v in vals)/n
    q = f"Returns (%) over 5 quarters: {vals}. Compute the sample variance and sample standard deviation."
    tr = (_assume([f"sample (n−1) denominator)", f"n = {n}"]) +
          f"Step 1. Mean = ({', '.join(str(v) for v in vals)})/{n} = {mean:.2f}.\n"
          f"Step 2. Sum of squared deviations = {sum((v-mean)**2 for v in vals):.2f}.\n"
          f"Step 3. Sample variance = SS/(n−1) = {fmt(var_s)}; SD = √variance = {fmt(sd_s)}.\n"
          f"Step 4. Trap: using n (population) gives {fmt(var_p)} — wrong for a sample.")
    flaw = {"answer": f"variance {fmt(var_p)}, SD {fmt(math.sqrt(var_p))}", "pitfall": "sample vs population denominator",
            "reasoning_trace": (_assume([f"population denominator n instead of n−1"]) +
            f"Step 1. Variance = SS/n = {fmt(var_p)} (should divide by n−1={n-1}). "
            f"Using the population denominator understates the sample variance and SD.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Descriptive Statistics",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["sample vs population denominator"]},
            "question":q, "answer":f"variance {fmt(var_s)}, SD {fmt(sd_s)}",
            "distractors":[f"var {fmt(var_p)}, SD {fmt(math.sqrt(var_p))}", f"var {fmt(var_s*2)}", f"var {fmt(mean)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"vals":vals}}

def stats_bayes(rng, seq):
    prev = rng.uniform(0.02, 0.10, 3)
    sens = rng.choice([0.85, 0.90, 0.95])
    spec = rng.choice([0.90, 0.95, 0.99])
    ppos = prev*sens + (1-prev)*(1-spec)
    post = prev*sens / ppos
    q = (f"Disease prevalence is {pct(prev,1)}. A test has sensitivity {sens:.2f} (P(+|D)) and "
         f"specificity {spec:.2f} (P(−|no D)). A patient tests positive. Compute P(D|+).")
    tr = (_assume([f"prevalence {pct(prev,1)}", f"sensitivity {sens:.2f}", f"specificity {spec:.2f}"]) +
          f"Step 1. P(Pos) = P(D)·sens + P(no D)·(1−spec) = {prev:.4f}×{sens:.2f} + {(1-prev):.4f}×{1-spec:.2f} = {ppos:.4f}.\n"
          f"Step 2. P(D|Pos) = P(D)·sens / P(Pos) = {prev:.4f}×{sens:.2f} / {ppos:.4f} = {pct(post)}.\n"
          f"Step 3. Trap: quoting sensitivity {pct(sens)} ignores the base rate — posterior is far lower.")
    flaw = {"answer": f"{pct(sens)}", "pitfall": "base-rate neglect",
            "reasoning_trace": (_assume([f"base-rate neglect"]) +
            f"Step 1. Quoting sensitivity P(Pos|D) = {sens:.2f} directly as P(D|Pos) ignores prevalence "
            f"{pct(prev,1)} and false positives {pct(1-spec,1)}. Bayes must condition on the test result.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Probability",
                     "difficulty":"L1_Hard","question_type":"Calculation","pitfalls":["base-rate neglect"]},
            "question":q, "answer":f"{pct(post)}", "distractors":[f"{pct(sens)}", f"{pct(prev,1)}", f"{pct(1-ppos)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"prev":prev,"sens":sens,"spec":spec}}

def stats_ci(rng, seq):
    mu = rng.randint(50, 80)
    sd = rng.randint(5, 15)
    n = rng.choice([25, 36, 49, 100])
    z = 1.96
    se = sd/math.sqrt(n)
    lo, hi = mu - z*se, mu + z*se
    q = (f"A sample of {n} has mean {mu} and sample SD {sd}. Build a 95% confidence interval for the "
         f"population mean (use z = 1.96).")
    tr = (_assume([f"CLT applies (n={n})", f"z* = 1.96"]) +
          f"Step 1. SE = s/√n = {sd}/√{n} = {se:.3f}.\n"
          f"Step 2. Margin = 1.96×SE = {1.96*se:.3f}.\n"
          f"Step 3. CI = {mu} ± {1.96*se:.2f} = [{lo:.2f}, {hi:.2f}].\n"
          f"Step 4. Trap: using t with df=n−1 changes the margin; z=1.96 is valid for large n.")
    flaw = {"answer": f"[{mu-sd:.2f}, {mu+sd:.2f}]", "pitfall": "SE vs SD",
            "reasoning_trace": (_assume([f"using SD instead of SE"]) +
            f"Step 1. Wrongly using the sample SD {sd} directly as the dispersion of the mean "
            f"(not SD/√n) gives [{mu-sd:.2f}, {mu+sd:.2f}]. The standard error must shrink by √{n}.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Sampling & Confidence Intervals",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["SE vs SD","z vs t"]},
            "question":q, "answer":f"[{lo:.2f}, {hi:.2f}]", "distractors":[f"[{mu-sd:.2f}, {mu+sd:.2f}]", f"[{mu-2*sd:.2f}, {mu+2*sd:.2f}]", f"[{lo-1:.2f}, {hi+1:.2f}]"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"mu":mu,"sd":sd,"n":n}}

def stats_tstat(rng, seq):
    mu0 = 50
    xbar = rng.randint(52, 58)
    sd = rng.randint(4, 8)
    n = rng.choice([25, 30, 36])
    t = (xbar - mu0)/(sd/math.sqrt(n))
    crit = 1.699
    q = (f"Test H0: μ = {mu0} vs Ha: μ > {mu0} at α = 0.05, one-tailed, df = {n-1}. "
         f"Sample mean {xbar}, SD {sd}, n = {n}. Compute the t-statistic; critical t(0.05, {n-1}) ≈ {crit}.")
    tr = (_assume([f"H0: μ={mu0}", f"one-tailed α=0.05", f"df={n-1}"]) +
          f"Step 1. SE = s/√n = {sd}/√{n} = {sd/math.sqrt(n):.3f}.\n"
          f"Step 2. t = (x̄−μ0)/SE = ({xbar}-{mu0})/{sd/math.sqrt(n):.3f} = {t:.3f}.\n"
          f"Step 3. t = {t:.3f} {'>' if t>crit else '<'} {crit} → {'reject H0' if t>crit else 'cannot reject H0'}.\n"
          f"Step 4. Trap: using z (SD, not SE) would give {fmt((xbar-mu0)/sd)} and a wrong conclusion.")
    flaw = {"answer": f"t = {fmt((xbar-mu0)/sd)}", "pitfall": "SE vs SD in test statistic",
            "reasoning_trace": (_assume([f"using SD instead of SE"]) +
            f"Step 1. Computing t = (x̄−μ0)/s = ({xbar}-{mu0})/{sd} = {fmt((xbar-mu0)/sd)} "
            f"instead of dividing by s/√n. The test statistic must standardize by the standard error.")}
    return {"meta": {"topic":"Quantitative Methods","subtopic":"Hypothesis Testing",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["SE vs SD","one vs two-tailed"]},
            "question":q, "answer":f"t = {t:.3f}; reject H0" if t>crit else f"t = {t:.3f}; cannot reject H0",
            "distractors":[f"t = {fmt((xbar-mu0)/sd)}", f"t = {fmt((xbar-mu0)/sd*2)}", f"t = {fmt(-t)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"mu0":mu0,"xbar":xbar,"sd":sd,"n":n}}

# ---------------- 3. Economics ----------------
def econ_elasticity(rng, seq):
    q0 = rng.randint(80, 150)
    q1 = rng.randint(50, 100)
    p0 = rng.randint(20, 40)
    p1 = rng.randint(30, 50)
    if q1 >= q0 or p1 <= p0:
        q1, p1 = q0 - rng.randint(20, 60), p0 + rng.randint(10, 20)
    midq, midp = (q0+q1)/2, (p0+p1)/2
    e = ((q1-q0)/midq) / ((p1-p0)/midp)
    q = (f"Price rises from ${p0} to ${p1}; quantity demanded falls from {q0} to {q1}. "
         f"Compute price elasticity of demand (midpoint method) and classify demand.")
    tr = (_assume([f"midpoint elasticity"]) +
          f"Step 1. ΔQ/Q_mid = ({q1}-{q0})/{midq:.0f} = {(q1-q0)/midq:.3f}.\n"
          f"Step 2. ΔP/P_mid = ({p1}-{p0})/{midp:.0f} = {(p1-p0)/midp:.3f}.\n"
          f"Step 3. E = {e:.2f}. {'Elastic (|E|>1): revenue falls as price rises.' if abs(e)>1 else 'Inelastic (|E|<1): revenue rises with price.'}")
    flaw = {"answer": f"{abs(e):.2f}", "pitfall": "sign convention",
            "reasoning_trace": (_assume([f"dropping the negative sign"]) +
            f"Step 1. The raw ratio is {(q1-q0)/midq:.3f}/{(p1-p0)/midp:.3f} = {e:.2f} but demand slopes "
            f"down, so elasticity is reported as −{e:.2f}; the sign encodes the law of demand.")}
    return {"meta": {"topic":"Economics","subtopic":"Elasticity",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["midpoint method","sign"]},
            "question":q, "answer":f"{e:.2f}", "distractors":[f"{fmt(e*2)}", f"{fmt((q1-q0)/(p1-p0))}", f"{fmt(e+1)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"q0":q0,"q1":q1,"p0":p0,"p1":p1}}

def econ_fisher(rng, seq):
    rnom = rng.choice([0.06, 0.08, 0.10, 0.12])
    rinf = rng.choice([0.02, 0.03, 0.04])
    rreal = (1 + rnom)/(1 + rinf) - 1
    q = (f"Nominal rate {pct(rnom,1)} and inflation {pct(rinf,1)}. Compute the exact real rate "
         f"of return (Fisher exact).")
    tr = (_assume([f"Fisher exact"]) +
          f"Step 1. (1+r_real) = (1+{rnom:.2f})/(1+{rinf:.2f}).\n"
          f"Step 2. r_real = {pct(rreal)}.\n"
          f"Step 3. Trap: the approximation r_real ≈ r_nom − inflation = {pct(rnom-rinf,1)} differs slightly; "
          f"exact divides, not subtracts.")
    flaw = {"answer": f"{pct(rnom-rinf,1)}", "pitfall": "Fisher approximation vs exact",
            "reasoning_trace": (_assume([f"additive approximation"]) +
            f"Step 1. Using r_real ≈ {pct(rnom,1)} − {pct(rinf,1)} = {pct(rnom-rinf,1)} "
            f"ignores the cross term; exact requires (1+r_nom)/(1+π)−1 = {pct(rreal)}.")}
    return {"meta": {"topic":"Economics","subtopic":"Interest Rates",
                     "difficulty":"L1_Easy","question_type":"Calculation","pitfalls":["Fisher exact vs approximation"]},
            "question":q, "answer":f"{pct(rreal)}", "distractors":[f"{pct(rnom-rinf,1)}", f"{pct(rnom+rinf)}", f"{pct(rinf-rnom)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"rnom":rnom,"rinf":rinf}}

# ---------------- 4. FSA ----------------
def fsa_dupont(rng, seq):
    ni = rng.randint(80, 150) * 100000
    sales = rng.randint(800, 1200) * 100000
    assets = rng.randint(500, 900) * 100000
    equity = rng.randint(300, 500) * 100000
    npm = ni/sales; at = sales/assets; em = assets/equity
    roe = npm * at * em
    roe_direct = ni/equity
    q = (f"NI ${ni:,}, Sales ${sales:,}, Assets ${assets:,}, Equity ${equity:,}. "
         f"Decompose ROE via DuPont (NPM × AT × EM).")
    tr = (_assume([f"DuPont: ROE = NPM × Asset Turnover × Equity Multiplier"]) +
          f"Step 1. NPM = NI/Sales = {fmt(ni)}/{fmt(sales)} = {pct(npm)}.\n"
          f"Step 2. Asset Turnover = Sales/Assets = {fmt(sales)}/{fmt(assets)} = {at:.2f}.\n"
          f"Step 3. Equity Multiplier = Assets/Equity = {fmt(assets)}/{fmt(equity)} = {em:.2f}.\n"
          f"Step 4. ROE = {pct(npm)} × {at:.2f} × {em:.2f} = {pct(roe)}. "
          f"Check: NI/Equity = {pct(roe_direct)} ✓")
    flaw = {"answer": f"{pct(npm*at)}", "pitfall": "omitting leverage",
            "reasoning_trace": (_assume([f"ROE without leverage"]) +
            f"Step 1. Computing ROE = NPM × Asset Turnover = {pct(npm)} × {at:.2f} = {pct(npm*at)} "
            f"omits the equity multiplier {em:.2f}; ROE must reflect financing leverage: ROE = NPM×AT×EM = {pct(roe)}.")}
    return {"meta": {"topic":"Financial Statement Analysis","subtopic":"Ratio Analysis",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["DuPont decomposition","leverage"]},
            "question":q, "answer":f"{pct(roe)}", "distractors":[f"{pct(npm*at)}", f"{pct(npm)}", f"{pct(at*em)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"ni":ni,"sales":sales,"assets":assets,"equity":equity}}

def fsa_inventory_turnover(rng, seq):
    cogs = rng.randint(400, 700) * 1000
    inv_beg = rng.randint(60, 100) * 1000
    inv_end = rng.randint(50, 90) * 1000
    avg = (inv_beg + inv_end)/2
    ito = cogs / avg
    days = 365 / ito
    q = (f"COGS ${cogs:,}, inventory beginning ${inv_beg:,}, ending ${inv_end:,}. "
         f"Compute inventory turnover and days of inventory on hand (365-day year).")
    tr = (_assume([f"avg inventory = (begin+end)/2"]) +
          f"Step 1. Avg inventory = ({fmt(inv_beg)} + {fmt(inv_end)})/2 = {fmt(avg)}.\n"
          f"Step 2. Turnover = COGS/avg = {fmt(cogs)}/{fmt(avg)} = {ito:.2f}.\n"
          f"Step 3. Days = 365/turnover = 365/{ito:.2f} = {days:.1f}.\n"
          f"Step 4. Trap: using ending inventory only inflates turnover to {fmt(cogs/inv_end)}.")
    flaw = {"answer": f"turnover {fmt(cogs/inv_end)}", "pitfall": "avg vs ending inventory",
            "reasoning_trace": (_assume([f"ending inventory only"]) +
            f"Step 1. Using ending inventory {fmt(inv_end)} instead of average gives turnover "
            f"{fmt(cogs/inv_end)}, overstating turnover when inventory is not stable; average is required.")}
    return {"meta": {"topic":"Financial Statement Analysis","subtopic":"Inventory & Activity Ratios",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["average inventory"]},
            "question":q, "answer":f"turnover {ito:.2f}; days {days:.1f}", "distractors":[f"turnover {fmt(cogs/inv_end)}", f"turnover {fmt(cogs/(inv_beg+inv_end))}", f"turnover {fmt(ito*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"cogs":cogs,"inv_beg":inv_beg,"inv_end":inv_end}}

# ---------------- 5. Equity ----------------
def eq_gordon(rng, seq):
    d0 = rng.randint(1, 5)
    g = rng.uniform(0.02, 0.07, 3)
    r = rng.choice([0.09, 0.10, 0.11, 0.12])
    d1 = d0*(1+g)
    v = d1/(r-g)
    q = (f"D0 = ${d0:.2f}, growth g = {pct(g,1)}, required return r = {pct(r,1)} (g < r). "
         f"Compute intrinsic value via the Gordon growth model (D1/(r−g)).")
    tr = (_assume([f"GGM: V = D1/(r−g)", f"g < r required"]) +
          f"Step 1. D1 = D0(1+g) = {d0:.2f}×{1+g:.3f} = {d1:.3f}.\n"
          f"Step 2. r−g = {r:.2f} − {g:.3f} = {r-g:.4f}.\n"
          f"Step 3. V = D1/(r−g) = {d1:.3f}/{r-g:.4f} = {fmt(v)}.\n"
          f"Step 4. Trap: using D0 (no growth) gives {fmt(d0/(r-g))}; use next dividend D1.")
    flaw = {"answer": f"{fmt(d0/(r-g))}", "pitfall": "D0 vs D1",
            "reasoning_trace": (_assume([f"using D0 instead of D1"]) +
            f"Step 1. Plugging D0 = {d0:.2f} directly (ignoring the first growth step) gives {d0:.2f}/{r-g:.4f} = {fmt(d0/(r-g))}. "
            f"GGM requires the next dividend D1 = D0(1+g) = {d1:.3f}.")}
    return {"meta": {"topic":"Equity Investments","subtopic":"Equity Valuation",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["D0 vs D1","g < r condition"]},
            "question":q, "answer":f"{fmt(v)}", "distractors":[f"{fmt(d0/(r-g))}", f"{fmt(v*1.5)}", f"{fmt(d1/g)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"d0":d0,"g":g,"r":r}}

def eq_pe_earnings(rng, seq):
    price = rng.randint(20, 60)
    eps = rng.randint(2, 6)
    pe = price/eps
    ev = rng.randint(8, 15)
    q = (f"Price ${price:.0f}, EPS ${eps:.0f}. Compute P/E ratio; is the stock cheaper or "
         f"dearer than a comparable with P/E {ev}?")
    tr = (_assume([f"P/E = Price/EPS"]) +
          f"Step 1. P/E = {price:.0f}/{eps:.0f} = {pe:.2f}.\n"
          f"Step 2. {'Cheaper than the comparable P/E '+str(ev)+' (lower P/E).' if pe<ev else 'Dearer than the comparable P/E '+str(ev)+' (higher P/E).'}\n"
          f"Step 3. Trap: earnings yield = EPS/Price = {pct(eps/price)}; don't invert the ratio.")
    flaw = {"answer": f"P/E = {fmt(eps/price)}", "pitfall": "inverted ratio",
            "reasoning_trace": (_assume([f"inverting P/E"]) +
            f"Step 1. Computing Price/EPS inverted as EPS/Price = {eps:.0f}/{price:.0f} = {fmt(eps/price)}. "
            f"This is the earnings yield, not the P/E multiple.")}
    return {"meta": {"topic":"Equity Investments","subtopic":"Equity Multiples",
                     "difficulty":"L1_Easy","question_type":"Calculation","pitfalls":["P/E inversion","earnings yield"]},
            "question":q, "answer":f"P/E {pe:.2f}", "distractors":[f"P/E {fmt(eps/price)}", f"P/E {fmt(pe*10)}", f"P/E {fmt(pe+1)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"price":price,"eps":eps,"ev":ev}}

# ---------------- 6. Fixed Income ----------------
def fi_current_yield(rng, seq):
    coupon = rng.randint(4, 8) * 10
    price = rng.randint(85, 115)
    cy = (coupon)/price*100
    ytm = rng.uniform(0.05, 0.09, 3)
    q = (f"A bond pays an annual coupon of ${coupon:.0f} and trades at ${price:.0f} (par 100). "
         f"Compute its current yield (%).")
    tr = (_assume([f"current yield = annual coupon / price"]) +
          f"Step 1. Current yield = ${coupon:.0f}/${price:.0f} = {cy:.2f}%.\n"
          f"Step 2. Note current yield ≠ YTM; if price ≠ par, YTM differs ({fmt(ytm*100)}% in this setup).\n"
          f"Step 3. Trap: for a premium bond, YTM < current yield < coupon rate.")
    flaw = {"answer": f"current yield = {fmt(ytm*100)}%", "pitfall": "CY vs YTM",
            "reasoning_trace": (_assume([f"using YTM instead of CY"]) +
            f"Step 1. Reporting the yield-to-maturity {fmt(ytm*100)}% as the current yield ignores the "
            f"cash-flow coupon ${coupon:.0f} on price ${price:.0f}. CY = coupon/price = {cy:.2f}%.")}
    return {"meta": {"topic":"Fixed Income","subtopic":"Yield Measures",
                     "difficulty":"L1_Easy","question_type":"Calculation","pitfalls":["current yield vs YTM"]},
            "question":q, "answer":f"{cy:.2f}%", "distractors":[f"{fmt(ytm*100)}%", f"{fmt(coupon)}%", f"{fmt(cy*2)}%"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"coupon":coupon,"price":price,"ytm":ytm}}

def fi_modified_duration(rng, seq):
    mac = rng.uniform(4, 9, 3)
    y = rng.choice([0.05, 0.06, 0.08])
    mod = mac/(1+y)
    price = rng.randint(85, 115)
    dy = rng.choice([0.01, 0.02])
    pchange = -mod*dy*100
    q = (f"A bond has Macaulay duration {mac:.2f} yrs, yield {pct(y,1)}. Compute modified duration "
         f"and the % price change for a {pct(dy,1)} yield increase.")
    tr = (_assume([f"ModDur = MacDur/(1+y)", f"ΔP/P ≈ −ModDur·Δy"]) +
          f"Step 1. ModDur = {mac:.2f}/(1+{y:.2f}) = {mod:.3f} yrs.\n"
          f"Step 2. ΔP/P ≈ −{mod:.3f} × {dy:.2f} = {pchange:.2f}%.\n"
          f"Step 3. Trap: using Macaulay duration directly (no yield adjustment) overstates the price move.")
    flaw = {"answer": f"ΔP/P ≈ {fmt(-mac*dy*100)}%", "pitfall": "Macaulay vs modified duration",
            "reasoning_trace": (_assume([f"using Macaulay duration"]) +
            f"Step 1. Applying Macaulay duration {mac:.2f} directly to ΔP/P = −{mac:.2f}×{dy:.2f} = {fmt(-mac*dy*100)}% "
            f"ignores the yield scaling 1/(1+y); modified duration is {mod:.3f}.")}
    return {"meta": {"topic":"Fixed Income","subtopic":"Duration & Convexity",
                     "difficulty":"L1_Hard","question_type":"Calculation","pitfalls":["Macaulay vs modified","duration sign"]},
            "question":q, "answer":f"ModDur {mod:.3f} yrs; ΔP/P {pchange:.2f}%",
            "distractors":[f"ΔP/P {fmt(-mac*dy*100)}%", f"ΔP/P {fmt(pchange*2)}%", f"ΔP/P {fmt(-pchange)}%"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"mac":mac,"y":y,"dy":dy}}

def fi_zero_duration(rng, seq):
    y = rng.choice([0.06, 0.08, 0.10])
    m = rng.randint(5, 20)
    mod = m/(1+y)
    q = (f"A zero-coupon bond matures in {m} years, yield {pct(y,1)}. Compute its modified duration.")
    tr = (_assume([f"zero-coupon: Macaulay duration = maturity"]) +
          f"Step 1. Macaulay duration = maturity = {m} yrs (no coupons).\n"
          f"Step 2. ModDur = {m}/(1+{y:.2f}) = {mod:.2f} yrs.\n"
          f"Step 3. Trap: for a coupon bond, Macaulay duration < maturity; a zero's duration equals its maturity.")
    flaw = {"answer": f"{fmt(m)} yrs", "pitfall": "modified vs Macaulay",
            "reasoning_trace": (_assume([f"using Macaulay as modified"]) +
            f"Step 1. Reporting Macaulay duration = {m} as modified duration omits the (1+y) "
            f"scaling; modified duration = {m}/(1+{y:.2f}) = {mod:.2f}.")}
    return {"meta": {"topic":"Fixed Income","subtopic":"Duration",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["zero-coupon duration","modified scaling"]},
            "question":q, "answer":f"{mod:.2f} yrs", "distractors":[f"{fmt(m)} yrs", f"{fmt(mod*2)} yrs", f"{fmt(m*(1+y))} yrs"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"y":y,"m":m}}

# ---------------- 7. Derivatives ----------------
def deriv_call_payoff(rng, seq):
    s = rng.randint(45, 60)
    k = rng.randint(40, 55)
    prem = rng.randint(2, 6)
    payoff = max(s-k, 0)
    profit = payoff - prem
    q = (f"Call option: spot ${s}, strike ${k}, premium ${prem}. At expiry spot is ${s}. "
         f"Compute intrinsic payoff and net profit.")
    tr = (_assume([f"call payoff = max(S−K, 0)", f"profit = payoff − premium"]) +
          f"Step 1. Intrinsic value = max({s}−{k}, 0) = {payoff} (ITM).\n"
          f"Step 2. Net profit = {payoff} − {prem} = {profit}.\n"
          f"Step 3. Trap: ignoring the premium paid would overstate profit; premium is a sunk cost.")
    flaw = {"answer": f"profit = {fmt(payoff)}", "pitfall": "ignoring premium",
            "reasoning_trace": (_assume([f"ignoring premium cost"]) +
            f"Step 1. Reporting intrinsic payoff {payoff} as net profit forgets the {prem} premium "
            f"paid to buy the option; net profit = {payoff} − {prem} = {profit}.")}
    return {"meta": {"topic":"Derivatives","subtopic":"Option Payoffs",
                     "difficulty":"L1_Easy","question_type":"Calculation","pitfalls":["premium as sunk cost","ITM/OTM"]},
            "question":q, "answer":f"payoff {payoff}; profit {profit}", "distractors":[f"profit {fmt(payoff)}", f"profit {fmt(payoff+prem)}", f"profit {fmt(-prem)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"s":s,"k":k,"prem":prem}}

def deriv_binomial_call(rng, seq):
    s0 = rng.randint(45, 60)
    u = rng.choice([1.15, 1.20, 1.25])
    d = rng.choice([0.85, 0.80, 0.75])
    k = rng.randint(45, 55)
    rf = rng.choice([0.04, 0.05, 0.06])
    su = s0*u; sd = s0*d
    cu = max(su-k, 0); cd = max(sd-k, 0)
    # hedge ratio
    h = (cu-cd)/(su-sd)
    c = (h*su - cu)/(1+rf)
    q = (f"One-period binomial: S0={s0}, u={u:.2f}, d={d:.2f}, K={k}, rf={pct(rf,1)}. "
         f"Compute the call price (no-arbitrage).")
    tr = (_assume([f"u = {u:.2f}, d = {d:.2f}", f"rf = {pct(rf,1)}"]) +
          f"Step 1. Up state: S_u = {s0}×{u:.2f} = {fmt(su)}, call = max({fmt(su)}−{k},0) = {fmt(cu)}.\n"
          f"Step 2. Down state: S_d = {s0}×{d:.2f} = {fmt(sd)}, call = max({fmt(sd)}−{k},0) = {fmt(cd)}.\n"
          f"Step 3. Hedge ratio h = (C_u−C_d)/(S_u−S_d) = {h:.3f}.\n"
          f"Step 4. Call = (h·S_u − C_u)/(1+rf) = ({h:.3f}×{fmt(su)} − {fmt(cu)})/{1+rf:.3f} = {fmt(c)}.\n"
          f"Step 5. Trap: averaging the payoffs without hedging (no risk-neutral discount) gives {fmt((cu+cd)/2/(1+rf))} — wrong.")
    flaw = {"answer": f"{fmt(k)}", "pitfall": "confusing strike with call price",
            "reasoning_trace": (_assume([f"simple payoff average"]) +
            f"Step 1. Averaging payoffs {fmt(cu)} and {fmt(cd)} = {fmt((cu+cd)/2)} without the "
            f"hedge ratio {h:.3f} and risk-neutral weighting breaks replication; true price {fmt(c)}."
            f"Without the hedge ratio {h:.3f}, the no-arbitrage replication is broken; true price {fmt(c)}.")}
    return {"meta": {"topic":"Derivatives","subtopic":"Option Pricing",
                     "difficulty":"L1_Hard","question_type":"Calculation","pitfalls":["no-arbitrage replication","risk-neutral"]},
            "question":q, "answer":f"{fmt(c)}", "distractors":[f"{fmt((cu+cd)/2/(1+rf))}", f"{fmt(cu)}", f"{fmt(cd)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"s0":s0,"u":u,"d":d,"k":k,"rf":rf}}

# ---------------- 8. Alternatives ----------------
def alt_hedge_fee(rng, seq):
    aum = rng.randint(100, 300) * 1000000
    mgmt = rng.choice([0.01, 0.02])
    perf = rng.choice([0.20, 0.25])
    hwm = rng.uniform(0.02, 0.10, 3)
    ret = rng.uniform(0.10, 0.25, 3)
    mgmt_fee = aum*mgmt
    perf_fee = aum*max(ret - hwm, 0)*perf
    total = mgmt_fee + perf_fee
    q = (f"Hedge fund: AUM ${fmt(aum)}, management fee {pct(mgmt,1)}, performance fee {pct(perf,1)} "
         f"with hurdle {pct(hwm,1)} (hurdle = high-water mark). Gross return {pct(ret,1)}. "
         f"Compute total fees (in $).")
    tr = (_assume([f"mgmt = AUM×{pct(mgmt,1)}", f"perf = AUM×max(ret−hwm,0)×{pct(perf,1)}"]) +
          f"Step 1. Mgmt fee = {fmt(aum)}×{mgmt:.2f} = {fmt(mgmt_fee)}.\n"
          f"Step 2. Performance = {pct(ret,1)} − hurdle {pct(hwm,1)} = {pct(max(ret-hwm,0),1)}; "
          f"perf fee = {fmt(aum)}×{pct(max(ret-hwm,0),1)}×{perf:.2f} = {fmt(perf_fee)}.\n"
          f"Step 3. Total = {fmt(mgmt_fee)} + {fmt(perf_fee)} = {fmt(total)}.\n"
          f"Step 4. Trap: skipping the hurdle when ret < hwm would charge a fee on no performance.")
    flaw = {"answer": f"{fmt(aum*mgmt + aum*ret*perf)}", "pitfall": "fee-on-return without hurdle",
            "reasoning_trace": (_assume([f"charging perf fee on full return"]) +
            f"Step 1. If perf fee applied to the full return {pct(ret,1)} instead of the excess over "
            f"hurdle {pct(hwm,1)}: {fmt(aum)}×{pct(ret,1)}×{perf:.2f} = {fmt(aum*ret*perf)}. "
            f"Fees must apply only to return above the high-water mark/hurdle.")}
    return {"meta": {"topic":"Alternative Investments","subtopic":"Hedge Funds",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["high-water mark","hurdle rate"]},
            "question":q, "answer":f"{fmt(total)}", "distractors":[f"{fmt(aum*ret*perf)}", f"{fmt(mgmt_fee)}", f"{fmt(total*0.5)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"aum":aum,"mgmt":mgmt,"perf":perf,"hwm":hwm,"ret":ret}}

def alt_cap_rate(rng, seq):
    noi = rng.randint(200, 500) * 1000
    val = rng.randint(2, 5) * 1000000
    cap = noi/val
    q = (f"Real estate: NOI ${fmt(noi)}, property value ${fmt(val)}. Compute the cap rate (%).")
    tr = (_assume([f"cap rate = NOI / property value"]) +
          f"Step 1. Cap rate = {fmt(noi)}/{fmt(val)} = {pct(cap)}.\n"
          f"Step 2. Trap: using gross income (not NOI) would overstate the cap rate; NOI excludes "
          f"non-operating items.")
    flaw = {"answer": f"{pct(cap*2)}", "pitfall": "NOI vs gross income",
            "reasoning_trace": (_assume([f"using gross income"]) +
            f"Step 1. If gross income (with vacancies/operating costs ignored) were {fmt(noi*2)} instead "
            f"of NOI {fmt(noi)}, the cap rate doubles to {pct(cap*2)}. Cap rate requires net operating income.")}
    return {"meta": {"topic":"Alternative Investments","subtopic":"Real Estate",
                     "difficulty":"L1_Easy","question_type":"Calculation","pitfalls":["NOI definition"]},
            "question":q, "answer":f"{pct(cap)}", "distractors":[f"{pct(cap*2)}", f"{pct(cap*0.5)}", f"{pct(cap+0.05)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"noi":noi,"val":val}}

# ---------------- 9. Portfolio ----------------
def port_capm(rng, seq):
    rf = rng.uniform(0.02, 0.04, 3)
    rm = rng.uniform(0.08, 0.12, 3)
    beta = rng.uniform(0.7, 1.6, 3)
    er = rf + beta*(rm - rf)
    q = (f"CAPM: rf = {pct(rf,1)}, E(Rm) = {pct(rm,1)}, β = {beta:.2f}. "
         f"Compute the required return.")
    tr = (_assume([f"CAPM: E(R) = rf + β(E(Rm)−rf)"]) +
          f"Step 1. Market risk premium = {pct(rm,1)} − {pct(rf,1)} = {pct(rm-rf,1)}.\n"
          f"Step 2. E(R) = {pct(rf,1)} + {beta:.2f}×{pct(rm-rf,1)} = {pct(er)}.\n"
          f"Step 3. Trap: using β as a discount on total return (no rf) understates required return.")
    flaw = {"answer": f"{pct(beta*(rm-rf))}", "pitfall": "omitting risk-free rate",
            "reasoning_trace": (_assume([f"omitting rf"]) +
            f"Step 1. Computing E(R) = β×market premium = {beta:.2f}×{pct(rm-rf,1)} = {pct(beta*(rm-rf))} "
            f"drops the risk-free rate {pct(rf,1)}; CAPM requires E(R) = rf + β(E(Rm)−rf) = {pct(er)}.")}
    return {"meta": {"topic":"Portfolio Management","subtopic":"CAPM & Risk",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["CAPM formula","market risk premium"]},
            "question":q, "answer":f"{pct(er)}", "distractors":[f"{pct(beta*(rm-rf))}", f"{pct(rm)}", f"{pct(er+0.02)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"rf":rf,"rm":rm,"beta":beta}}

def port_sharpe_treynor(rng, seq):
    rp = rng.uniform(0.08, 0.15, 3)
    rf = rng.uniform(0.02, 0.04, 3)
    sig = rng.uniform(0.10, 0.25, 3)
    beta = rng.uniform(0.8, 1.5, 3)
    sharpe = (rp - rf)/sig
    treynor = (rp - rf)/beta
    q = (f"Portfolio return {pct(rp,1)}, rf {pct(rf,1)}, σ {pct(sig,1)}, β {beta:.2f}. "
         f"Compute Sharpe and Treynor ratios.")
    tr = (_assume([f"Sharpe = (R−rf)/σ", f"Treynor = (R−rf)/β"]) +
          f"Step 1. Excess return = {pct(rp,1)} − {pct(rf,1)} = {pct(rp-rf,1)}.\n"
          f"Step 2. Sharpe = {pct(rp-rf,1)}/{pct(sig,1)} = {sharpe:.3f}.\n"
          f"Step 3. Treynor = {pct(rp-rf,1)}/{beta:.2f} = {treynor:.3f}.\n"
          f"Step 4. Trap: dividing by β instead of σ (or vice versa) mixes denominators — Sharpe uses total risk, Treynor uses systematic risk.")
    flaw = {"answer": f"Sharpe = {fmt((rp-rf)/beta)}", "pitfall": "denominator mixup",
            "reasoning_trace": (_assume([f"using β in Sharpe"]) +
            f"Step 1. Dividing excess return by β instead of σ: {pct(rp-rf,1)}/{beta:.2f} = {fmt((rp-rf)/beta)}. "
            f"Sharpe standardizes by total risk σ, not beta.")}
    return {"meta": {"topic":"Portfolio Management","subtopic":"Performance Evaluation",
                     "difficulty":"L1_Medium","question_type":"Calculation","pitfalls":["Sharpe vs Treynor denominator"]},
            "question":q, "answer":f"Sharpe {sharpe:.3f}; Treynor {treynor:.3f}", "distractors":[f"Sharpe {fmt((rp-rf)/beta)}", f"Sharpe {fmt(sharpe*2)}", f"Treynor {fmt(sharpe)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"rp":rp,"rf":rf,"sig":sig,"beta":beta}}

def port_variance(rng, seq):
    w1 = rng.uniform(0.3, 0.7, 3)
    w2 = 1 - w1
    s1 = rng.uniform(0.10, 0.20, 3)
    s2 = rng.uniform(0.10, 0.25, 3)
    rho = rng.uniform(-0.5, 0.5, 3)
    var = w1**2*s1**2 + w2**2*s2**2 + 2*w1*w2*rho*s1*s2
    sd = math.sqrt(var)
    q = (f"Portfolio: w1 = {w1:.2f}, σ1 = {pct(s1,1)}, w2 = {w2:.2f}, σ2 = {pct(s2,1)}, ρ = {rho:.2f}. "
         f"Compute portfolio variance and SD.")
    tr = (_assume([f"σp² = w1²σ1² + w2²σ2² + 2w1w2ρσ1σ2"]) +
          f"Step 1. w1²σ1² = {w1:.2f}²×{pct(s1,1)}² = {w1**2*s1**2:.4f}.\n"
          f"Step 2. w2²σ2² = {w2:.2f}²×{pct(s2,1)}² = {w2**2*s2**2:.4f}.\n"
          f"Step 3. Cross term = 2×{w1:.2f}×{w2:.2f}×{rho:.2f}×{pct(s1,1)}×{pct(s2,1)} = {2*w1*w2*rho*s1*s2:.4f}.\n"
          f"Step 4. Variance = {fmt(var)}; SD = √ = {pct(sd)}.\n"
          f"Step 5. Trap: ignoring the correlation term treats the portfolio as if assets were uncorrelated.")
    flaw = {"answer": f"{pct(math.sqrt(w1**2*s1**2 + w2**2*s2**2))}", "pitfall": "ignoring correlation",
            "reasoning_trace": (_assume([f"ρ = 0 (uncorrelated)"]) +
            f"Step 1. Dropping the 2w1w2ρσ1σ2 term: variance = {fmt(w1**2*s1**2 + w2**2*s2**2)}, "
            f"SD = {pct(math.sqrt(w1**2*s1**2 + w2**2*s2**2))}. Correlation {rho:.2f} materially changes "
            f"portfolio risk and must be included.")}
    return {"meta": {"topic":"Portfolio Management","subtopic":"Portfolio Risk",
                     "difficulty":"L1_Hard","question_type":"Calculation","pitfalls":["correlation term","diversification"]},
            "question":q, "answer":f"σ² {fmt(var)}; SD {pct(sd)}", "distractors":[f"SD {pct(math.sqrt(w1**2*s1**2 + w2**2*s2**2))}", f"SD {pct(w1*s1+w2*s2)}", f"SD {pct(sd*2)}"],
            "reasoning_trace":tr, "flawed":flaw, "params":{"w1":w1,"s1":s1,"s2":s2,"rho":rho}}

def fi_ytm_approx(rng, seq):
    face = 1000
    coupon_rate = rng.choice([0.04, 0.05, 0.06])
    n = rng.randint(3, 10)
    price = rng.randint(850, 980)
    coupon = face * coupon_rate
    ytm = (coupon + (face - price) / n) / ((face + price) / 2)
    cy = coupon / price
    q = (f"A bond has {face:,} face value, an annual coupon of {coupon_rate*100:.0f}% "
         f"(${coupon:.0f}/yr), {n} years to maturity, and is priced at ${price:,}. "
         f"Estimate its yield to maturity (approx).")
    tr = (_assume([f"annual coupon ${coupon:.0f}", f"{n} years", f"price ${price:,}", f"face ${face:,}"]) +
          f"Step 1. Annual coupon C = {coupon_rate*100:.0f}% × 1000 = ${coupon:.0f}.\n"
          f"Step 2. Average capital gain = (Face − Price)/n = (1000 − {price})/{n} = {(face-price)/n:.2f}.\n"
          f"Step 3. Approx YTM = (C + gain) / avg price = "
          f"({coupon:.0f} + {(face-price)/n:.2f}) / {(face+price)/2:.0f} = {fmt(ytm)}.\n"
          f"Step 4. Trap: current yield = C/Price = {coupon:.0f}/{price} = {fmt(cy)} ignores the "
          f"capital gain, so it is the wrong approximation.")
    flaw = {"answer": f"{fmt(cy)}", "pitfall": "current yield vs YTM",
            "reasoning_trace": (_assume([f"current-yield shortcut"]) +
              f"Step 1. Using current yield C/Price = {coupon:.0f}/{price} = {fmt(cy)} ignores "
              f"the capital gain over {n} years; YTM must also capture (Face − Price)/n.")}
    return {"meta": {"topic": "Fixed Income", "subtopic": "Yield Measures",
                     "difficulty": "L1_Medium", "question_type": "Calculation",
                     "pitfalls": ["current yield vs YTM", "capital gain amortization"]},
            "question": q, "answer": f"{fmt(ytm)}",
            "distractors": [f"{fmt(cy)}", f"{fmt(coupon)}", f"{fmt((face+price)/2)}"],
            "reasoning_trace": tr, "flawed": flaw}


def port_alpha(rng, seq):
    rf = rng.choice([0.03, 0.04, 0.05])
    beta = rng.choice([0.8, 1.2, 1.5])
    rm = rng.choice([0.08, 0.10, 0.12])
    ra = rng.choice([0.06, 0.09, 0.12, 0.14])
    expected = rf + beta * (rm - rf)
    alpha = ra - expected
    q = (f"An asset has a CAPM required return of rf={fmt(rf)}, β={beta:.1f}, "
         f"E(Rm)={fmt(rm)}. Its actual return was {fmt(ra)}. Compute Jensen's alpha.")
    tr = (_assume([f"rf {fmt(rf)}", f"β {beta:.1f}", f"E(Rm) {fmt(rm)}", f"actual {fmt(ra)}"]) +
          f"Step 1. Expected return = rf + β(E(Rm) − rf) = {fmt(rf)} + {beta:.1f}×({fmt(rm)} − {fmt(rf)}) "
          f"= {fmt(expected)}.\n"
          f"Step 2. Alpha = Actual − Expected = {fmt(ra)} − {fmt(expected)} = {fmt(alpha)}.\n"
          f"Step 3. Trap: reporting the CAPM expected return as alpha forgets the actual-return "
          f"differential, so it is the wrong value.")
    flaw = {"answer": f"{fmt(expected)}", "pitfall": "alpha vs CAPM expected return",
            "reasoning_trace": (_assume([f"CAPM-expected shortcut"]) +
              f"Step 1. Reporting the CAPM expected return {fmt(expected)} as if it were alpha "
              f"ignores Jensen's definition α = Actual − Expected.")}
    return {"meta": {"topic": "Portfolio Management", "subtopic": "Performance Evaluation",
                     "difficulty": "L1_Medium", "question_type": "Calculation",
                     "pitfalls": ["alpha vs CAPM expected return", "sign convention"]},
            "question": q, "answer": f"{fmt(alpha)}",
            "distractors": [f"{fmt(expected)}", f"{fmt(ra)}", f"{fmt(rm-rf)}"],
            "reasoning_trace": tr, "flawed": flaw}



def corp_wacc(rng, seq):
    e = rng.randint(40, 70) * 100000
    d = rng.randint(20, 40) * 100000
    re = rng.uniform(0.10, 0.14, 3)
    rd = rng.uniform(0.06, 0.09, 3)
    tax = rng.choice([0.20, 0.25, 0.30])
    we = e / (e + d)
    wd = d / (e + d)
    wacc = we * re + wd * rd * (1 - tax)
    wacc_notax = we * re + wd * rd
    q = (f"A firm has market equity {fmt(e)}, market debt {fmt(d)}, cost of equity "
         f"{pct(re)}, cost of debt {pct(rd)}, and a marginal tax rate {pct(tax)}. "
         f"Compute the WACC.")
    tr = (_assume([f"market-value weights w_e = {pct(we)}, w_d = {pct(wd)}",
                   f"after-tax cost of debt = {pct(rd)}×(1−{tax*100:.0f}%) = {pct(rd*(1-tax))}"]) +
          f"Step 1. Weights: w_e = {pct(we)}, w_d = {pct(wd)}.\n"
          f"Step 2. WACC = w_e·r_e + w_d·r_d(1−t) = {pct(we)}×{pct(re)} + {pct(wd)}×{pct(rd*(1-tax))} = {pct(wacc)}.\n"
          f"Step 3. Trap: forgetting the debt tax shield gives {pct(wacc_notax)} (too high).")
    flaw = {"answer": f"{pct(wacc_notax)}", "pitfall": "tax shield on debt",
            "reasoning_trace": (_assume([f"pre-tax debt cost used"]) +
            f"Step 1. Using {pct(rd)} without (1−t): WACC = {pct(we)}×{pct(re)} + {pct(wd)}×{pct(rd)} = {pct(wacc_notax)}.")}
    return {"meta": {"topic":"Corporate Issuers","subtopic":"Cost of Capital","difficulty":"L1_Medium",
                     "question_type":"Calculation","pitfalls":["tax shield on debt","market-value weights"]},
            "question":q, "answer":f"{pct(wacc)}",
            "distractors":[f"{pct(wacc_notax)}", f"{pct(we*re)}", f"{pct(wd*rd)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"e":e,"d":d,"re":re,"rd":rd,"tax":tax}}

def fsa_current_quick(rng, seq):
    cash = rng.randint(10, 20) * 1000
    ar = rng.randint(20, 30) * 1000
    inv = rng.randint(30, 40) * 1000
    ca = cash + ar + inv
    cl = rng.randint(40, 60) * 1000
    current = ca / cl
    quick = (cash + ar) / cl
    q = (f"A firm has cash {fmt(cash)}, receivables {fmt(ar)}, inventory {fmt(inv)}, "
         f"and current liabilities {fmt(cl)}. Compute the current ratio and the quick ratio.")
    tr = (_assume([f"current assets = {fmt(cash)}+{fmt(ar)}+{fmt(inv)} = {fmt(ca)}",
                   f"quick assets exclude inventory = {fmt(cash)}+{fmt(ar)} = {fmt(cash+ar)}"]) +
          f"Step 1. Current assets = {fmt(ca)}.\n"
          f"Step 2. Current ratio = {fmt(ca)}/{fmt(cl)} = {fmt(current)}.\n"
          f"Step 3. Quick ratio (no inventory) = {fmt(cash+ar)}/{fmt(cl)} = {fmt(quick)}.\n"
          f"Step 4. Trap: keeping inventory in the quick ratio gives {fmt(current)} — too high.")
    flaw = {"answer": f"current {fmt(quick)}; quick {fmt(current)}", "pitfall": "inventory excluded from quick ratio",
            "reasoning_trace": (_assume([f"inventory wrongly kept in quick assets"]) +
            f"Step 1. Quick ratio = {fmt(ca)}/{fmt(cl)} = {fmt(current)} (wrongly keeping inventory).")}
    return {"meta": {"topic":"Financial Statement Analysis","subtopic":"Financial Analysis Techniques","difficulty":"L1_Easy",
                     "question_type":"Calculation","pitfalls":["inventory excluded from quick ratio"]},
            "question":q, "answer":f"current {fmt(current)}; quick {fmt(quick)}",
            "distractors":[f"current {fmt(quick)}; quick {fmt(current)}", f"current {fmt(cl/ca)}; quick {fmt(cl/(cash+ar))}", f"current {fmt(ca)}; quick {fmt(cash+ar)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"cash":cash,"ar":ar,"inv":inv,"cl":cl}}

def deriv_forward_payoff(rng, seq):
    f0 = rng.randint(50, 70) * 100
    st = f0 + rng.randint(2, 8) * 100
    payoff = st - f0
    q = (f"A long forward contract was entered at price {fmt(f0)}. At maturity the spot "
         f"price is {fmt(st)}. Compute the payoff.")
    tr = (_assume([f"long forward payoff = spot − forward price"]) +
          f"Step 1. Payoff = S_T − F_0 = {fmt(st)} − {fmt(f0)} = {fmt(payoff)}.\n"
          f"Step 2. Trap: a short position would pay F_0 − S_T = {fmt(f0-st)}.")
    flaw = {"answer": f"{fmt(f0-st)}", "pitfall": "long vs short direction",
            "reasoning_trace": (_assume([f"short forward payoff used"]) +
            f"Step 1. Short payoff = F_0 − S_T = {fmt(f0)} − {fmt(st)} = {fmt(f0-st)}.")}
    return {"meta": {"topic":"Derivatives","subtopic":"Forward Commitments","difficulty":"L1_Easy",
                     "question_type":"Calculation","pitfalls":["long vs short direction"]},
            "question":q, "answer":f"{fmt(payoff)}",
            "distractors":[f"{fmt(f0)}", f"{fmt(st)}", f"{fmt(f0-st)}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"f0":f0,"st":st}}

def econ_cross_rate(rng, seq):
    eur_usd = rng.uniform(1.10, 1.20, 3)
    usd_gbp = rng.uniform(1.25, 1.35, 3)
    eur_gbp = eur_usd * usd_gbp
    q = (f"The EUR/USD quote is {eur_usd:.4f} and the USD/GBP quote is {usd_gbp:.4f}. "
         f"Compute the cross rate EUR/GBP.")
    tr = (_assume([f"cross rate = EUR/USD × USD/GBP"]) +
          f"Step 1. EUR/GBP = {eur_usd:.4f} × {usd_gbp:.4f} = {eur_gbp:.4f}.\n"
          f"Step 2. Trap: dividing instead of multiplying gives {eur_usd/usd_gbp:.4f}.")
    flaw = {"answer": f"{eur_usd/usd_gbp:.4f}", "pitfall": "multiply vs divide cross rates",
            "reasoning_trace": (_assume([f"division used"]) +
            f"Step 1. EUR/GBP = EUR/USD ÷ USD/GBP = {eur_usd:.4f} ÷ {usd_gbp:.4f} = {eur_usd/usd_gbp:.4f}.")}
    return {"meta": {"topic":"Economics","subtopic":"Currency Exchange Rates","difficulty":"L1_Medium",
                     "question_type":"Calculation","pitfalls":["multiply vs divide cross rates"]},
            "question":q, "answer":f"{eur_gbp:.4f}",
            "distractors":[f"{eur_usd/usd_gbp:.4f}", f"{eur_usd:.4f}", f"{usd_gbp:.4f}"],
            "reasoning_trace":tr, "flawed":flaw,
            "params":{"eur_usd":eur_usd,"usd_gbp":usd_gbp}}

m_corp_wacc = wrap_mcq(corp_wacc)

TEMPLATES = {
    "tvm_annuity_fv": tvm_annuity_fv, "tvm_pv_lump": tvm_pv_lump,
    "tvm_eay": tvm_eay, "tvm_npv_irr": tvm_npv_irr,
    "stats_var_sd": stats_var_sd, "stats_bayes": stats_bayes,
    "stats_ci": stats_ci, "stats_tstat": stats_tstat,
    "econ_elasticity": econ_elasticity, "econ_fisher": econ_fisher,
    "fsa_dupont": fsa_dupont, "fsa_inventory_turnover": fsa_inventory_turnover,
    "eq_gordon": eq_gordon, "eq_pe_earnings": eq_pe_earnings,
    "fi_current_yield": fi_current_yield, "fi_modified_duration": fi_modified_duration,
    "fi_zero_duration": fi_zero_duration,
    "deriv_call_payoff": deriv_call_payoff, "deriv_binomial_call": deriv_binomial_call,
    "alt_hedge_fee": alt_hedge_fee, "alt_cap_rate": alt_cap_rate,
    "port_capm": port_capm, "port_sharpe_treynor": port_sharpe_treynor,
    "port_variance": port_variance,
    "fi_ytm_approx": fi_ytm_approx,
    "port_alpha": port_alpha,
    "v_tvm_annuity_fv": wrap_vignette(tvm_annuity_fv),
    "cr_eq_gordon": wrap_cr(eq_gordon),
    "corp_wacc": corp_wacc,
    "fsa_current_quick": fsa_current_quick,
    "deriv_forward_payoff": deriv_forward_payoff,
    "econ_cross_rate": econ_cross_rate,
    "m_corp_wacc": m_corp_wacc,
}
