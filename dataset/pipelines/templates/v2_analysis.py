"""
Analysis record generators for Cosimo dataset v2.

Each function generates an "analysis"-type record with long-form numerical answers,
trade-offs, and perturbations. See FORMAT.md lines 69-109 for the schema.
"""
import math, json
from pipelines.core import fmt, pct

# Output control: set to None to print to stdout instead
# _OUTFILE = "/tmp/v2_analysis_out.jsonl"
_OUTFILE = None


_registry = {}


def _out(rec):
    s = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    if _OUTFILE:
        with open(_OUTFILE, "a") as f:
            f.write(s + "\n")
    else:
        print(s)
    return rec


def _reg(name, fn):
    """Register a template and emit all its parameters."""
    fn()  # run once to emit records
    pass  # do not add to registry since we already ran it


# Now define generator functions. Each returns metadata dict defining its params.
# But the function itself, when called (with rng, seq), calls _out(...) to emit.

# ---- Equities ----


def eq_fcff_dcf(rng, seq):
    """FCFF DCF valuation with growth perturbations."""
    rev = rng.uniform(500, 2000)
    g = rng.uniform(0.020, 0.045)
    wacc = rng.uniform(0.080, 0.115)
    o_margin = rng.uniform(0.12, 0.25)
    capex_p = rng.uniform(0.05, 0.13)
    nwc_p = rng.uniform(0.03, 0.12)
    tax = rng.uniform(0.21, 0.30)
    n = rng.randint(5, 8)
    fcff0 = rev * o_margin * (1 - tax) - rev * capex_p + rev * nwc_p
    fcf = [fcff0 * (1+g)**i for i in range(n)]
    tv = fcf[-1] * (1+g) / (wacc - g)
    ev0 = sum(fcf[i]/(1+wacc)**i for i in range(n)) + tv/(1+wacc)**n
    g_lo = g * 0.80
    g_hi = min(g * 1.20, 0.055)
    ev_lo = sum(fcf[i]/(1+wacc)**i for i in range(n)) + fcf[-1]*(1+g_lo)/(wacc-g_lo)/(1+wacc)**n
    ev_hi = sum(fcf[i]/(1+wacc)**i for i in range(n)) + fcf[-1]*(1+g_hi)/(wacc-g_hi)/(1+wacc)**n
    pv_exp = round(sum(v/(1+wacc)**i for i, v in enumerate(fcf)), 2)
    pv_tv = round(tv/(1+wacc)**n, 2)
    stem = (f"Mature firm: revenue {fmt(rev)}M, FCFF margin {pct(o_margin)}. "
            f"Growth {pct(g)} for {n} years then perpetuity. WACC {pct(wacc)}. "
            f"Capex {pct(capex_p)} of revenue, NWC {pct(nwc_p)} of revenue, "
            f"tax {pct(tax)}. What EV results and how does a +/- perturbation "
            f"of terminal growth change it?")
    answer = (f"NOPAT = {fmt(rev*o_margin*(1-tax))}M. Capex = {fmt(rev*capex_p)}M. "
              f"NWC add-back = {fmt(rev*nwc_p)}M. "
              f"FCFF0 = {fmt(fcff0)}M. The {n}-year projected stream: "
              f"{[round(x,1) for x in fcf]}M. "
              f"Terminal value at t={n}: {fmt(fcf[-1])}*(1+{pct(g)})/({pct(wacc)}-{pct(g)}) = "
              f"{fmt(tv)}M. PV of explicit FCFF = {fmt(pv_exp)}M. "
              f"PV of TV = {fmt(pv_tv)}M. Enterprise value = {fmt(round(ev0,2))}M. "
              f"TV is {round(pv_tv/ev0*100)}% of total value, so the growth rate assumption "
              f"dwarfs the explicit forecast period. Perturbation at {pct(g_lo)}: "
              f"EV = {fmt(round(ev_lo,2))}M ({pct((ev_lo-ev0)/ev0)} change). "
              f"At {pct(g_hi)}: EV = {fmt(round(ev_hi,2))}M ({pct((ev_hi-ev0)/ev0)} change). "
              f"The key tension: more explicit years reduces TV reliance but increases "
              f"model complexity. With only {wacc-g:.3f} spread between WACC and g, "
              f"terminal value dominates because the Gordon denominator is tight."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Equity Valuation","subtopic":"FCFF DCF",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["terminal value dominates","WACC close to g"]},
         "question":stem,"answer":answer}
    _out(r); return r

def eq_residual_income(rng, seq):
    """Residual income model with ROE normalization."""
    bvps0 = rng.uniform(20, 80)
    roe0 = rng.uniform(0.15, 0.35)
    cost_eq = rng.uniform(0.10, 0.16)
    long_roe = rng.uniform(0.10, 0.18)
    payout = 0.25 + rng.uniform(0, 0.10)
    n = 5
    eps0 = bvps0 * roe0
    rps = []
    bv = bvps0
    for i in range(n):
        eps = bv * roe0 * (1 - (roe0 - long_roe) * i / (n * roe0))
        bv = bv * (1 + eps * (1 - payout) / bv)
        rps.append(eps - cost_eq * bv)
    vi_explicit = sum(rps[i]/(1+cost_eq)**i for i in range(n))
    vi_term1 = sum(rps[i] for i in range(n)) / cost_eq * (1+cost_eq)**(-n)
    vi = bvps0 + vi_explicit
    stem = (f"BPS={fmt(bvps0)}, current ROE={pct(roe0)}, cost of equity={pct(cost_eq)}. "
            f"ROE normalizes toward {pct(long_roe)} over 5 years with {pct(payout)} payout. "
            f"Estimate intrinsic value per share via residual income approach.")
    answer = (f"EPS0 = {fmt(bvps0)} * {pct(roe0)} = {fmt(eps0)}. "
              f"Annual residual earnings: {[round(x,1) for x in rps]}. "
              f"Sum PV of explicit RI = {fmt(round(vi_explicit,1))}. "
              f"BVPS is {fmt(bvps0)}. RI value = "
              f"{fmt(bvps0)} + {fmt(round(vi_explicit,1))} = "
              f"{fmt(round(bvps0 + vi_explicit,1))}. "
              f"The gap between {pct(roe0)} ROE and {pct(cost_eq)} cost of equity "
              f"drives value creation. As ROE converges to {pct(long_roe)}, residual "
              f"earnings narrow. With payout of {pct(payout)}, reinvestment rate "
              f"is {pct(1-payout)} which must earn above the hurdle rate to sustain growth."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Equity Valuation","subtopic":"Residual Income",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["ROE convergence","book value proxy"]},
         "question":stem,"answer":answer}
    _out(r); return r

def eq_comparable_multiples(rng, seq):
    """Comparable company valuation using trading multiples."""
    rev = rng.uniform(800, 1800)
    ebitda = rev * rng.uniform(0.20, 0.35)
    ebit = ebitda * rng.uniform(0.80, 0.90)
    n_et = rng.randint(200, 700)
    mktcap_total = 0
    ev_list = []
    for i in range(5):
        ebitda_i = ebitda * rng.uniform(0.6, 1.4)
        ev_i = ebitda_i * rng.uniform(8.0, 16.0)
        ev_list.append(ev_i / ebitda_i)
    ev_ebitda_median = sorted(ev_list)[2]
    ev_target = ebitda * ev_ebitda_median
    mktcap_target = ev_target * n_et / (n_et + rng.uniform(200, 800))
    stem = (f"Firm: revenue {fmt(rev)}M, EBITDA {fmt(ebitda)}M, EBIT {fmt(ebit)}M. "
            f"5 comps: EV/EBITDA multiples {ev_list}. "
            f"Target has {n_et}M diluted shares. Estimate share price from "
            f"comparable company EV/EBITDA and describe the key assumptions.")
    answer = (f"Comp EV/EBITDA multiples: {[round(x,1) for x in ev_list]}. "
              f"Median multiple = {ev_ebitda_median:.1f}x. "
              f"Implied EV = {fmt(ebitda)} * {ev_ebitda_median:.1f} = {fmt(ev_target)}M. "
              f"Implied market cap (assuming debt-equity ratio = "
              f"{rng.uniform(0.20,0.60):.2f}, net debt ~{n_et*0.3:.0f}M) "
              f"= {fmt(mktcap_target)}M / {n_et}M shares = "
              f"{fmt(mktcap_target/n_et)}. Trading at "
              f"{ev_ebitda_median:.1f}x implies {(ev_target/n_et/(ebit/n_et)):.1f}x EBIT, "
              f"which is {ev_ebitda_median * (1/0.85):.1f}x on average implied EBIT/EBITDA ratio. "
              f"Key assumptions: peer comps are truly comparable, market is fairly "
              f"evaluated, and capital structure is stable. The median multiple reduces "
              f"sensitivity to outliers. Multiple spread of "
              f"{max(ev_list)-min(ev_list):.1f}x captures comp heterogeneity."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Equity Valuation","subtopic":"Comps Multiples",
                 "difficulty":"CFA L1","question_type":"Analysis",
                 "pitfalls":["outlier sensitivity","non-comparable peers"]},
         "question":stem,"answer":answer}
    _out(r); return r

def eq_pvt_company_val(rng, seq):
    """Private company valuation with illiquidity discount."""
    ebits = [rng.uniform(8, 15) for _ in range(3)]
    mean_ebitda = sum(ebits) * rng.uniform(1.15, 1.35) / 3
    pub_multiple = rng.uniform(8.0, 14.0)
    illiq = rng.uniform(0.15, 0.35)
    keyperson = rng.uniform(0.03, 0.10)
    customer_conc = rng.uniform(0.15, 0.50)
    ev = mean_ebitda * pub_multiple * (1 - illiq)
    stem = (f"Private firm: 3-year trailing EBITDA {ebits}. Comparable public "
            f"trades at {pub_multiple:.1f}x EV/EBITDA. Customer concentration "
            f"at {pct(customer_conc)}, key-person risk {pct(keyperson)}, "
            f"and estimated illiquidity discount of {pct(illiq)}. "
            f"What enterprise value and what factors create the biggest uncertainty?")
    median_ev = round(mean_ebitda * pub_multiple * (1 - illiq), 1)
    answer = (f"Mean annual EBITDA = {fmt(mean_ebitda)}M. "
              f"Public equivalent value = {fmt(mean_ebitda)} * {pub_multiple:.1f} = "
              f"{fmt(mean_ebitda * pub_multiple)}M. "
              f"Illiquidity discount of {pct(illiq)}: {fmt(median_ev)}M. "
              f"Dollar value of discount = {fmt(mean_ebitda * pub_multiple - median_ev)}M. "
              f"Customer concentration at {pct(customer_conc)} means revenue loss "
              f"of top client could reduce EBITDA by {pct(customer_conc)}. "
              f"Key-person risk at {pct(keyperson)} may impact retention and "
              f"operations. Alternative approach: DCF discounting private FCFF at "
              f"{pct(rng.uniform(0.14, 0.20))} (higher due to illiquidity premium)."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Equity Valuation","subtopic":"Private Co Val",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["illiquidity discount subjectivity","customer concentration"]},
         "question":stem,"answer":answer}
    _out(r); return r

def eq_dividend_policy(rng, seq):
    """Dividend discount model and policy analysis."""
    eps0 = rng.uniform(3, 10)
    payout0 = rng.uniform(0.20, 0.55)
    div0 = eps0 * payout0
    g_high = rng.uniform(0.08, 0.18)
    n_supernormal = rng.randint(3, 6)
    payout_supernormal = rng.uniform(0.10, 0.25)
    g_stable = rng.uniform(0.03, 0.055)
    cost_eq = rng.uniform(0.10, 0.15)
    eps_supernormal = [eps0 * (1+g_high)**i for i in range(n_supernormal)]
    div_supernormal = [e * payout_supernormal for e in eps_supernormal]
    eps_stable_eps = eps_supernormal[-1] * (1 + g_stable)
    d1_stable = eps_stable_eps * rng.uniform(0.40, 0.70)
    pv_divs = sum(d / (1+cost_eq)**i for i, d in enumerate(div_supernormal))
    if d1_stable > 0:
        p_n = d1_stable / (cost_eq - g_stable)
        pv_terminal = p_n / (1+cost_eq)**n_supernormal
    else:
        pv_terminal = 0
    v0 = pv_divs + pv_terminal
    stem = (f"EPS0 = {fmt(eps0)}, current payout {pct(payout0)}. "
            f"Supernormal growth {pct(g_high)} for {n_supernormal} years "
            f"with {pct(payout_supernormal)} payout, then stable growth {pct(g_stable)} "
            f"with payout rising to {pct(rng.uniform(0.50,0.70))}. "
            f"Cost of equity {pct(cost_eq)}. What is the intrinsic value?")
    st_payout = round(payout_supernormal + (rng.uniform(0.50,0.70) - payout_supernormal) * 0.5, 2)
    answer = (f"Dividends during supernormal: "
              f"{[round(d,2) for d in div_supernormal]}. "
              f"EPS grows from {fmt(eps0)} to {fmt(round(eps_supernormal[-1],2))} during stage. "
              f"Terminal dividend {d1_stable:.2f}, constant growth value at year "
              f"{n_supernormal}: {p_n:.2f}. "
              f"PV of supernormal divs = {fmt(round(pv_divs,2))}. "
              f"PV of terminal value = {fmt(round(pv_terminal,2))}. "
              f"Total value = {fmt(round(v0,2))}. "
              f"Higher payout = higher current value but slower EPS growth. "
              f"Stable payout of {pct(st_payout)} balances growth and income. "
              f"Residual earnings during stable phase: "
              f"{pct(eps_stable_eps/eps0*(1+payout_supernormal) - cost_eq)} vs cost of {pct(cost_eq)}."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Equity Valuation","subtopic":"Dividend Policy",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["residual growth from payout","terminal value sensitivity"]},
         "question":stem,"answer":answer}
    _out(r); return r

def eq_revenue_based(rng, seq):
    """Revenue-based valuation for high-growth companies."""
    rev = rng.uniform(100, 800)
    rev_growth = [rng.uniform(0.25, 0.60) for _ in range(4)]
    margin_ramp = [rng.uniform(0, 0.05) + i * 0.03 for i in range(4)]
    long_rev = rev * (1+rev_growth[0]) * (1+rev_growth[1]) * (1+rev_growth[2]) * (1+rev_growth[3])
    stable_margin = rng.uniform(0.08, 0.18)
    pub_rev_multiple = rng.uniform(2.0, 6.0)
    implied_ev = long_rev * stable_margin * pub_rev_multiple * rng.uniform(0.80, 1.20)
    stem = (f"SaaS firm: revenue {fmt(rev)}M, 4-year growth {rev_growth}. "
            f"Operating margin ramps from {[round(m,2) for m in margin_ramp]} toward "
            f"{pct(stable_margin)}. Comparable SaaS trades at {pub_rev_multiple:.1f}x revenue. "
            f"Estimate implied enterprise value.")
    answer = (f"Year-4 revenue = {fmt(round(long_rev,0))}M. "
              f"At stable margin {pct(stable_margin)}, EBITDA ~ "
              f"{fmt(long_rev * stable_margin)}M. Applying comparable "
              f"multiple of {pub_rev_multiple:.1f}x revenue: "
              f"{fmt(implied_ev)}M. The growth trajectory must be sustainable "
              f"- current 4-year CAGR of {pct((long_rev/rev)**0.25 - 1)} needs network "
              f"effects or moat defense. Margins at scale depend on fixed cost absorption "
              f"and gross margin stability. The key variable is whether the {pub_rev_multiple}x "
              f"multiple is appropriate for a company with {pct(rev_growth[0])} growth at the "
              f"low end of the observed range."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Equity Valuation","subtopic":"Revenue-Based",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["growth sustainability","multiple selection"]},
         "question":stem,"answer":answer}
    _out(r); return r

def eq_asset_based(rng, seq):
    """Asset-based valuation and sum-of-the-parts."""
    op_income = rng.uniform(30, 90)
    op_multiple = rng.uniform(8, 14)
    invest_income = rng.uniform(5, 20)
    real_estate_mv = rng.uniform(50, 200)
    real_estate_bv = real_estate_mv * rng.uniform(0.60, 0.90)
    other_assets = rng.uniform(20, 80)
    net_debt = rng.uniform(40, 150)
    op_ev = op_income * op_multiple
    invest_ev = invest_income * rng.uniform(12, 20)
    sopv = op_ev + invest_ev + real_estate_mv + other_assets - net_debt
    stem = (f"Firm has operations earning {fmt(op_income)}M (comps at "
            f"{op_multiple:.1f}x), investments yielding "
            f"{fmt(invest_income)}M, real estate at BV of "
            f"{fmt(real_estate_bv)}M but estimated MV "
            f"{fmt(real_estate_mv)}M, other assets worth "
            f"{fmt(other_assets)}M, net debt {fmt(net_debt)}M. "
            f"What is the sum-of-the-parts value?")
    answer = (f"Operating business: {fmt(op_income)} * {op_multiple:.1f} = "
              f"{fmt(op_ev)}M. Investments: "
              f"{fmt(invest_income)} * {rng.uniform(12,20):.1f} = "
              f"{fmt(round(invest_ev,1))}M (typical dividend yield multiple). "
              f"Real estate revaluation: {fmt(real_estate_mv)}M MV vs "
              f"{fmt(real_estate_bv)}M BV = {fmt(real_estate_mv - real_estate_bv)}M uplift. "
              f"Net: {fmt(op_ev)} + {fmt(round(invest_ev,1))} + {fmt(real_estate_mv)} + "
              f"{fmt(other_assets)} - {fmt(net_debt)} = "
              f"{fmt(sopv)}M. Key question: which parts can be sold or spun off, "
              f"and is the market underpricing standalone business value as a whole? "
              f"A holding company discount of 10-30% may be appropriate due to complexity."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Equity Valuation","subtopic":"Asset-Based",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["real estate revaluation","holding company discount"]},
         "question":stem,"answer":answer}
    _out(r); return r



# ---- Fixed Income ----

def fi_yield_curve(rng, seq):
    """Yield curve construction from bootstrapping spot rates."""
    coupon_bond1 = r"2.0%", 1
    coupon_bond2 = r"2.5%", 2
    zero1 = rng.uniform(0.025, 0.035)
    zero2_raw = zero1 + rng.uniform(0.005, 0.030)
    # Bootstrap: for 2Y bond with coupon c, price = c/(1+z1) + (100+c)/(1+z2)^2
    p2 = 100 * (1 - rng.uniform(0.005, 0.025))
    z2 = ((100 * (1 + zero2_raw * 2)) / (p2 - 2/(1+zero1)) )**(0.5) - 1
    z3 = zero1 + rng.uniform(0.020, 0.050)
    z4 = z3
    z5 = z3 + rng.uniform(0.005, 0.025)
    rates = [zero1, z2, z3, z4, z5]
    curve_shape = rng.choice(["normal", "inverted", "humped"])
    avg_fwd = sum((rates[i] - rates[0])/(i+1) for i in range(1,5)) / 4 if len(rates)==5 else 0

    stem = (f"Zero curve: 1Y={pct(zero1)}, 2Y bootstrapped to {pct(z2)}, "
            f"2Y={pct(z3)}, 3Y={pct(z4)}, 4Y={pct(z5)}. "
            f"Curve shape is {curve_shape}. What are the 1x2, 2x3, 3x4 forward rates? "
            f"How does the curve slope affect the strategy of riding the yield curve?")
    fwd12 = ((1+z2)**2 / (1+zero1) - 1)
    fwd23 = ((1+z3)**3 / (1+z2)**2 - 1)
    fwd34 = ((1+z4)**3 / (1+z3)**2 - 1)
    fwd45 = ((1+z5)**4 / (1+z4)**3 - 1) if len(rates)>4 else 0
    answer = (f"Forward rates from the bootstrap curve: "
              f"1x2 = {pct(fwd12)}, 2x3 = {pct(fwd23)}, "
              f"3x4 = {pct(fwd34)}, 4x5 = {pct(fwd45)}. "
              f"Under a normal curve, riding the roll-down generates alpha: holding a 5Y bond, "
              f"each year it rolls 1Y closer and is priced at the higher short end yield. "
              f"Roll-down return for year 1: {pct(z5 - fwd45)}. A curve steepener benefits "
              f"barbell strategies while an inverser favors ladder or bar strategies with "
              f"shorter duration. The forward rates embed market expectations: if realized "
              f"rates differ systematically from forwards, the curve was mispriced."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Fixed Income","subtopic":"Yield Curve",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["forward rates vs realized","roll-down mechanics"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fi_duration_convexity(rng, seq):
    """Duration and convexity analysis for bond immunization."""
    coupon = rng.uniform(0.03, 0.07)
    ytm = rng.uniform(0.04, 0.08)
    mat = rng.randint(5, 20)
    n_periods = mat * 2
    cpn_per = ytm / 2
    cpm_per = coupon / 2
    # Modified duration approximation
    price_base = sum(cpn_per/100*100/(1+cpn_per)**i for i in range(1,n_periods+1)) + 100/(1+cpn_per)**n_periods
    dy = 0.005
    price_up = sum(cpn_per/100*100/(1+cpn_per+dy)**i for i in range(1,n_periods+1)) + 100/(1+cpn_per+dy)**n_periods
    price_dn = sum(cpn_per/100*100/(1+cpn_per-dy)**i for i in range(1,n_periods+1)) + 100/(1+cpn_per-dy)**n_periods
    mod_dur = -(price_up - price_dn) / (2 * price_base * dy) * 100
    eff_convexity = (price_up + price_dn - 2*price_base) / (price_base * dy**2) * 100
    price_change_approx = -mod_dur * dy + 0.5 * eff_convexity * dy**2

    stem = (f"Annual coupon {pct(coupon)}, YTM {pct(ytm)}, maturity {mat} years. "
            f"Calculate modified duration, effective convexity, and the estimated price "
            f"change for a 50bp parallel shift. Is duration sufficient for predicting P/L?")
    answer = (f"Price at YTM {pct(ytm)}: {fmt(price_base, 2)}. "
              f"Price at YTM+50bp: {fmt(price_up, 2)}. "
              f"Price at YTM-50bp: {fmt(price_dn, 2)}. "
              f"Modified duration = {fmt(mod_dur, 2)}, effectively "
              f"{fmt(mod_dur/2 * mat * 2, 2)} in semi-annual terms. "
              f"Effective convexity = {fmt(eff_convexity, 2)} (per 100 face). "
              f"Price change estimate for +50bp: "
              f"-{fmt(mod_dur,2)}*0.005 + 0.5*{fmt(eff_convexity,2)}*0.005^2 = "
              f"{fmt(price_change_approx, 4)}. "
              f"Actual: {fmt((price_up - price_base)/price_base*100, 2)}%. "
              f"Convexity adjustment accounts for the curvature: linear duration "
              f"underestimates gains and overestimates losses. Bond A with duration 5 and "
              f"convexity 40 vs bond B with duration 5 and convexity 80: same duration "
              f"but B outperforms in any rate movement due to higher convexity. A "
              f"swaption buyer prefers positive convexity."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Fixed Income","subtopic":"Duration & Convexity",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["duration only linear approximation","convexity benefit"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fi_bond_pricing(rng, seq):
    """Full bond pricing with embedded call/put analysis."""
    coupon = rng.uniform(0.04, 0.08)
    yield_clean = rng.uniform(0.04, 0.09)
    mat = rng.randint(3, 10)
    periods = mat * 2
    cpn = coupon / 2 * 100
    y_per = yield_clean / 2
    price = sum(cpn / (1+y_per)**i for i in range(1, periods+1)) + 100/(1+y_per)**periods
    # Accrued interest
    days_accrued = rng.randint(10, 180)
    days_total = rng.choice([181, 182, 183, 184, 365, 366])
    accrued = round(cpn * days_accrued / days_total, 2)
    full_price = price + accrued

    stem = (f"{mat}Y bond: coupon {pct(coupon)}, yield to maturity {pct(yield_clean)}. "
            f"{days_accrued}/{days_total} days into coupon period. "
            f"What is clean price, full price, accrued interest, and how does the "
            f"clean price convention affect trading comparability?")
    answer = (f"Full price (dirty): {fmt(full_price, 2)}. "
              f"Accrued interest: {fmt(cpn)} * {days_accrued}/{days_total} = "
              f"{fmt(accrued)}. "
              f"Clean price: {fmt(full_price - accrued, 2)}. "
              f"Clean price convention removes accrued interest so bond prices "
              f"stay continuous between coupons. Without it, the bond price would "
              f"drop by the full coupon amount at each payment date, which is "
              f"discontinuous and harder to analyze trends. Two identical bonds "
              f"traded with different settlement dates will have the same clean price "
              f"but different full prices - the buyer at the later settlement "
              f"pays accrued interest to compensate the seller."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Fixed Income","subtopic":"Bond Pricing",
                 "difficulty":"CFA L1","question_type":"Analysis",
                 "pitfalls":["clean vs dirty price","day count conventions"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fi_credit_spreads(rng, seq):
    """Credit spread analysis and default probability."""
    gov_yield = rng.uniform(0.03, 0.06)
    corpcoupon = rng.uniform(0.045, 0.10)
    corpprice = rng.uniform(90, 103)
    # YTM of corporate bond
    ytm = 0  # approximate: solve for YTM given price
    mat = 5
    c = corpcoupon * 100
    # Simple YTM approximation
    ytm_approx = (c + (100 - corpprice)/mat) / ((100 + corpprice)/2) / 100
    spread = ytm_approx - gov_yield
    # Risk-neutral default probability
    recovery = rng.uniform(0.30, 0.50)
    avg_p = spread / (1 - recovery)
    annual_pd = avg_p / mat

    stem = (f"5Y corporate bond: coupon {pct(corpcoupon)}, priced at {fmt(corpprice,1)} "
            f"(of 100). 5Y government yield {pct(gov_yield)}. What is the yield spread? "
            f"What implied annual default probability does this spread encode "
            f"(given recovery rate {pct(recovery)})? How does this differ from historical PD?")
    answer = (f"Corporate YTM: {pct(ytm_approx)}. "
              f"Nominal spread = {pct(spread)}. "
              f"Risk-neutral default probability with recovery {pct(recovery)}: "
              f"PD = spread / (1-recovery) / {mat} = "
              f"{spread} / (1-{recovery}) / {mat} = {fmt(annual_pd, 6)}. "
              f"Annualized cumulative: {fmt(1-(1-annual_pd)**mat)}. "
              f"The risk-neutral PD embeds risk aversion premium, not just expected losses. "
              f"Historical PDs for BB-rated corporate bonds are typically below the "
              f"implied risk-neutral rates because investors demand compensation for "
              f"tail risk and liquidity risk. The spread between risk-neutral and "
              f"physical PD is the risk premium. In a stress scenario, the actual PD "
              f"may exceed the risk-neutral estimate by a factor of 2-5x."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Fixed Income","subtopic":"Credit Spreads",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["risk-neutral vs physical PD","spread decomposition"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fi_convertibles(rng, seq):
    """Convertible bond valuation: convertible parity and bond floor."""
    strike = rng.uniform(25, 80)
    stock = rng.uniform(20, 100)
    conv_ratio = 100 / strike
    bond_price = rng.uniform(92, 100)
    stock_price = rng.uniform(strike * 0.7, strike * 1.3)
    parity = conv_ratio * stock_price
    conv_premium = (bond_price / parity - 1) if parity > 0 else 0
    option_val = bond_price - (parity if bond_price > parity else bond_price * 0.5)
    stem = (f"Convertible: strike {strike}, par value 100, bond price {fmt(bond_price)}. "
            f"Underlying stock at {fmt(stock_price)}. "
            f"What is conversion parity, conversion premium, option value? "
            f"How does the convertible behave at different stock levels?")
    answer = (f"Conversion ratio = 100/{strike} = {conv_ratio:.2f}. "
              f"Conversion parity = {conv_ratio:.2f} * {stock_price:.2f} = "
              f"{fmt(parity, 2)}. "
              f"Conversion premium = {fmt(bond_price)}/{fmt(parity)} - 1 = "
              f"{fmt(conv_premium * 100, 2)}%. "
              f"Pure bond value (option-free) should be near {fmt(bond_price)} at the "
              f"governement rate, so the option component is {fmt(option_val, 2)}. "
              f"At stock = strike: parity = bond value at par, convertible = straight bond. "
              f"Stock doubles: parity = {fmt(2*parity)}, premium compresses. "
              f"Stock halves: parity = {fmt(parity*0.5)}, floor protects downside. "
              f"The convertible investor accepts lower coupon for equity upside optionality "
              f"bounded by the bond floor. The implicit put: the bondholder can force conversion "
              f"when stock is above strike or hold for maturity to receive par."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Fixed Income","subtopic":"Convertibles",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["parity vs bond value","implicit option features"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fi_inflation_linked(rng, seq):
    """TIPS pricing and real vs nominal yield analysis."""
    nominal_y = rng.uniform(0.03, 0.065)
    real_y = rng.uniform(0.005, 0.035)
    term = rng.randint(5, 30)
    breakeven = (1 + nominal_y) / (1 + real_y) - 1
    # TIPS price with adjusted principal
    adjusted_principal = 100 * (1 + rng.uniform(0.005, 0.04))
    tip_price = adjusted_principal * sum(real_y/2/(1+real_y/2)**i for i in range(1, term*2+1)) + adjusted_principal/(1+real_y/2)**(term*2)
    nominal_price = sum(nominal_y/2/(1+nominal_y/2)**i for i in range(1, term*2+1)) + 100/(1+nominal_y/2)**(term*2)
    real_return_nominal = nominal_price / tip_price * (adjusted_principal/100 - 1)

    stem = (f"Nominal bond yield {pct(nominal_y)}, TIPS real yield {pct(real_y)}, "
            f"{term}Y maturity. What is the breakeven inflation rate? "
            f"If actual inflation averages {pct(breakeven)} over the period "
            f"(the breakeven), what is the real return? "
            f"How would you position between nominal and TIPS?")
    breakeven_pct = round(breakeven * 100, 2)
    answer = (f"Breakeven inflation rate: {(1+nominal_y)/(1+real_y)-1:.4f} = "
              f"{pct(breakeven, 2)}. "
              f"TIPS price at real yield {pct(real_y)}: {fmt(round(tip_price, 2))}. "
              f"Nominal bond price: {fmt(round(nominal_price, 2))}. "
              f"If inflation runs exactly at breakeven, both bonds deliver equal nominal return. "
              f"TIPS: principal adjusts with CPI so nominal payments grow with inflation. "
              f"Real return is locked at {pct(real_y)} regardless of inflation path. "
              f"Nominal bond: real return is uncertain - falls if inflation > {breakeven_pct}%. "
              f"The TIPS premium (real yield below nominal) reflects the inflation insurance value. "
              f"Positioning: buy TIPS if you expect inflation > breakeven, buy nominal if "
              f"inflation < breakeven or if you expect deflation (where nominal bonds gain)."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Fixed Income","subtopic":"Inflation Linked",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["breakeven != expected inflation","deflation floor"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fi_mbs_analysis(rng, seq):
    """MBS prepayment risk and duration analysis."""
    coupon = rng.uniform(0.04, 0.07)
    current_rate = rng.uniform(0.03, 0.06)
    price = rng.uniform(95, 105)
    cur_dur = rng.uniform(3.5, 7.0)
    # Extension risk: rates fall, prepayments slow
    ext_dur = cur_dur * rng.uniform(1.2, 1.8)
    # Contraction risk: rates rise, prepayments accelerate
    con_dur = cur_dur * rng.uniform(0.5, 0.8)
    cpr = rng.uniform(0.06, 0.15)
    cpr_low = cpr * 0.6 if current_rate > 0.03 else cpr
    cpr_high = cpr * 1.4 if current_rate < 0.06 else cpr

    stem = (f"MBS: coupon {pct(coupon)}, current price {fmt(price, 1)}, "
            f"effective duration {fmt(cur_dur, 1)}. "
            f"CPR is {pct(cpr)}. Under rates falling 100bp, duration extends to "
            f"{fmt(ext_dur, 1)}. Under rates rising 100bp, duration contracts to "
            f"{fmt(con_dur, 1)}. How does prepayment risk make MBS duration asymmetric?")
    answer = (f"Contraction phase: rates fall below {pct(coupon)} and borrowers "
              f"refinance at lower rates, shortening MBS life. Duration compresses from "
              f"{fmt(cur_dur, 1)} to {fmt(con_dur, 1)}x (a contraction of "
              f"{pct((cur_dur - con_dur)/cur_dur)}x - price sensitivity drops "
              f"precisely when bond prices should gain most). "
              f"Extension phase: rates jump above {pct(coupon)}, refinancers hold "
              f"high-coupon MBS. Duration extends to {fmt(ext_dur, 1)}x and price "
              f"declines MORE than duration alone predicts. "
              f"CPR reduction from {pct(cpr)} to {pct(cpr_low)} when rates fall "
              f"means cash flows extend. CPR of {fmt(cpr_high)} in a rising rate "
              f"environment accelerates principal returns, capping price upside. "
              f"The negative convexity means MBS has a price cap at the call boundary - "
              f"unlike option-free bonds that gain symmetrically. This makes hedging "
              f"with duration alone inadequate."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Fixed Income","subtopic":"MBS",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["negative convexity","prepayment model uncertainty"]},
         "question":stem,"answer":answer}
    _out(r); return r



# ---- Quantitative Methods ----

def qs_time_series(rng, seq):
    """Time series forecasting: AR and moving average models."""
    n_obs = rng.randint(50, 100)
    mean_ret = rng.uniform(0.0001, 0.0005)
    volatility = rng.uniform(0.008, 0.025)
    ar_coeff = rng.uniform(0.1, 0.5)
    residuals = [rng.uniform(-2, 2) * volatility for _ in range(n_obs)]
    series = [mean_ret]
    for i in range(1, len(residuals)):
        series.append(mean_ret + ar_coeff * series[-1] + residuals[i])
    half = n_obs // 2
    train_mean = sum(series[:half]) / half
    actual_mean = sum(series) / n_obs
    forecast_error = [series[half+i] - train_mean for i in range(len(series)-half)]
    rmse = (sum(e**2 for e in forecast_error)/len(forecast_error))**0.5
    mape = sum(abs(e/s) for e, s in zip(forecast_error, series[half:]) if abs(s) > 0.0001) / len(forecast_error) * 100

    stem = (f"{n_obs} monthly returns: mean {pct(mean_ret)} per period, "
            f"AR(1) coefficient {ar_coeff:.3f}. Training on first half, "
            f"testing on second half with naive forecast (constant = train mean). "
            f"What is the RMSE and what does the AR(1) structure imply "
            f"compared to a random walk model?")
    answer = (f"Train period mean: {fmt(train_mean, 6)}. "
              f"Full sample mean: {fmt(actual_mean, 6)}. "
              f"RMSE of naive forecast = {fmt(rmse, 6)}. "
              f"MAPE: {mape:.2f}%. "
              f"The AR(1) structure implies mean reversion: returns gravitate back to "
              f"the mean at rate (1-ar_coeff) = {(1-ar_coeff):.3f} per period. A random "
              f"walk would predict no mean reversion, making the constant-mean forecast "
              f"less useful. The R-squared of AR(1) is approximately {ar_coeff**2:.4f}, "
              f"meaning {ar_coeff**2*100:.1f}% of variance is predictable. "
              f"In practice, time-varying volatility is the bigger concern: GARCH models "
              f"capture clustering that AR(1) misses. The half-sample out-of-test is "
              f"sufficient but a longer test period would reduce forecast error uncertainty."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Quantitative Methods","subtopic":"Time Series",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["non-stationarity","overfitting on short periods"]},
         "question":stem,"answer":answer}
    _out(r); return r

def qs_bayesian(rng, seq):
    """Bayesian inference: updating probability with new evidence."""
    prior_prob = rng.uniform(0.20, 0.50)
    acc_rate_true = rng.uniform(0.70, 0.90)
    acc_rate_false = rng.uniform(0.10, 0.30)
    likelihood_true = acc_rate_true
    likelihood_false = acc_rate_false
    # Posterior: P(true|positive) = P(Pos|True)P(True) / [P(Pos|True)P(True) + P(Pos|False)P(False)]
    posterior = likelihood_true * prior_prob / (likelihood_true * prior_prob + likelihood_false * (1 - prior_prob))
    posterior_odds = posterior / (1 - posterior)
    prior_odds = prior_prob / (1 - prior_prob)
    bayes_factor = likelihood_true / likelihood_false

    stem = (f"Prior probability of model alpha = {pct(prior_prob)}. "
            f"Given the model generates "
            f"correct returns {pct(acc_rate_true)} of the time when alpha is genuine, "
            f"but also generates correct returns {pct(acc_rate_false)} of the time "
            f"when alpha is spurious (false positive). After one quarter of "
            f"correct returns, what is the updated probability?")
    answer = (f"Prior odds: prior_prob/(1-prior_prob) = {prior_odds:.3f}. "
              f"Bayes factor: {acc_rate_true}/{acc_rate_false} = {bayes_factor:.3f}. "
              f"Posterior odds: {prior_odds:.3f} * {bayes_factor:.3f} = "
              f"{prior_odds * bayes_factor:.3f}. "
              f"Posterior probability = posterior = {posterior:.4f} = "
              f"{pct(posterior, 2)}. "
              f"Prior to posterior: {pct(prior_prob)} to {pct(posterior)}. "
              f"The single observation shifts probability by "
              f"{pct(posterior-prior_prob)} in one quarter. Each additional confirmed "
              f"period compounds the Bayes factor: two quarters at correct rates would "
              f"make posterior = {acc_rate_true**2 * prior_prob / (acc_rate_true**2 * prior_prob + acc_rate_false**2 * (1 - prior_prob)):.4f}. "
              f"Prior is critical: a low prior {pct(prior_prob)} means many "
              f"observations may still not push posterior above 0.5, which is the "
              f"practical threshold for treating a strategy as valid."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Quantitative Methods","subtopic":"Bayesian",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["prior sensitivity","base rate neglect"]},
         "question":stem,"answer":answer}
    _out(r); return r

def qs_monte_carlo(rng, seq):
    """Monte Carlo simulation for portfolio return distributions."""
    e_ret = rng.uniform(0.05, 0.12)
    volatility = rng.uniform(0.12, 0.25)
    horizon = rng.randint(5, 15)
    n_sim = 10000
    port_mean = (1 + e_ret)**horizon - 1
    port_vol = volatility * horizon**0.5
    # Simulate using simple normal approximation for the portfolio
    import numpy as np
    s = np.random.RandomState(42)
    final_values = s.lognormal(mean=np.log(1+e_ret)*horizon - 0.5*volatility**2*horizon,
                                sigma=volatility*horizon**0.5, size=n_sim)
    final_values = [max(x, 0) for x in final_values]
    median_fv = sorted(final_values)[n_sim // 2]
    p5 = sorted(final_values)[int(0.05 * n_sim)]
    p95 = sorted(final_values)[int(0.95 * n_sim)]
    avg_fv = sum(final_values) / n_sim

    stem = (f"Portfolio: annual return {pct(e_ret)}, vol {pct(volatility)}. "
            f"Horizon {horizon}Y. Run monte carlo with {n_sim} sims. "
            f"What is the median, 5th, and 95th percentile terminal value? "
            f"Compare to expected value vs arithmetic average approach.")
    answer = (f"Median final value: {fmt(median_fv, 2)} (50th percentile). "
              f"5th percentile: {fmt(p5, 2)} (Value at Risk equivalent). "
              f"95th percentile: {fmt(p95, 2)}. "
              f"Arithmetic expected value: {fmt(avg_fv, 2)}. "
              f"Difference (avg - median) = {fmt(avg_fv - median_fv, 2)}. "
              f"This gap reflects lognormal skew: the mean is pulled up by the "
              f"right tail (outcomes above median). The median is the preferred "
              f"summary statistic for long-horizon investing because: (1) it "
              f"represents the most likely outcome, (2) it does not overstate "
              f"typical outcomes the way the arithmetic mean does. The median "
              f"of {fmt(median_fv, 2)} is {(median_fv-1)*100:.0f}% growth, while "
              f"arithmetic average suggests {avg_fv*100-100:.0f}%."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Quantitative Methods","subtopic":"Monte Carlo",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["mean vs median","distributional assumptions"]},
         "question":stem,"answer":answer}
    _out(r); return r

def qs_regression(rng, seq):
    """Multiple regression analysis with interpretation."""
    r_squared = rng.uniform(0.30, 0.75)
    n_obs = rng.randint(50, 200)
    n_params = rng.randint(2, 5)
    adj_rsq = r_squared - (1 - r_squared) * (n_params - 1) / (n_obs - n_params - 1)
    f_stat = adj_rsq / (1 - r_squared) * (n_obs - n_params - 1) / n_params
    rmse = rng.uniform(0.008, 0.025)
    coef_vals = [rng.uniform(-0.5, 0.5) for _ in range(n_params)]
    se_vals = [abs(c) * rng.uniform(0.2, 0.5) + 0.01 for c in coef_vals]
    t_stats = [abs(c)/s if s > 0 else 0 for c, s in zip(coef_vals, se_vals)]

    stem = (f"Multiple regression: {n_obs} observations, {n_params} independent "
            f"variables. R-squared = {r_squared:.4f}, adjusted R-squared = "
            f"{adj_rsq:.4f}, F-stat = {f_stat:.2f}, RMSE = {fmt(rmse, 4)}. "
            f"Coefficients: {coef_vals}, SEs: {[round(s, 4) for s in se_vals]}. "
            f"Interpret the model fit and which variables are statistically significant.")
    answer = (f"R-squared = {r_squared:.4f} means "
              f"{r_squared*100:.1f}% of dependent variable variance explained. "
              f"Adjusted R-squared = {adj_rsq:.4f}, accounting for "
              f"{n_params} predictors. F-stat = {f_stat:.2f} > 3.84 "
              f"(F(2,{n_obs-6}) at 5%), so at least one coefficient is non-zero. "
              f"T-statistics: {[round(t, 2) for t in t_stats]}. "
              f"At 5% level (threshold ~1.96 for large n), variables with "
              f"t-stat > 2.0 are significant. "
              f"Key issue: high R-squared doesn't guarantee causality. "
              f"With {n_obs} obs and {n_params} params, degrees of freedom = "
              f"{n_obs - n_params - 1}. Multi-collinearity between predictors "
              f"can inflate SEs without reducing R-squared. The adjusted R-squared "
              f"penalizes adding irrelevant variables: {adj_rsq:.4f} vs {r_squared:.4f}."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Quantitative Methods","subtopic":"Regression",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["spurious correlation","multi-collinearity"]},
         "question":stem,"answer":answer}
    _out(r); return r

def qs_hypothesis_testing(rng, seq):
    """Hypothesis testing for investment strategy performance."""
    sample_mean = rng.uniform(0.0003, 0.0015)
    pop_mean_null = 0.0002
    std_err = rng.uniform(0.0002, 0.0008)
    n = rng.randint(30, 120)
    t_stat = (sample_mean - pop_mean_null) / std_err
    p_value_approach = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    alpha = 0.05 if rng.randint(0, 1) == 0 else 0.01

    stem = (f"Strategy: {n} monthly returns, mean {pct(sample_mean)}, "
            f"SE of mean {fmt(std_err, 6)}. Null hypothesis: mean return = "
            f"{pct(pop_mean_null)}. What is the t-stat, p-value, and does the "
            f"strategy earn true alpha at alpha={pct(alpha)} significance?")
    answer = (f"t-stat = ({fmt(sample_mean, 6)} - {fmt(pop_mean_null, 6)}) / "
              f"{fmt(std_err, 6)} = {t_stat:.3f}. "
              f"With {n-1} degrees of freedom, critical value at 5% (two-tailed) "
              f"is approximately 1.96. P-value = {p_value_approach:.4f}. "
              f"At {pct(alpha)}, if t_stat > 1.96 and p < {alpha}, reject null. "
              f"With {t_stat:.3f} {'vs' if t_stat > 0 else ''} critical value "
              f"{1.96 if alpha == 0.05 else 2.58} and p-value "
              f"{p_value_approach:.4f} {'< ' if p_value_approach < alpha else '= '}"
              f"{alpha}: the strategy {'does' if p_value_approach < alpha else 'does not'} "
              f"earn statistically significant alpha. Key consideration: the sample "
              f"size matters for power. With {n} months, you need true mean "
              f"exceeding {pct(pop_mean_null + 1.96*std_err)} to achieve 80% power."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Quantitative Methods","subtopic":"Hypothesis Testing",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["p-value misinterpretation","look-ahead bias"]},
         "question":stem,"answer":answer}
    _out(r); return r

def qs_garch(rng, seq):
    """GARCH volatility modeling: conditional vs unconditional variance."""
    unconditional_vol = rng.uniform(0.15, 0.30)
    omega = rng.uniform(0.00005, 0.0002)
    alpha = rng.uniform(0.05, 0.20)
    beta = rng.uniform(0.70, 0.95)
    # Check stationarity: alpha + beta < 1
    alpha_sum = alpha + beta
    long_run_vol = (omega / (1 - alpha_sum))**0.5
    shock = rng.uniform(-2, 2) * unconditional_vol
    prev_var = unconditional_vol**2
    garch_var = omega + alpha * shock**2 + beta * prev_var
    garch_vol = garch_var**0.5

    stem = (f"GARCH(1,1): omega={omega:.8f}, alpha={alpha:.3f}, beta={beta:.3f}. "
            f"Today shock = {pct(shock/unconditional_vol)}. "
            f"What is today conditional variance? Compare to long-run variance. "
            f"How fast does volatility revert to the mean?")
    answer = (f"Long-run variance: {pct(long_run_vol)} (unconditional). "
              f"Today: variance = {omega:.8f} + {alpha:.3f} * ({pct(shock)})^2 + "
              f"{beta:.3f} * yesterday's variance. "
              f"Conditional variance = {fmt(garch_var, 8)}, "
              f"conditional std dev = {fmt(garch_vol, 6)}. "
              f"Alpha+beta = {alpha_sum:.4f} < 1 (stationary). "
              f"Half-life of shock: 0.5^(1/(alpha+beta)) periods. "
              f"Volatility reverts to mean at rate {1-alpha_sum:.4f} per period "
              f"weighted towards persistent component ({beta:.3f}). "
              f"The GARCH(1,1) structure means: omega is the long-run variance, "
              f"alpha captures the ARCH effect (new information), "
              f"beta captures volatility persistence. With beta of "
              f"{beta:.3f}, roughly {beta*100:.0f}% of yesterday variance "
              f"carries forward. The shock impact: {alpha:.3f} * shock^2 = "
              f"{fmt(alpha*shock**2, 8)}, which adds to the baseline."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Quantitative Methods","subtopic":"GARCH",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["stationarity check","conditional vs unconditional var"]},
         "question":stem,"answer":answer}
    _out(r); return r

def qs_nonparametric(rng, seq):
    """Non-parametric tests and rank-based methods."""
    n = rng.randint(30, 80)
    group1 = [rng.r.gauss(0, 1) for _ in range(n // 2)]
    group2 = [rng.r.gauss(rng.uniform(0, 0.5), 1) for _ in range(n - n//2)]
    # Wilcoxon rank-sum statistic approximation
    combined = sorted(group1 + group2)
    ranks1 = sum(combined.index(x) + 1 for x in group1)
    ranks2 = sum(combined.index(x) + 1 for x in group2)
    w_stat = min(ranks1, ranks2)
    u_stat = ranks1 - (n//2) * (n//2 + 1) / 2
    z_stat = (u_stat - (n//2) * (n-n//2) / 2) / ((n//2) * (n-n//2) / 2 * (n) / 3)**0.5

    stem = (f"Two groups: n1={n//2}, n2={n-n//2}. "
            f"Group 1 values: {group1[:4]}. Group 2: {group2[:4]}. "
            f"Distributions are non-normal. What non-parametric test should be used? "
            f"What is the Wilcoxon rank-sum statistic and its z-approximation?")
    answer = (f"Wilcoxon rank-sum: U1 = "
              f"sum of ranks for group 1 - n1*(n1+1)/2 = "
              f"{ranks1:.1f} - {n//2*3:.1f} = {fmt(u_stat, 2)}. "
              f"(Ranks: {ranks1:.1f} for group 1, {ranks2:.1f} for group 2.) "
              f"With n = {n}, the distribution of U approaches normal: "
              f"E(U) = n1*n2/2 = {((n//2)*(n-n//2)/2):.1f}, "
              f"Var(U) = n1*n2*(n+1)/12 = {(n//2)*(n-n//2)*(n+1)/12:.1f}. "
              f"z = (U - E(U)) / sqrt(Var(U)) = {z_stat:.3f}. "
              f"At 5% two-sided, |z| > 1.96 indicates significant difference. "
              f"Non-parametric approaches don't assume normality but reduce power; "
              f"the t-test could lose ~95% efficiency in normal conditions but "
              f"gains when distributions are heavy-tailed or skewed."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Quantitative Methods","subtopic":"Nonparametric",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["sample size for normal approx","power loss"]},
         "question":stem,"answer":answer}
    _out(r); return r



# ---- Portfolio Management ----

def pm_factor_model(rng, seq):
    """Factor model decomposition and attribution."""
    e_return = rng.uniform(0.06, 0.14)
    rf = rng.uniform(0.02, 0.05)
    market_rf = e_return - rf
    beta_mkt = rng.uniform(0.8, 1.4)
    beta_smb = rng.uniform(-0.2, 0.3)
    beta_hml = rng.uniform(-0.1, 0.2)
    alpha = rng.uniform(0.002, 0.008)
    r_squared = rng.uniform(0.40, 0.70)
    active_beta = beta_mkt - 1.0
    active_rf = market_rf * beta_mkt + beta_smb * rng.uniform(0.01, 0.06) + beta_hml * rng.uniform(-0.02, 0.02)
    factor_contrib = active_rf
    specific_return = e_return - (rf + beta_mkt * market_rf + beta_smb * 0.03 + beta_hml * 0.01)
    tracking_error_mkt = abs(active_beta) * rng.uniform(0.08, 0.15)
    tracking_error_total = (r_squared * tracking_error_mkt**2 + (1-r_squared) * rng.uniform(0.02, 0.05)**2)**0.5

    stem = (f"Portfolio: annual return {pct(e_return)}, risk-free {pct(rf)}. "
            f"Market excess = {pct(market_rf)}. Factor exposures: "
            f"market beta {beta_mkt:.2f}, SMB {beta_smb:.2f}, HML "
            f"{beta_hml:.2f}. Alpha = {pct(alpha)}. R-squared = {r_squared:.2f}. "
            f"Break down the return into factor contributions, alpha, and idiosyncratic return.")
    answer = (f"Market factor: {beta_mkt:.2f} * {pct(market_rf)} = {pct(beta_mkt * market_rf)}. "
              f"SMB factor: {beta_smb:.2f} * {pct(0.03)} = {pct(beta_smb * 0.03)}. "
              f"HML factor: {beta_hml:.2f} * {pct(0.01)} = {pct(beta_hml * 0.01)}. "
              f"Total factor return = "
              f"{pct(beta_mkt * market_rf + beta_smb * 0.03 + beta_hml * 0.01)}. "
              f"Alpha (Jensen) = {pct(alpha)}, implying {pct(alpha*12)} monthly. "
              f"Idiosyncratic return = {pct(specific_return)}. "
              f"R-squared = {r_squared:.2f}: {r_squared*100:.0f}% of variance explained by "
              f"factors, remaining {(1-r_squared)*100:.0f}% is specific. "
              f"Active risk from market tilt: "
              f"{pct(abs(active_beta) * 0.10)} vs total tracking error "
              f"{pct(tracking_error_total)}. The factor model decomposes both "
              f"returns and risks. Active weight in factor space: market beta "
              f"excess = {active_beta:.2f}, which implies an aggressive posture "
              f"relative to a cap-weighted benchmark with beta = 1.0."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Portfolio Management","subtopic":"Factor Models",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["factor proxy errors","non-stationary betas"]},
         "question":stem,"answer":answer}
    _out(r); return r

def pm_rebalancing(rng, seq):
    """Portfolio rebalancing: drift analysis and cost-benefit."""
    target_eq = rng.uniform(0.45, 0.70)
    target_bond = 1 - target_eq
    eq_return = rng.uniform(0.10, 0.25)
    bond_return = rng.uniform(0.02, 0.06)
    vol_eq = rng.uniform(0.15, 0.25)
    vol_bond = rng.uniform(0.03, 0.08)
    corr = rng.uniform(0.0, 0.3)
    # After 1 year
    eq_weight = target_eq * (1 + eq_return)
    bond_weight = target_bond * (1 + bond_return)
    total = eq_weight + bond_weight
    actual_eq = eq_weight / total
    drift = actual_eq - target_eq
    rebalance_pct = abs(drift) / target_eq * 100

    stem = (f"Target allocation: {pct(target_eq)} equity, {pct(target_bond)} bonds. "
            f"After {pct(eq_return)} equity and {pct(bond_return)} bond returns "
            f"over one year, what is the drift? How large must the drift be to "
            f"justify rebalancing costs?")
    answer = (f"Equity weight after returns: {pct(target_eq * (1+eq_return / total))}. "
              f"Bond weight: {pct(target_bond * (1 + bond_return) / total)}. "
              f"Actual equity share: {pct(actual_eq)}. "
              f"Drift from target: {pct(drift * 100)} percentage points, "
              f"or {rebalance_pct:.1f}% relative to target. "
              f"Portfolio return = {pct(total-1)} vs target-weighted "
              f"{pct(target_eq * eq_return + target_bond * bond_return)}. "
              f"Difference = rebalancing drag if we rebalanced at start (we missed "
              f"the equity rally). The key question: does the rebalance gain from "
              f"selling high and buying low exceed the transaction costs? With "
              f"drift of {drift:.3f} and estimated cost of "
              f"{rebalance_pct:.1f}% in turn costs, the break-even is: if the "
              f"expected reversion is greater than the cost, rebalance. The Sharpe "
              f"ratio of the active strategy (rebalancing vs buy-and-hold) depends "
              f"on the correlation regime and mean reversion properties of the "
              f"equity-bond relationship."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Portfolio Management","subtopic":"Rebalancing",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["rebalancing drag","transaction cost threshold"]},
         "question":stem,"answer":answer}
    _out(r); return r

def pm_performance_attribution(rng, seq):
    """Brinson attribution: allocation, selection, and interaction effects."""
    equity_alloc = rng.uniform(0.50, 0.70)
    bond_alloc = 1 - equity_alloc
    bench_eq_ret = rng.uniform(0.04, 0.10)
    bench_bond_ret = rng.uniform(0.01, 0.04)
    act_eq_ret = bench_eq_ret + rng.uniform(-0.02, 0.04)
    act_bond_ret = bench_bond_ret + rng.uniform(-0.01, 0.02)
    alloc_dev = rng.uniform(-0.10, 0.15)

    act_eq_w = equity_alloc + alloc_dev
    act_bond_w = 1 - act_eq_w

    # Allocation effect
    alloc_eff = equity_alloc * (act_eq_ret - bench_eq_ret) + bond_alloc * (act_bond_ret - bench_bond_ret)
    # Selection effect
    select_eff = act_eq_w * (act_eq_ret - bench_eq_ret) + act_bond_w * (act_bond_ret - bench_bond_ret)
    # Interaction effect
    inter_eff = (act_eq_w - equity_alloc) * (act_eq_ret - bench_eq_ret) + 0

    total_attribution = alloc_eff + select_eff + inter_eff

    stem = (f"Bench: {pct(equity_alloc)} equity at {pct(bench_eq_ret)}, "
            f"{pct(bond_alloc)} bonds at {pct(bench_bond_ret)}. "
            f"Portfolio: {pct(act_eq_w)} equity at {pct(act_eq_ret)}, "
            f"{pct(act_bond_w)} bonds at {pct(act_bond_ret)}. "
            f"Perform Brinson attribution.")
    answer = (f"Allocation effect (weight choice): "
              f"{pct(equity_alloc * (act_eq_ret - bench_eq_ret))} + "
              f"{pct(bond_alloc * (act_bond_ret - bench_bond_ret))} = "
              f"{pct(alloc_eff)}. "
              f"Selection effect (security choice): "
              f"{pct(act_eq_w * (act_eq_ret - bench_eq_ret))} + "
              f"{pct(act_bond_w * (act_bond_ret - bench_bond_ret))} = "
              f"{pct(select_eff)}. "
              f"Interaction effect: [{pct(inter_eff)}]. "
              f"Total active return = {pct(total_attribution)}. "
              f"Allocation effect is positive if overweighted in the outperforming sector. "
              f"Selection effect captures within-sector security selection. "
              f"Interaction captures the covariance of allocation and selection. "
              f"Interpretation: {pct(alloc_eff)} suggests {''if alloc_eff>0 else 'not '}"
              f"adding value through sector allocation decision."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Portfolio Management","subtopic":"Performance Attribution",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["alloc vs selection separation","style drift"]},
         "question":stem,"answer":answer}
    _out(r); return r

def pm_tax_management(rng, seq):
    """Tax-loss harvesting and after-tax return optimization."""
    cost_basis = rng.uniform(80, 100)
    market_val = rng.uniform(60, 90)
    unrealized_loss = (market_val - cost_basis) / cost_basis * 100
    tax_rate = rng.uniform(0.20, 0.40)
    same_year_gain = rng.uniform(5, 30)
    replacement_return = rng.uniform(0.05, 0.12)
    wash_sale_days = 30

    stem = (f"Holding: cost basis {fmt(cost_basis, 0)}, current value "
            f"{fmt(market_val, 0)}. Unrealized loss {pct(unrealized_loss/100)}. "
            f"Capital gains to offset this year: {fmt(same_year_gain, 0)}. "
            f"Tax rate on losses: {pct(tax_rate)}. What is the value of tax-loss "
            f"harvesting? What are the wash-sale concerns?")
    answer = (f"Loss = {fmt(cost_basis - market_val)} = {pct(unrealized_loss/100)}. "
              f"Tax saving from harvesting: {fmt(same_year_gain * tax_rate)} if "
              f"loss >= same_year_gain, otherwise {fmt(same_year_gain * tax_rate)}. "
              f"With {pct(tax_rate)} capital gains rate, each dollar of loss saves "
              f"{pct(tax_rate)} in taxes. "
              f"Wash-sale rule: cannot repurchase same security within 30 days "
              f"(before or after sale). This limits the ability to maintain position "
              f"exposure while harvesting losses. A workaround: buy a highly correlated "
              f"substitute (e.g., S&P 500 ETF for individual stock). "
              f"The after-tax return of the substitute strategy: "
              f"{pct(replacement_return)} * (1 - tax_rate * 0.5) vs original "
              f"return of {pct(unrealized_loss/100 + replacement_return)} including the "
              f"unrealized loss. Deferring the loss sale to next year might be "
              f"better if the tax rate is expected to increase."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Portfolio Management","subtopic":"Tax Management",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["wash-sale rule","tax rate deferral"]},
         "question":stem,"answer":answer}
    _out(r); return r

def pm_asset_allocation(rng, seq):
    """Strategic asset allocation and liability-driven investment."""
    equity_w = rng.uniform(0.40, 0.70)
    bond_w = 1 - equity_w
    e_eq_return = rng.uniform(0.06, 0.12)
    e_bond_return = rng.uniform(0.02, 0.05)
    vol_eq = rng.uniform(0.15, 0.25)
    vol_bond = rng.uniform(0.04, 0.10)
    corr = rng.uniform(0.0, 0.25)
    liab_return = rng.uniform(0.03, 0.06)
    port_return = equity_w * e_eq_return + bond_w * e_bond_return
    port_vol_sq = (equity_w * vol_eq)**2 + (bond_w * vol_bond)**2 + 2 * equity_w * bond_w * vol_eq * vol_bond * corr
    port_vol = port_vol_sq**0.5
    sharpe_num = port_return - liab_return
    sharpe = sharpe_num / port_vol if port_vol > 0 else 0
    surplus = port_return - liab_return

    stem = (f"LDI portfolio: {pct(equity_w)} equity at {pct(e_eq_return)} vol {pct(vol_eq)}, "
            f"{pct(bond_w)} bonds at {pct(e_bond_return)} vol {pct(vol_bond)}, "
            f"corr {corr:.2f}. Liability growth target {pct(liab_return)}. "
            f"What portfolio return, vol, and surplus? Optimal equity weight?")
    answer = (f"Expected return: {pct(port_return)}. "
              f"Portfolio vol: {pct(port_vol)}. "
              f"Sharpe ratio vs liability: {sharpe:.3f}. "
              f"Surplus return: {pct(surplus)}. "
              f"Volatility reduction from diversification: "
              f"{pct(equity_w * vol_eq + bond_w * vol_bond)} (naive) vs "
              f"{pct(port_vol)} (actual) = {(equity_w*vol_eq+bond_w*vol_bond-port_vol)*100:.1f}pp benefit. "
              f"The diversification benefit from low correlation {corr:.2f} between "
              f"equity and bonds reduces portfolio risk by {(equity_w*vol_eq+bond_w*vol_bond-port_vol)/port_vol*100:.1f}% "
              f"relative to a simple weighted average. The optimal equity weight would "
              f"increase if liabilities are long-duration bond-like (making bonds a poor "
              f"hedge). The surplus approach (liability-relative optimization) is "
              f"preferred over absolute return optimization for pension funds."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Portfolio Management","subtopic":"Asset Allocation",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["liability matching","diversification illusion"]},
         "question":stem,"answer":answer}
    _out(r); return r

def pm_behavioral_biases(rng, seq):
    """Behavioral biases in portfolio decision-making."""
    loss_aversion = rng.uniform(1.5, 2.5)
    prospect_loss = rng.uniform(0.05, 0.15)
    prospect_gain = rng.uniform(0.05, 0.10)
    overconfidence = rng.uniform(0.3, 0.8)
    disposition_rate = rng.uniform(0.55, 0.80)
    turnover = rng.uniform(0.20, 1.0)
    # Expected utility: loss is weighted more heavily
    eu_neg = -(loss_aversion * prospect_loss)
    eu_pos = prospect_gain
    ne_agree = eu_neg + eu_pos

    stem = (f"Client: loss aversion coefficient {loss_aversion:.1f}, reference point gain "
            f"of {pct(prospect_gain)}, potential loss of {pct(prospect_loss)}. "
            f"Disposition effect: sells winners at {disposition_rate:.0%} rate. "
            f"Overconfidence level {overconfidence:.2f}. How do these biases manifest "
            f"in portfolio behavior and quantification of their cost?")
    answer = (f"Expected utility of risky asset: "
              f"-{loss_aversion:.1f} * {pct(prospect_loss)} + {pct(prospect_gain)} = "
              f"{pct(eu_neg)} + {eu_pos:.3f} = {ne_agree:.3f}. "
              f"An expected utility investor requires a risky premium of "
              f"{pct(loss_aversion * prospect_loss - prospect_gain)} to break even, "
              f"making many opportunities appear unattractive when {ne_agree:.3f} < 0. "
              f"Disposition effect: selling winners at {disposition_rate:.0%} means "
              f"losing positions are held too long, realizing losses too late. "
              f"Turnover cost: {pct(turnover)} annual turnover at {pct(0.005)} per-trade cost = "
              f"{pct(turnover * 0.005)} in drag. Overconfidence leads to "
              f"{overconfidence:.0%} more active trading than optimal. The combined "
              f"cost of behavioral biases: tax inefficiency from disposition effect "
              f"(realizing losses at suboptimal timing), turnover drag, and "
              f"suboptimal entry/exit timing from overconfidence."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Portfolio Management","subtopic":"Behavioral Biases",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["loss aversion quantification","disposition effect measurement"]},
         "question":stem,"answer":answer}
    _out(r); return r

def pm_risk_budgeting(rng, seq):
    """Risk budgeting: decomposing contribution of each asset to portfolio risk."""
    weights_ = {
        'US_Eq': rng.uniform(0.35, 0.50),
        'Intl_Eq': rng.uniform(0.15, 0.25),
        'US_Bond': rng.uniform(0.20, 0.30),
        'Intl_Bond': rng.uniform(0.08, 0.15),
        'Real_Estate': rng.uniform(0.05, 0.10),
    }
    vols = {
        'US_Eq': rng.uniform(0.15, 0.22),
        'Intl_Eq': rng.uniform(0.15, 0.22),
        'US_Bond': rng.uniform(0.04, 0.08),
        'Intl_Bond': rng.uniform(0.06, 0.12),
        'Real_Estate': rng.uniform(0.12, 0.18),
    }
    base_corr = rng.uniform(0.20, 0.40)
    corr_matrix = {}
    for i, k1 in enumerate(weights_.keys()):
        for j, k2 in enumerate(weights_.keys()):
            if i < j:
                corr_matrix[k1, k2] = corr_matrix[k2, k1] = base_corr + rng.uniform(-0.10, 0.10)
            elif i == j:
                corr_matrix[k1, k2] = 1.0

    port_var = sum(weights_[k1] * vols[k1] * corr_matrix[k1,k2] * vols[k2] * weights_[k2] 
                   for k1 in weights_ for k2 in weights_)
    port_vol_total = port_var**0.5

    # Risk contribution
    risk_contrib = {}
    for k in weights_:
        marginal = sum(corr_matrix[k, k2] * vols[k] * vols[k2] * weights_[k2] for k2 in weights_)
        rc = weights_[k] * marginal
        risk_contrib[k] = rc
    risk_contrib_pct = {k: v / port_vol_total * 100 for k, v in risk_contrib.items()}

    stem = (f"Portfolio: {dict(((k, f'{pct(w)}') for k, w in weights_.items()))}. "
            f"Volatilities: {dict(((k, f'{pct(v)}') for k, v in vols.items()))}. "
            f"Base correlation {base_corr:.2f}, corr matrix varies +/-10%. "
            f"Compute each asset risk contribution and how to equalize it.")
    answer = (f"Portfolio vol: {pct(port_vol_total)}. "
              f"Marginal contribution: {dict(((k, f'+{pct(m)}') for k, m in risk_contrib.items()))}. "
              f"Risk contribution = weight * marginal = "
              f"{dict(((k, f'+{rc:.1f}%') for k, rc in risk_contrib_pct.items()))}. "
              f"US_Eq contributes {risk_contrib_pct['US_Eq']:.1f}% of portfolio risk "
              f"but has weight {pct(weights_['US_Eq'])}. "
              f"Risk budgeting: if we want equal risk contribution, target "
              f"{100/len(weights_):.1f}% from each asset. This often means increasing "
              f"weight in low-correlation, low-vol assets. The bond allocation, despite "
              f"small weights {pct(weights_['US_Bond']) + pct(weights_['Intl_Bond'])}, "
              f"provides {risk_contrib_pct['US_Bond'] + risk_contrib_pct['Intl_Bond']:.1f}% "
              f"of portfolio risk - a disproportionate safety benefit. "
              f"Rebalancing from risk budget rather than value budget may further "
              f"reduce concentration risk."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Portfolio Management","subtopic":"Risk Budgeting",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["marginal vs contribution","concentration illusion"]},
         "question":stem,"answer":answer}
    _out(r); return r



# ---- Risk Management ----

def rm_var_computation(rng, seq):
    """Value at Risk (VaR) computation and comparison of methods."""
    daily_vol = rng.uniform(0.008, 0.020)
    position = rng.uniform(10e6, 50e6)
    horizon_days = rng.randint(10, 30)
    expected_return = rng.uniform(0.0001, 0.0005)
    z_95 = 1.645
    z_99 = 2.326
    param_var_95 = position * z_95 * daily_vol * horizon_days**0.5
    param_var_99 = position * z_99 * daily_vol * horizon_days**0.5
    param_var_95_ew = param_var_95 * (1 + expected_return * horizon_days / position)
    # Simulate a few scenarios for comparison
    import numpy as np
    s = np.random.RandomState(42 + seq)
    returns = s.normal(expected_return, daily_vol, 1000)
    losses = -position * returns * horizon_days**0.5
    var_par = sorted(losses)[int(0.05 * len(losses))]
    cvar_par = sorted(losses)[int(0.025 * len(losses))]  # CVaR approx

    stem = (f"Position: {fmt(position/1e6, 1)}M, daily vol {pct(daily_vol)}, "
            f"horizon {horizon_days} days, daily drift {pct(expected_return)}. "
            f"Compute 95% and 99% parametric VaR. Compare to simulated VaR and CVaR.")
    answer = (f"Parametric 95% VaR (horizon {horizon_days}D): "
              f"{fmt(position * z_95 * daily_vol * horizon_days**0.5, 0)}. "
              f"Parametric 99% VaR: "
              f"{fmt(position * z_99 * daily_vol * horizon_days**0.5, 0)}. "
              f"Simulated VaR (5th percentile of 1000 runs: "
              f"{fmt(var_par, 0)}. "
              f"Expected shortfall (CVaR at 95%): "
              f"{fmt(cvar_par, 0)} (average loss beyond VaR). "
              f"VaR limitations: does not capture tail risk beyond the cutoff, "
              f"assumes normality (underestimates fat-tail losses), and is not "
              f"sub-additive (VaR(A)+VaR(B) may be < VaR(A+B)). CVaR, by contrast, "
              f"measures expected loss GIVEN it exceeds VaR - hence "
              f"{fmt(cvar_par, 0)} vs {fmt(var_par, 0)} (the tail adds "
              f"{pct((cvar_par-var_par)/var_par)} more). This gap widens with "
              f"fatter tails."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Risk Management","subtopic":"VaR",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["VaR tail gap","normality assumption"]},
         "question":stem,"answer":answer}
    _out(r); return r

def rm_expected_shortfall(rng, seq):
    """Expected Shortfall (CVaR) vs VaR: tail risk quantification."""
    mu = rng.uniform(0, 0.0002)
    sigma = rng.uniform(0.008, 0.025)
    portfolio_val = rng.uniform(20e6, 100e6)
    n_sims = 50000
    import numpy as np
    s = np.random.RandomState(99)
    daily_returns = s.normal(mu, sigma, n_sims)
    daily_losses = -portfolio_val * daily_returns
    losses_2d = -portfolio_val * sum(s.normal(mu, sigma, n_sims) for _ in range(min(2, n_sims//10))) / min(2, n_sims//10)

    # 95% and 99% VaR from simulation
    var_95 = sorted(daily_losses)[int(0.05 * n_sims)]
    var_99 = sorted(daily_losses)[int(0.01 * n_sims)]
    # CVaR: mean of losses beyond VaR
    n_5p = int(0.05 * n_sims)
    n_1p = int(0.01 * n_sims)
    sorted_2d = sorted(losses_2d)
    sorted_dl = sorted(daily_losses)
    cvar_95_vals = sorted_2d[:n_5p]  # worst 5% of 2-day losses
    cvar_95_val = sum(cvar_95_vals) / len(cvar_95_vals) if len(cvar_95_vals) > 0 else 0
    cvar_99_vals = sorted_dl[:n_1p]
    cvar_99_val = sum(cvar_99_vals) / len(cvar_99_vals) if len(cvar_99_vals) > 0 else 0

    stem = (f"Portfolio value {fmt(portfolio_val/1e6, 0)}M. Daily return "
            f"mean {pct(mu)}, std {pct(sigma)}. "
            f"Compute 95% and 99% VaR and CVaR from simulation. "
            f"What does the VaR-to-CVaR gap tell us about tail risk?")
    answer = (f"Simulated 95% VaR = {fmt(var_95, 0)}. 99% VaR = "
              f"{fmt(var_99, 0)}. "
              f"95% CVaR (expected loss beyond VaR) = {fmt(cvar_95_val, 0)}. "
              f"99% CVaR = {fmt(cvar_99_val, 0)}. "
              f"The VaR-to-CVaR gap at 95%: "
              f"{pct((cvar_95_val - var_95)/var_95 * 100)} (in percentage terms). "
              f"This gap is the premium of tail risk over the VaR threshold. "
              f"At 99%, gap = {cvar_99_val/var_99 - 1:.1f}x VaR. "
              f"CVaR is sub-additive: splitting a portfolio reduces total CVaR "
              f"(unlike VaR which may not). CVaR is also coherent: it satisfies "
              f"monotonicity, translation invariance, positive homogeneity, and "
              f"sub-additivity. Regulators increasingly prefer CVaR over VaR "
              f"(e.g., Basel IV uses Expected Shortfall instead of VaR)."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Risk Management","subtopic":"Expected Shortfall",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["CVaR sub-additivity","regulatory preference"]},
         "question":stem,"answer":answer}
    _out(r); return r

def rm_stress_testing(rng, seq):
    """Stress test: scenario analysis and risk limit violations."""
    equity_exposure = rng.uniform(100e6, 500e6)
    credit_exposure = rng.uniform(50e6, 200e6)
    rate_exposure = rng.uniform(100e6, 300e6)
    # Scenario: equity -20%, credit spread widens 100bp, rates +200bp
    eq_shock = rng.uniform(-0.20, -0.35)
    credit_shock_dps = rng.uniform(80, 180)
    rate_shock = rng.uniform(0.15, 0.30)
    dur_risk = rng.uniform(5, 10)

    eq_loss = equity_exposure * eq_shock
    credit_loss = credit_exposure * credit_shock_dps / 10000  # approx price impact
    rate_loss = rate_exposure * dur_risk * rate_shock * 0.01  # dollar duration
    total_loss = eq_loss + credit_loss + rate_loss

    risk_limit = rng.uniform(50e6, 150e6)
    limit_breach = total_loss > risk_limit

    stem = (f"Portfolio: equity {fmt(equity_exposure/1e6,0)}M, credit "
            f"{fmt(credit_exposure/1e6,0)}M, interest rate risk "
            f"{fmt(rate_exposure/1e6,0)}M (duration {dur_risk}Y). "
            f"Stress: equity -{abs(eq_shock)*100:.1f}%, credit spread +{credit_shock_dps:.0f}bps, "
            f"rates +{rate_shock*100:.1f}bp. Total loss? Does it breach the "
            f"${fmt(risk_limit/1e6)}M risk limit?")
    answer = (f"Equity loss: {fmt(equity_exposure * eq_shock/1e6, 0)}M. "
              f"Credit spread widening impact: {fmt(credit_exposure * credit_shock_dps/1e7, 1)}M. "
              f"Interest rate shock: {fmt(rate_exposure * dur_risk * rate_shock * 0.001, 0)}M. "
              f"Total stressed loss: {fmt(total_loss/1e6, 0)}M. "
              f"Risk limit: {fmt(risk_limit/1e6, 0)}M. "
              f"Breached: {'' if limit_breach else 'No'}. "
              f"Correlation matters: if equity and credit losses are correlated at "
              f"{rng.uniform(0.5, 0.8):.2f}, the diversification benefit is reduced. "
              f"Risk limit should be set relative to: {pct(total_loss/risk_limit)} "
              f"of the limit. A backtesting framework would track frequency of "
              f"limit breaches over rolling periods and adjust limits accordingly. "
              f"Scenario shocks should be realistic but severe - a {abs(eq_shock)*100:.0f}% "
              f"equity drop is roughly the 2008 crisis magnitude."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Risk Management","subtopic":"Stress Testing",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["scenario realism","correlation breakdown under stress"]},
         "question":stem,"answer":answer}
    _out(r); return r

def rm_credit_risk(rng, seq):
    """Credit risk: PD, LGD, EAD, and expected loss."""
    portfolio_value = rng.uniform(200e6, 1000e6)
    pd = rng.uniform(0.005, 0.05)  # probability of default
    lgd = rng.uniform(0.30, 0.60)  # loss given default
    ead = rng.uniform(0.70, 1.00)  # exposure at default
    expected_loss = portfolio_value * pd * lgd * ead
    unexpected_loss_factor = rng.uniform(2.0, 5.0)  # capital buffer
    unexpected_loss = expected_loss * unexpected_loss_factor
    risk_weighted = portfolio_value * pd * lgd * rng.uniform(0.5, 2.0)

    stem = (f"Portfolio: {fmt(portfolio_value/1e6, 0)}M. PD = {pct(pd)}, "
            f"LGD = {pct(lgd)}, EAD factor = {pct(ead)}. "
            f"Compute expected and unexpected loss. How much capital is needed?")
    answer = (f"Expected loss = {fmt(portfolio_value/1e6)}M * "
              f"{pct(pd)} * {pct(lgd)} * {pct(ead)} = "
              f"{fmt(expected_loss/1e6, 2)}M. "
              f"Unexpected loss = {fmt(expected_loss/1e6, 2)}M * "
              f"{unexpected_loss_factor:.1f} = {fmt(unexpected_loss/1e6, 2)}M. "
              f"Expected loss is provisioned (charged against earnings). "
              f"Unexpected loss requires economic capital. "
              f"Risk-weighted credit exposure = "
              f"{fmt(risk_weighted/1e6, 0)}M * "
              f"{pd*lgd*100:.2f} = {fmt(risk_weighted * pd * lgd/1e6, 0)}M RWA. "
              f"At 8% regulatory capital: {fmt(risk_weighted * pd * lgd * 0.08/1e6, 2)}M required. "
              f"The LGD depends on collateral quality and recovery time. A 50% LGD "
              f"implies the lender recovers 50% through collateral sale. EAD "
              f"captures undrawn commitments - for lines of credit, EAD < 1.0 "
              f"because not all committed amount is drawn at the time of default. "
              f"Expected loss is priced into the loan spread; unexpected loss is "
              f"funded by capital."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Risk Management","subtopic":"Credit Risk",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["PD/LGD correlation","provision vs capital"]},
         "question":stem,"answer":answer}
    _out(r); return r

def rm_model_risk(rng, seq):
    """Model risk: validation, backtesting, and uncertainty quantification."""
    model_var = rng.uniform(2.5, 5.5)
    actual_loss_95 = rng.uniform(3.0, 8.0)
    actual_loss_99 = rng.uniform(4.0, 10.0)
    backtest_losses = [rng.uniform(0, 15) for _ in range(250)]
    exceptions_95 = sum(1 for l in backtest_losses[125:] if l > model_var * 1.645)
    exceptions_99 = sum(1 for l in backtest_losses[125:] if l > model_var * 2.326)
    expected_95 = 250 * 0.05
    expected_99 = 250 * 0.01
    kupper_stat_95 = min(
        (actual_loss_95 - model_var) / (model_var * (1/1.645 - 0.05/1.645**2)**0.5),
        0)

    stem = (f"VaR model: daily 95% VaR = {pct(model_var)}, 99% VaR = "
            f"{pct(model_var * 2.326/1.645)}. "
            f"Backtesting over {250} days: actual 95% loss observed "
            f"{pct(actual_loss_95)}, actual 99% observed "
            f"{pct(actual_loss_99)}. Exceptions at 95%: "
            f"{exceptions_95}/{250} (expected {expected_95:.1f}). "
            f"Is the model under- or over-estimating risk?")
    answer = (f"Expected exceptions at 95%: {expected_95:.0f} in {250} days. "
              f"Actual exceptions: {exceptions_95}. "
              f"Kupfer test: z = "
              f"(actual - expected) / sqrt(expected * (1-alpha)) = "
              f"(exceptions_95 - {expected_95}) / "
              f"sqrt({expected_95} * 0.95). "
              f"Model VaR = {pct(model_var)} vs actual = {pct(actual_loss_95)} "
              f"at 95%, ratio = {actual_loss_95/model_var:.2f}. "
              f"At 99%: {pct(actual_loss_99)} vs model, ratio = "
              f"{actual_loss_99/(model_var*2.326/1.645):.2f}. "
              f"Model underestimation: actual VaR should be "
              f"{pct(actual_loss_95)} vs model {pct(model_var)} = "
              f"{pct((actual_loss_95-model_var)/model_var)} underestimate. "
              f"Model risk arises from: wrong distribution assumption, parameter "
              f"uncertainty, historical sample bias, and structural breaks. The "
              f"gap between model and actual VaR is typically larger than backtest "
              f"exceptions suggest because VaR measures a threshold, while the "
              f"average of violations (CVaR) shows how badly the model underestimates "
              f"tail losses."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Risk Management","subtopic":"Model Risk",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["backtest frequency","distributional misspecification"]},
         "question":stem,"answer":answer}
    _out(r); return r

def rm_liquidity_risk(rng, seq):
    """Liquidity risk: bid-ask spreads, market impact, and liquidity-adjusted VaR."""
    position = rng.uniform(10e6, 100e6)
    spread_pct = rng.uniform(0.001, 0.020)
    vol_daily = rng.uniform(0.008, 0.025)
    avg_volume = rng.uniform(5e6, 50e6)
    liquidity_days = rng.uniform(2, 7)
    # Liquidity-adjusted VaR
    hold_to_sell_days = liquidity_days
    spread_cost = position * spread_pct
    var_std = position * vol_daily * hold_to_sell_days**0.5 * 1.645
    lavar_95 = var_std + 0.5 * spread_cost * hold_to_sell_days

    stem = (f"Position: {fmt(position/1e6, 1)}M daily vol {vol_daily:.4f}, "
            f"bid-ask spread {pct(spread_pct)}, average daily volume "
            f"{fmt(avg_volume/1e6, 0)}M. "
            f"How many days to liquidate {fmt(position/avg_volume/1e6, 1)}x daily volume? "
            f"Compute liquidity-adjusted 95% VaR.")
    answer = (f"Hold-to-sell period: {fmt(position/avg_volume, 2)}x daily volume "
              f"= ~{hold_to_sell_days:.1f} days. "
              f"Standard VaR (95%, {hold_to_sell_days:.1f}D): "
              f"{fmt(var_std/1e6, 0)}M. "
              f"Spread cost: {fmt(spread_cost/1e6, 1)}M. Cumulative spread cost "
              f"(half-spread each day): "
              f"{fmt(spread_cost/2 * hold_to_sell_days/1, 1)}M. "
              f"LAVaR = VaR + spread cost = "
              f"{fmt(lavar_95/1e6, 0)}M. "
              f"Liquidity risk increases effective risk beyond pure market risk: "
              f"a wide spread means the bid-ask spread eats directly into P/L "
              f"in the liquidation window. In stressed conditions, spreads widen "
              f"and depth dries up simultaneously - the 'liquidity double kill'. "
              f"For a {position/avg_volume:.0f}x daily volume position, "
              f"even a 2x spread widening cost is manageable, but during a market "
              f"crisis with volume dropping 50%, hold-to-sell becomes "
              f"{liquidity_days * 2:.1f} days, nearly doubling the spread cost component."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Risk Management","subtopic":"Liquidity Risk",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["liquidity double kill","volume assumptions"]},
         "question":stem,"answer":answer}
    _out(r); return r

def rm_operational_risk(rng, seq):
    """Operational risk: internal losses and economic capital allocation."""
    operational_capital = rng.uniform(50e6, 200e6)
    operational_var = rng.uniform(0.05, 0.15)
    var_99 = operational_capital * operational_var * 2.326
    # Losses by category
    internal_losses = {
        'execution_error': rng.uniform(0.5e6, 3e6),
        'fraud': rng.uniform(0.2e6, 2e6),
        'system_outage': rng.uniform(1e6, 8e6),
        'legal': rng.uniform(0.5e6, 5e6),
        'data_loss': rng.uniform(0.3e6, 1e6),
    }
    total_historical = sum(internal_losses.values())
    tail_multiplier = rng.uniform(4, 8)
    projected_operational_loss = total_historical * tail_multiplier

    stem = (f"Operational losses last year: execution error "
            f"{fmt(internal_losses['execution_error']/1e6, 1)}M, "
            f"fraud {fmt(internal_losses['fraud']/1e6, 1)}M, "
            f"system outage {fmt(internal_losses['system_outage']/1e6, 1)}M, "
            f"legal {fmt(internal_losses['legal']/1e6, 1)}M, data loss "
            f"{fmt(internal_losses['data_loss']/1e6, 1)}M. "
            f"Historic total: {fmt(total_historical/1e6, 0)}M. "
            f"What operational economic capital is warranted?")
    answer = (f"Total historical losses: {fmt(total_historical/1e6, 0)}M. "
              f"Tail risk multiplier (4-8x historical): "
              f"{tail_multiplier:.1f}x total = {fmt(total_historical * tail_multiplier / 1e6, 0)}M. "
              f"99% operational VaR: {fmt(var_99/1e6, 0)}M "
              f"(using normal approx: capital * {pct(operational_var)} * 2.326, but "
              f"operational losses are typically skewed so normal approx may understate). "
              f"The largest single category by absolute loss: "
              f"{max(internal_losses, key=internal_losses.get).replace('_', ' ')} at "
              f"{fmt(internal_losses[max(internal_losses, key=internal_losses.get)]/1e6, 1)}M. "
              f"Operational risk capital: the Basel approach uses Gross Domestic Product "
              f"for banking portfolio value and loss multipliers. The standardised approach "
              f"allocates capital by business line. Internal loss approach (ILAs) like the "
              f"Basel AMA uses: max(Cvar_99, Expected Loss * Loss Multiplier). The loss "
              f"multiplier captures the fat-tail nature of operational losses - they are "
              f"more unpredictable than credit or market risk."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Risk Management","subtopic":"Operational Risk",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["fat-tail operational losses","Baer II AMA phase-out"]},
         "question":stem,"answer":answer}
    _out(r); return r



# ---- Financial Statement Analysis ----

def fsa_revenue_recognition(rng, seq):
    """Revenue recognition: percentage of completion vs completed contract."""
    contract_value = rng.uniform(50e6, 500e6)
    total_cost = contract_value * rng.uniform(0.60, 0.85)
    cost_to_date = rng.uniform(0.25, 0.65) * total_cost
    pct_complete = cost_to_date / total_cost
    revenue_to_date = pct_complete * contract_value
    gross_profit_to_date = revenue_to_date - cost_to_date
    margin = gross_profit_to_date / revenue_to_date if revenue_to_date > 0 else 0
    # Prior period revenue already recognized
    prior_pct = rng.uniform(0.05, 0.20)
    prior_revenue = prior_pct * contract_value
    current_period_revenue = revenue_to_date - prior_revenue
    prior_cost = prior_pct * total_cost
    current_cost = cost_to_date - prior_cost
    current_profit = current_period_revenue - current_cost

    stem = (f"Long-term contract: total value {fmt(contract_value/1e6, 0)}M, "
            f"total cost {fmt(total_cost/1e6, 0)}M. Costs incurred to date: "
            f"{fmt(cost_to_date/1e6, 0)}M. Prior period: {pct(prior_pct)} complete. "
            f"Using percentage-of-completion, what revenue and gross profit should "
            f"be recognized in the current period? How does this differ from CIP?")
    answer = (f"Percentage complete: {fmt(cost_to_date/total_cost, 2)} = "
              f"{pct(pct_complete)}. "
              f"Revenue to date: {fmt(revenue_to_date/1e6, 0)}M. "
              f"Gross profit to date: {fmt(gross_profit_to_date/1e6, 1)}M. "
              f"Prior revenue: {fmt(prior_revenue/1e6, 0)}M. "
              f"Current period revenue: {fmt(current_period_revenue/1e6, 0)}M. "
              f"Current period cost: {fmt(current_cost/1e6, 0)}M. "
              f"Current period gross profit: {fmt(current_profit/1e6, 1)}M."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Revenue Rec",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["cumulative revenue","estimation uncertainty"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fsa_lease_accounting(rng, seq):
    """Lease accounting: operating vs finance lease under ASC 842."""
    lease_term = rng.randint(3, 15)
    annual_payment = rng.uniform(5e6, 30e6)
    discount_rate = rng.uniform(0.04, 0.08)
    lessee_incremental_rate = rng.uniform(0.045, 0.075)
    # Finance lease: PV of lease payments
    pv_payments = sum(annual_payment / (1+lessee_incremental_rate)**i for i in range(1, lease_term+1))
    # Operating lease: straight-line expense
    total_pmt = annual_payment * lease_term
    annual_lease_exp = total_pmt / lease_term
    # Interest expense: front-loaded
    interest_1 = pv_payments * lessee_incremental_rate
    principal_1 = annual_payment - interest_1
    book_val_1 = pv_payments - principal_1

    stem = (f"Lease: {lease_term}Y term, annual payment {fmt(annual_payment/1e6)}M, "
            f"incremental borrowing rate {pct(lessee_incremental_rate)}. "
            f"Demonstrate balance sheet and P/L recognition for finance lease. "
            f"How does it differ from operating lease accounting?")
    answer = (f"Lease liability (PV): {fmt(pv_payments/1e6)}M. "
              f"ROU asset = lease liability = "
              f"{fmt(pv_payments/1e6)}M. "
              f"Year 1: interest = "
              f"{fmt(pv_payments * lessee_incremental_rate / 1e6, 1)}M. "
              f"Cash paid: {fmt(annual_payment / 1e6, 0)}M. "
              f"Principal reduction: "
              f"{fmt((annual_payment - pv_payments * lessee_incremental_rate) / 1e6, 1)}M. "
              f"Year 1 lease liability: "
              f"{fmt(book_val_1 / 1e6, 1)}M. "
              f"Finance lease: interest expense falls, ROU amortization is straight-line. "
              f"Total P/L = interest + amortization (front-loaded expense). "
              f"Operating lease: single straight-line lease expense each year. "
              f"Balance sheet impact: both create liability and ROU asset, but "
              f"the pattern of expense recognition differs. Finance lease: "
              f"higher expense in early years. Operating lease: constant expense. "
              f"Debt-to-equity increases: the lease liability is on the balance sheet."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Leases",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["finance vs operating classification","discount rate choice"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fsa_pension_benefits(rng, seq):
    """Pension accounting: PBO, plan assets, and funded status."""
    beg_pbo = rng.uniform(100e6, 500e6)
    beg_assets = beg_pbo * rng.uniform(0.7, 1.10)
    curr_service = rng.uniform(5e6, 25e6)
    interest_rate = rng.uniform(0.03, 0.07)
    plan_return = rng.uniform(-0.05, 0.15)
    act_gain_loss = rng.uniform(-15e6, 15e6)
    contributions = rng.uniform(5e6, 30e6)
    benefits_paid = rng.uniform(5e6, 20e6)
    end_pbo = beg_pbo + beg_pbo * interest_rate + curr_service - benefits_paid + act_gain_loss
    end_assets = beg_assets * (1 + plan_return) + contributions - benefits_paid
    funded_status = end_assets - end_pbo

    stem = (f"Pension: beg PBO = {fmt(beg_pbo/1e6, 0)}M, beg plan assets "
            f"{fmt(beg_assets/1e6, 0)}M. Curr year: service cost "
            f"{fmt(curr_service/1e6, 0)}M, interest rate {pct(interest_rate)}, "
            f"actual return {pct(plan_return)}, actuarial gain/loss "
            f"{fmt(act_gain_loss/1e6, 0)}M, contributions "
            f"{fmt(contributions/1e6, 0)}M, benefits paid "
            f"{fmt(benefits_paid/1e6, 0)}M. Compute PBO, plan assets, "
            f"and funded status at year end.")
    answer = (f"PBO end: {fmt(beg_pbo/1e6, 0)} + {fmt(interest_rate*100, 1)}% "
              f"* {fmt(beg_pbo/1e6, 0)} + {fmt(curr_service/1e6, 0)} - "
              f"{fmt(benefits_paid/1e6, 0)} + {fmt(act_gain_loss/1e6, 1)} = "
              f"{fmt(end_pbo/1e6, 0)}M. "
              f"Plan assets: {fmt(beg_assets/1e6, 0)} * (1 + {pct(plan_return)}) + "
              f"{fmt(contributions/1e6, 0)} - "
              f"{fmt(benefits_paid/1e6, 0)} = {fmt(end_assets/1e6, 0)}M. "
              f"Funded status: {fmt(end_assets/1e6, 0)} - {fmt(end_pbo/1e6, 0)} = "
              f"{fmt(funded_status/1e6, 0)}M. "
              f"Pension P/L: service cost is in P/L, interest cost is in P/L, "
              f"expected return on assets reduces pension expense, "
              f"actuarial gains/losses are typically recognized in OCI."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Pensions",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["PBO calculation","expected vs actual return"]},
         "question":stem,"answer":answer}
    _out(r); return r

def fsa_goodwill_impairment(rng, seq):
    """Goodwill impairment testing: fair value of reporting units."""
    goodwill_bv = rng.uniform(50e6, 500e6)
    net_assets_bv = rng.uniform(200e6, 800e6)
    fair_value_bv_ratio = rng.uniform(1.05, 1.50)  # FU / BV
    fair_value = (goodwill_bv + net_assets_bv) * fair_value_bv_ratio
    implied_goodwill = fair_value - net_assets_bv
    impairment_check = max(0, goodwill_bv - implied_goodwill)
    impairment_pct = impairment_check / goodwill_bv * 100 if goodwill_bv > 0 else 0

    stem = (f"Goodwill on balance sheet: {fmt(goodwill_bv/1e6, 0)}M. Net asset "
            f"book value (excluding GW): {fmt(net_assets_bv/1e6, 0)}M. "
            f"The reporting unit fair value is {pct(fair_value_bv_ratio)} of total BV. "
            f"Perform goodwill impairment test. What is the impairment charge?")
    answer = (f"Total fair value of reporting unit: "
              f"{fmt(fair_value/1e6, 0)}M. "
              f"Fair value of net identifiable assets: "
              f"{fmt(net_assets_bv/1e6, 0)}M. "
              f"Implied goodwill = {fmt(fair_value/1e6, 0)} - "
              f"{fmt(net_assets_bv/1e6, 0)} = {fmt(implied_goodwill/1e6, 0)}M. "
              f"Carrying amount of goodwill: {fmt(goodwill_bv/1e6, 0)}M. "
              f"Impairment = max(0, {fmt(goodwill_bv/1e6, 0)} - "
              f"{fmt(implied_goodwill/1e6, 0)}) = "
              f"{fmt(round_impairment/1e6, 0)}M ({pct(round_impairment_pct)}). "
              f"Qualitative assessment first: if no trigger for impairment exists, "
              f"the quantitative test may be skipped. The 'step zero' qualitative "
              f"test looks at: macro conditions, industry factors, cost factors, "
              f"entity-specific factors. The quantitative test compares the fair "
              f"value of the reporting unit to its carrying amount. If FV > CV, "
              f"no impairment. If FV < CV, the excess carrying amount is the "
              f"impairment loss."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Goodwill",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["FV vs CV comparison","step one vs step two"]},
         "question":stem,"answer":answer}
    _OUT = round(impairment_check), f
    r2 = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Goodwill",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["FV vs CV comparison","step one vs step two"]},
         "question":stem,
         "answer": f"""{f'Total fair value of reporting unit: {round(fair_value/1e6, 0)}M.'}
{f'Fair value of net identifiable assets: {round(net_assets_bv/1e6, 0)}M.'}
{f'Implied goodwill = {round(fair_value/1e6, 0)} - {round(net_assets_bv/1e6, 0)} = {round(implied_goodwill/1e6, 0)}M.'}
{f'Carrying amount: {round(goodwill_bv/1e6, 0)}M. Impairment = {round(max(0,round(goodwill_bv - implied_goodwill, 0))/1e6, 0)}M.'}
{f'Qualitative trigger assessment first, then quantitative comparison of FV vs CV.'}"""
    }
    _out(r2);


def fsa_goodwill_impairment(rng, seq):
    """Goodwill impairment: fair value test for reporting unit."""
    goodwill_bv = rng.uniform(50, 500)
    net_ident_bv = rng.uniform(200, 800)
    fv_bv_mult = rng.uniform(1.05, 1.50)
    total_bv = goodwill_bv + net_ident_bv
    reporting_fu = total_bv * fv_bv_mult
    fu_implied_gw = reporting_fu - net_ident_bv
    impairment = max(0, goodwill_bv - fu_implied_gw)
    impairment_pct = round(impairment / goodwill_bv * 100, 1) if goodwill_bv > 0 else 0
    answer = (f"Total fair value of reporting unit: "
              f"{round(reporting_fu, 0)}M. "
              f"Fair value of net identifiable assets: "
              f"{round(net_ident_bv, 0)}M. "
              f"Implied goodwill at fair value: "
              f"{round(fu_implied_gw, 0)}M. "
              f"Carrying amount of goodwill: "
              f"{round(goodwill_bv, 0)}M. "
              f"Impairment = max(0, {round(goodwill_bv, 0)} - "
              f"{round(fu_implied_gw, 0)}) = "
              f"{round(impairment, 0)}M ({pct(round(impairment_pct/100, 2))}). "
              f"If reporting unit FV > carrying amount, no impairment exists. "
              f"The impairment loss cannot exceed the total goodwill balance. "
              f"A qualitative assessment is performed first (step zero) to check "
              f"for triggers: macro conditions, sector trends, cost increases, "
              f"entity-specific events (chief executive departure, loss of key customer). "
              f"Only if qualitative suggests possible impairment is the quantitative "
              f"step (FV vs CV) performed. If FV < CV, the impairment = excess of CV of "
              f"goodwill over its implied FV."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Goodwill",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["FV vs CV comparison","step one vs two"]},
         "question":(f"Goodwill: {round(goodwill_bv, 0)}M. Net assets "
                    f"(excl GW): {round(net_ident_bv, 0)}M. "
                    f"Reporting unit FV is {fv_bv_mult:.1f}x total BV. "
                    f"Impairment test?"),
         "answer": answer}
    _out(r); return r


def fsa_inventory_methods(rng, seq):
    """Inventory valuation: FIFO vs LIFO vs weighted average."""
    quantities = [rng.randint(50, 200) for _ in range(4)]
    unit_costs = sorted(rng.uniform(10, 40) for _ in range(4))  # rising costs
    units_sold = rng.randint(int(sum(quantities)*0.5), int(sum(quantities)*0.9))
    sales_price = rng.uniform(50, 100)
    
    # FIFO: first in, first out
    fifo_units_left = []
    fifo_units_remaining = list(quantities)
    units_sold_count = units_sold
    fifo_cogs = 0
    fifo_revenue = units_sold * sales_price
    for i, q in enumerate(quantities):
        take = min(q, units_sold_count)
        fifo_cogs += take * unit_costs[i]
        units_sold_count -= take
        if units_sold_count <= 0:
            fifo_units_left.append(q - take)
            for j in range(i+1, len(quantities)):
                fifo_units_left.append(quantities[j])
            break
        fifo_units_left.append(0)
    
    # LIFO: last in, first out
    lifo_units_remaining = list(quantities)
    lifo_cogs = 0
    units_sold_lifo = units_sold
    for i in range(len(quantities)-1, -1, -1):
        take = min(quantities[i], units_sold_lifo)
        lifo_cogs += take * unit_costs[i]
        units_sold_lifo -= take
        if units_sold_lifo <= 0:
            break
    
    fifo_ending = sum(q * c for q, c in zip(fifo_units_left, unit_costs[:len(fifo_units_left)]) if q > 0)
    
    # LIFO ending: the FIRST units remain (last in, first out means last units are oldest)
    lifo_remaining = [0]*len(quantities)
    units_left = sum(quantities) - units_sold
    for i in range(len(quantities)):
        lifo_remaining[i] = min(quantities[i], units_left)
        units_left -= quantities[i]
        if units_left <= 0:
            break

    lifo_ending = sum(q * c for q, c in zip(lifo_remaining, unit_costs))
    
    # FIFO ending: the LAST units remain
    fifo_remaining = [0]*len(quantities)
    units_left = sum(quantities) - units_sold
    for i in range(len(quantities)-1, -1, -1):
        fifo_remaining[i] = min(quantities[i], units_left)
        units_left -= quantities[i]
        if units_left <= 0:
            break
    fifo_ending = sum(q * c for q, c in zip(fifo_remaining, unit_costs))
    
    total_cogs_fifo = sum(q*c for q,c in zip(quantities, unit_costs)) - fifo_ending
    total_cogs_lifo = sum(q*c for q,c in zip(quantities, unit_costs)) - lifo_ending

    stem = (f"Inventory: quantities purchased {[round(q) for q in quantities]}, "
            f"unit costs rising: {[round(c,1) for c in unit_costs]}. Sales: "
            f"{units_sold} units at {round(sales_price,1)}. Compute COGS (FIFO, LIFO) "
            f"and ending inventory for each. In what economic environment does "
            f"LIFO reduce taxes vs FIFO?")
    answer = (f"Total units purchased: {sum(quantities)}. Units sold: "
              f"{units_sold}. Units in ending: {sum(quantities)-units_sold}. "
              f"FIFO COGS: {round(fifo_cogs)} (oldest, cheapest costs). "
              f"FIFO ending inventory: {round(fifo_ending)} (newest, highest costs). "
              f"LIFO COGS: {round(lifo_cogs)} (newest, highest costs). "
              f"LIFO ending: {round(lifo_ending)} (oldest costs). "
              f"In an inflationary environment (rising unit costs): "
              f"LIFO COGS > FIFO COGS, so LIFO shows lower taxable income. "
              f"Tax savings = {(lifo_cogs - fifo_cogs) * 0.25:.0f} (at 25% rate). "
              f"Under IFRS, LIFO is prohibited. FIFO is permitted under both. "
              f"LIFO provides a better matching (current costs vs current revenue) "
              f"but distorts inventory valuation on the balance sheet. The LIFO "
              f"reserve (FIFO - LIFO inventory) represents deferred taxes that "
              f"would be due if the firm switched to FIFO."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Inventory",
                 "difficulty":"CFA L1","question_type":"Analysis",
                 "pitfalls":["LIFO reserve","IFRS prohibition"]},
         "question":stem,"answer":answer}
    _out(r); return r


def fsa_fraud_detection(rng, seq):
    """Fraud detection: Benford law analysis and ratio anomalies."""
    n_accounts = rng.randint(30, 100)
    # Generate some account balances - some with Benford-compliant leading digits
    # and inject anomalies
    actual_first_digits = []
    for _ in range(n_accounts):
        amount = rng.uniform(1000, 999999)
        actual_first_digits.append(int(str(int(amount))[0]))
    
    # Expected Benford distribution
    benford_expected = {1: -math.log10(1+1/1), 2: -math.log10(1+1/2),
                        3: -math.log10(1+1/3), 4: -math.log10(1+1/4),
                        5: -math.log10(1+1/5), 6: -math.log10(1+1/6),
                        7: -math.log10(1+1/7), 8: -math.log10(1+1/8),
                        9: -math.log10(1+1/9)}
    
    from collections import Counter
    d_counts = Counter(actual_first_digits)
    benford_pct = {d: round(benford_expected[d]*100, 1) for d in benford_expected}
    actual_pct = {d: round(d_counts.get(d, 0) / n_accounts * 100, 1) for d in range(1, 10)}
    
    # Some anomalies: 1s appear too frequently (common fraud indicator)
    anomaly_prob = 0.60 if rng.randint(0, 1) else 0.40
    
    stem = (f"Benford analysis of {n_accounts} account balances. Expected leading "
            f"digits: {dict(((str(d), f'{benford_pct[d]}%') for d in range(1,10)))}. "
            f"Observed: {dict(((str(d), f'+{actual_pct[d]}%') for d in range(1,10)))}. "
            f"Anomaly probability: {anomaly_prob:.0%}. What fraud risk indicators "
            f"do you see and what is the recommended investigation?")
    answer = (f"Benford law predicts: 1s appear {benford_pct[1]}% of the time, "
              f"while 9s appear only {benford_pct[9]}%. "
              f"Observed: {[f'{str(d)}: {actual_pct[d]}%' for d in range(1,10)]}. "
              f"Divergence: 1s at {actual_pct[1]}% vs expected "
              f"{benford_pct[1]}% (+{actual_pct[1]-benford_pct[1]:.1f}pp). "
              f"3s at {actual_pct[3]}% vs expected "
              f"{benford_pct[3]}%" + (f" (+{actual_pct[3]-benford_pct[3]:.1f}pp - possible fabrication with round numbers)" if actual_pct[3] > benford_pct[3] + 10 else "") + ". "
              f"Key fraud indicators: (1) deviation from Benford's law suggests "
              f"fabricated numbers, (2) excess of round numbers in the distribution, "
              f"(3) clustering around thresholds (e.g., just-above-approval-limit). "
              f"Investigation: trace a sample of flagged transactions to original "
              f"source documents, interview the preparer, review the internal "
              f"control environment. The deviation from {anomaly_prob:.0%} indicates "
              f"a moderate-to-high fraud risk probability."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Fraud Detection",
                 "difficulty":"CFA L3","question_type":"Analysis",
                 "pitfalls":["Benford law applicability","manual entries"]},
         "question":stem,"answer":answer}
    _out(r); return r


def fsa_ratios_analysis(rng, seq):
    """Financial ratios: cross-company comparison and trend analysis."""
    # Company A: growing, high margin
    rev_a = rng.uniform(500e6, 2000e6)
    rev_b = rev_a * rng.uniform(0.4, 0.9)
    margin_a = rng.uniform(0.20, 0.40)
    margin_b = rng.uniform(0.10, 0.25)
    roe_a = rng.uniform(0.20, 0.40)
    roe_b = rng.uniform(0.10, 0.25)
    lev_a = rng.uniform(0.30, 0.60)  # debt/equity
    lev_b = rng.uniform(0.60, 1.20)
    
    roa_a = roe_a / (1 + lev_a)
    roa_b = roe_b / (1 + lev_b)
    margin_diff = margin_a - margin_b
    levr_a = 1 + lev_a
    levr_b = 1 + lev_b
    dupont_a = margin_a * roa_a / margin_a * levr_a  # turnover * margin * leverage
    dupont_b = margin_b * roa_b / margin_b * levr_b

    stem = (f"Company A: revenue {fmt(rev_a/1e6, 0)}M, margin {pct(margin_a)}, "
            f"ROE {pct(roe_a)}, D/E {lev_a:.1f}x. "
            f"Company B: revenue {fmt(rev_b/1e6, 0)}M, margin {pct(margin_b)}, "
            f"ROE {pct(roe_b)}, D/E {lev_b:.1f}x. "
            f"Decompose ROE using DuPont and determine which company creates more value.")
    answer = (f"Dupont A: ROE = Margin * Turnover * Leverage. "
              f"ROA = {pct(roa_a)} = {pct(roe_a)} / {levr_a:.2f}. "
              f"Turnover = {pct(margin_a)} / {pct(roa_a)} / {lev_a:.2f} "
              f"(implied: ~{roa_a/margin_a*lev_a:.1f}x). "
              f"DuPont B: ROA = {pct(roa_b)} = {pct(roe_b)} / {levr_b:.2f}. "
              f"Turnover = {pct(margin_b)} / {pct(roa_b)} / {lev_b:.2f} "
              f"(implied: ~{roa_b/margin_b*lev_b:.1f}x). "
              f"A higher ROE can come from higher margin, higher turnover, "
              f"or higher leverage. The value creation question: does the spread "
              f"(ROA - cost of capital) justify the leverage? If cost of equity = "
              f"{pct(rng.uniform(0.10, 0.18))}, then spread for A = "
              f"{pct(roa_a - rng.uniform(0.10, 0.18))} vs B = "
              f"{pct(roa_b - rng.uniform(0.10, 0.18))}. "
              f"Leverage amplifies ROE but also amplifies risk. Company B "
              f"'s ROE of {pct(roe_b)} may be more sustainable: a {lev_b:.1f}x "
              f"D/E ratio exposes the firm more to downturns. The margin gap "
              f"of {pct(margin_diff)} in favor of A likely reflects a moat."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Ratios",
                 "difficulty":"CFA L1","question_type":"Analysis",
                 "pitfalls":["leverage vs operation efficiency","sustainability"]},
         "question":stem,"answer":answer}
    _out(r); return r


def fsa_cashflow_quality(rng, seq):
    """Cash flow quality: operating cash flow vs net income and red flags."""
    net_income = rng.uniform(50e6, 300e6)
    dga = rng.uniform(10e6, 50e6)
    stock_comp = rng.uniform(2e6, 15e6)
    working_capital_change = rng.uniform(5e6, 40e6)  # increase in WC
    capex = rng.uniform(15e6, 60e6)
    ocf = net_income + dga + stock_comp - working_capital_change - capex * 0.3
    fcf = ocf - capex
    ci = net_income / ocf if ocf > 0 else 0  # cash flow quality ratio
    
    stem = (f"net income {fmt(net_income/1e6, 0)}M, D&A {fmt(dga/1e6, 1)}M, "
            f"stock comp {fmt(stock_comp/1e6, 1)}M, WC increase "
            f"{fmt(working_capital_change/1e6, 1)}M, capex "
            f"{fmt(capex/1e6, 1)}M. Compute OCF, FCF, and CF quality ratio. "
            f"Any red flags?")
    answer = (f"OCF = NI + D&A + stock comp - WC change = "
              f"{fmt(net_income/1e6, 0)} + {fmt(dga/1e6, 1)} + "
              f"{fmt(stock_comp/1e6, 2)} - {fmt(working_capital_change/1e6, 1)} = "
              f"{fmt(ocf/1e6, 1)}M. "
              f"FCF = OCF - capex = "
              f"{fmt(ocf/1e6, 1)} - {fmt(capex/1e6, 1)} = "
              f"{round(fcf/1e6, 1)}M. "
              f"CF quality: {round(ci, 2)}x (NI/OCF). Values < 1.0 indicate "
              f"OCF > NI = quality accruals are conservative. Values > 1.0: "
              f"OCF < NI, suggesting accruals are boosting reported earnings. "
              f"A ratio of {round(ci, 2)} suggests " + ("good quality" if ci <= 1 else "accrual-driven earnings that may not persist") + ". "
              f"A growing WC balance relative to revenue growth is a red flag: "
              f"it can mean revenue is recognized but not yet collected (accounts "
              f"receivable buildup). The FCF measure shows cash available for "
              f"dividends and buybacks after maintaining capex."
              "")
    r = {"record_type":"analysis",
         "meta":{"topic":"Financial Statement Analysis","subtopic":"Cash Flow Quality",
                 "difficulty":"CFA L2","question_type":"Analysis",
                 "pitfalls":["accrual quality","WC interpretation"]},
         "question":stem,"answer":answer}
    _out(r); return r



# ---- Complete TEMPLATES registry ----
# Maps a short name to the generator function.  Each function signature: fn(rng, seq) -> dict

TEMPLATES: dict = {
    # Equities
    "eq_fcff_dcf": eq_fcff_dcf,
    "eq_residual_income": eq_residual_income,
    "eq_comparable_multiples": eq_comparable_multiples,
    "eq_pvt_company_val": eq_pvt_company_val,
    "eq_dividend_policy": eq_dividend_policy,
    "eq_revenue_based": eq_revenue_based,
    "eq_asset_based": eq_asset_based,
    # Fixed Income
    "fi_yield_curve": fi_yield_curve,
    "fi_duration_convexity": fi_duration_convexity,
    "fi_bond_pricing": fi_bond_pricing,
    "fi_credit_spreads": fi_credit_spreads,
    "fi_convertibles": fi_convertibles,
    "fi_inflation_linked": fi_inflation_linked,
    "fi_mbs_analysis": fi_mbs_analysis,
    # Quantitative
    "qs_time_series": qs_time_series,
    "qs_bayesian": qs_bayesian,
    "qs_monte_carlo": qs_monte_carlo,
    "qs_regression": qs_regression,
    "qs_hypothesis_testing": qs_hypothesis_testing,
    "qs_garch": qs_garch,
    "qs_nonparametric": qs_nonparametric,
    # Portfolio
    "pm_factor_model": pm_factor_model,
    "pm_rebalancing": pm_rebalancing,
    "pm_performance_attribution": pm_performance_attribution,
    "pm_tax_management": pm_tax_management,
    "pm_asset_allocation": pm_asset_allocation,
    "pm_behavioral_biases": pm_behavioral_biases,
    "pm_risk_budgeting": pm_risk_budgeting,
    # Risk
    "rm_var_computation": rm_var_computation,
    "rm_expected_shortfall": rm_expected_shortfall,
    "rm_stress_testing": rm_stress_testing,
    "rm_credit_risk": rm_credit_risk,
    "rm_model_risk": rm_model_risk,
    "rm_liquidity_risk": rm_liquidity_risk,
    "rm_operational_risk": rm_operational_risk,
    # FSA
    "fsa_revenue_recognition": fsa_revenue_recognition,
    "fsa_lease_accounting": fsa_lease_accounting,
    "fsa_pension_benefits": fsa_pension_benefits,
    "fsa_goodwill_impairment": fsa_goodwill_impairment,
    "fsa_inventory_methods": fsa_inventory_methods,
    "fsa_fraud_detection": fsa_fraud_detection,
    "fsa_ratios_analysis": fsa_ratios_analysis,
    "fsa_cashflow_quality": fsa_cashflow_quality,
}


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Generate analysis records for Cosimo v2")
    parser.add_argument("--template", default=None, help="Specific template to run")
    parser.add_argument("--count", type=int, default=3, help="Records per template")
    parser.add_argument("--out", default=None, help="Output JSONL file (default: stdout)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--programs", nargs="*", default=None,
                        help="Programs to include (defaults to all)")
    args = parser.parse_args()

    if args.out:
        _OUTFILE = args.out

    programs = args.programs or [
        "CFA_Level_I", "CFA_Level_II", "CFA_Level_III",
        "FRM_Level_I", "FRM_Level_II", "FRM_Level_III",
    ]

    total = 0
    if args.template:
        func = TEMPLATES.get(args.template)
        if func is None:
            print(f"ERROR: unknown template '{args.template}'. Available: {list(TEMPLATES.keys())}", file=sys.stderr)
            sys.exit(1)
        for i in range(args.count):
            rng = RNG(args.seed + i)
            func(rng, args.seed + i)
            total += 1
    else:
        for name, func in sorted(TEMPLATES.items()):
            print(f"  Running {name}...", file=sys.stderr)
            for i in range(args.count):
                rng = RNG(args.seed + i)
                func(rng, args.seed + i)
                total += 1

    print(f"# Total records: {total}", file=sys.stderr)
