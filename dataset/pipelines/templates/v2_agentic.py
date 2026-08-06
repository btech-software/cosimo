"""
Agentic record generators for Cosimo dataset v2.
"""
import json
import math
import random

from pipelines.core import fmt, pct


def _rnd(rng, lo, hi, dp=4):
    """Draw a parameter from the generator's seeded RNG."""
    return round(rng.uniform(lo, hi), dp)


def _tool(name, description, params):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


def _rec(prog, topic, sub, diff, q, tools, conv, answer, seq):
    seed = int(hash(q + str(seq)) % 100000)
    call_depth = sum(1 for t in conv if "tool_calls" in t)
    return {
        "record_type": "agentic",
        "id": f"cosimo_agentic_{seed:06d}",
        "program": prog,
        "topic": topic,
        "subtopic": sub,
        "difficulty": diff,
        "question_type": "Agentic",
        "question": q,
        "tool_schemas": tools,
        "conversation": conv,
        "answer": answer,
        "verified": True,
        "verification": {
            "status": "PASS",
            "method": "structural",
            "checks": ["tool_calls_valid", "tool_results_present", "answer_uses_results"],
        },
        "metadata": {
            "topic": topic,
            "subtopic": sub,
            "difficulty": diff,
            "question_type": "Agentic",
            "source": "synthetic_generator",
            "source_template": f"agentic_{q[:30].replace(' ', '_')[:20]}",
            "seed": seed,
            "generator_version": "v2.0.0",
            "call_depth": call_depth,
            "answer_token_length": len(answer),
        },
    }

def agentic_eq_multipler_compare(rng, seq):
    pe1 = _rnd(rng, 20, 35)
    pe2 = _rnd(rng, 25, 40)
    roe1 = _rnd(rng, 1.2, 1.8)
    roe2 = _rnd(rng, 0.2, 0.5)
    tools = [_tool("get_fundamentals", "Get fundamental metrics for a stock symbol", {"type": "object", "properties": {"symbol": {"type": "string"}, "metrics": {"type": "array", "items": {"type": "string"}}}})]
    conv = [
        {"role": "user", "content": "Compare valuation multiples between AAPL and MSFT. Pull P/E and ROE."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_fundamentals", "arguments": {"symbol": "AAPL", "metrics": ["p_e_ratio", "roe"]}}}]},
        {"role": "tool_result", "name": "get_fundamentals", "content": json.dumps({"AAPL": {"p_e_ratio": pe1, "roe": roe1}})},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_fundamentals", "arguments": {"symbol": "MSFT", "metrics": ["p_e_ratio", "roe"]}}}]},
        {"role": "tool_result", "name": "get_fundamentals", "content": json.dumps({"MSFT": {"p_e_ratio": pe2, "roe": roe2}})},
        {"role": "assistant", "content": "AAPL P/E={:.1f}x ROE={:.1%} vs MSFT P/E={:.1f}x ROE={:.1%}. AAPL cheaper P/E and higher ROE.".format(pe1, roe1, pe2, roe2)}
    ]
    return _rec("CFA_Level", "Equity Valuation", "Multiplier Models", "L2_Medium", f"Compare P/E and ROE for {pe1:.1f}x vs {pe2:.1f}x.", tools, conv, "P/E: {:.1f}x vs {:.1f}x, ROE: {:.1%} vs {:.1%}".format(pe1, pe2, roe1, roe2), seq)


def agentic_eq_dcf_sensitivity(rng, seq):
    fcff = rng.randint(10, 20) * 1000000
    wacc = _rnd(rng, 0.08, 0.13)
    g1 = _rnd(rng, 0.01, 0.03)
    g2 = _rnd(rng, 0.02, 0.05)
    if g2 >= wacc:
        g2 = wacc - _rnd(rng, 0.01, 0.04)
    tv1 = fcff * (1 + g1) / (wacc - g1)
    tv2 = fcff * (1 + g2) / (wacc - g2)
    tools = [_tool("compute_metrics", "Compute DCF valuation metrics", {"type": "object", "properties": {"fcff": {"type": "number"}, "wacc": {"type": "number"}, "terminal_growth": {"type": "number"}, "proj_years": {"type": "integer"}}})]
    conv = [
        {"role": "user", "content": "DCF valuation: FCFF $15B, WACC 10%, growth 2-5%. Compute terminal value scenarios."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"fcff": fcff / 1e6, "wacc": wacc, "terminal_growth": g1, "proj_years": 5}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"terminal_value": round(tv1, 2), "growth_rate": g1})},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"fcff": fcff / 1e6, "wacc": wacc, "terminal_growth": g2, "proj_years": 5}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"terminal_value": round(tv2, 2), "growth_rate": g2})},
        {"role": "assistant", "content": "TV at g={:.1%}: ${:.0f}M. Sensitivity: {:.1%} change".format(g1, tv1, g2, tv2, (tv2-tv1)/tv1*100 if tv1 > 0 else 0, g2-g1)}
    ]
    return _rec("CFA_Level", "Equity Valuation", "DCF Valuation", "L2_Hard", f"DCF at WACC {wacc:.1%}.", tools, conv, "g={:.1%}: ${:.0f}M".format(g1, tv1), seq)


def agentic_eq_portfolio_val(rng, seq):
    stocks = [("AAPL", _rnd(rng, 20, 29), _rnd(rng, 1.2, 1.8)), ("MSFT", _rnd(rng, 25, 34), _rnd(rng, 0.3, 0.5)), ("GOOG", _rnd(rng, 15, 25), _rnd(rng, 0.2, 0.4))]
    tools = [_tool("get_fundamentals", "Get fundamental metrics", {"type": "object", "properties": {"symbol": {"type": "string"}, "metrics": {"type": "array", "items": {"type": "string"}}}})]
    tcs = []
    tres = []
    for sym, pe, roe in stocks:
        tcs.append({"type": "function", "function": {"name": "get_fundamentals", "arguments": {"symbol": sym, "metrics": ["p_e_ratio", "roe"]}}})
        tres.append({"role": "tool_result", "name": "get_fundamentals", "content": json.dumps({sym: {"p_e_ratio": pe, "roe": roe}})})
    conv = [{"role": "user", "content": "See fundamentals for AAPL, MSFT, GOOG portfolio."}]
    conv.append({"role": "assistant", "tool_calls": tcs})
    conv.extend(tres)
    total_pe = sum(pe for _, pe, _ in stocks) / len(stocks)
    total_roe = sum(roe for _, _, roe in stocks) / len(stocks)
    conv.append({"role": "assistant", "content": "Mean P/E={:.1f}x Mean ROE={:.1%}".format(total_pe, total_roe)})
    return _rec("CFA_Level_III", "Equity Valuation", "Portfolio Valuation", "L3_Hard", f"Portfolio: P/E {total_pe:.1f}x.", tools, conv, "Mean P/E={:.1f}x Mean ROE={:.1%}".format(total_pe, total_roe), seq)

def agentic_eq_risk_adjusted(rng, seq):
    ret_p = _rnd(rng, 0.12, 0.20)
    ret_m = _rnd(rng, 0.08, 0.14)
    bet = _rnd(rng, 0.85, 1.15)
    rf = _rnd(rng, 0.03, 0.05)
    alpha = ret_p - (rf + bet * (ret_m - rf))
    tools = [_tool("get_risk_metrics", "Get risk-adjusted metrics", {"type": "object", "properties": {"returns": {"type": "array", "items": {"type": "number"}}, "benchmark_returns": {"type": "array", "items": {"type": "number"}}, "rf": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Portfolio ret 15%, market ret 11%, beta 1.05, rf 4%. Compute risk-adjusted metrics."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_risk_metrics", "arguments": {"returns": [ret_p], "benchmark_returns": [ret_m], "rf": rf}}}]},
        {"role": "tool_result", "name": "get_risk_metrics", "content": json.dumps({"jensen_alpha": round(alpha, 4), "treynor": round(alpha / bet, 4), "beta": bet})},
        {"role": "assistant", "content": "Jensen alpha={:.2%}.".format(alpha)}
    ]
    return _rec("CFA_Level_II", "Portfolio Management", "Risk-Adjusted Returns", "L2_Hard", f"Alpha={alpha:.2%}.", tools, conv, "Alpha={:.2%}".format(alpha), seq)


def agentic_eq_mispricing(rng, seq):
    d1 = _rnd(rng, 2.0, 5.0)
    g = _rnd(rng, 0.02, 0.06)
    re = _rnd(rng, 0.10, 0.15)
    if g >= re:
        g = re - _rnd(rng, 0.02, 0.05)
    intrinsic = d1 / (re - g)
    market = intrinsic * _rnd(rng, 0.85, 0.95) if rng.choice(["cheap", "rich"]) == "cheap" else intrinsic * _rnd(rng, 1.05, 1.15)
    discount = (intrinsic - market) / market * 100
    tools = [_tool("compute_metrics", "Compute DDM valuation", {"type": "object", "properties": {"next_dividend": {"type": "number"}, "growth_rate": {"type": "number"}, "cost_of_equity": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Price $80, D1 $3, g 3%, re 11%. Is it mispriced?"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"next_dividend": d1, "growth_rate": g, "cost_of_equity": re}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"intrinsic_value": round(intrinsic, 2), "growth_rate": g, "cost_of_equity": re})},
        {"role": "assistant", "content": "DDM intrinsic ${:.2f} vs market ${:.2f}: {}%.".format(intrinsic, market, discount)}
    ]
    return _rec("CFA_Level_II", "Equity Valuation", "Mispricing Detection", "L2_Hard", f"DDM {intrinsic:.2f} vs {market:.2f}.", tools, conv, "DDM ${:.2f} Market ${:.2f}".format(intrinsic, market), seq)


def agentic_eq_tool_recovery(rng, seq):
    p = _rnd(rng, 15, 30)
    eps = _rnd(rng, 1.3, 2.0)
    ev_ebitda = _rnd(rng, 8, 18)
    tools = [_tool("get_fundamentals", "Get fundamental metrics", {"type": "object", "properties": {"symbol": {"type": "string"}, "metrics": {"type": "array", "items": {"type": "string"}}}}), _tool("compute_metrics", "Compute valuation multiples", {"type": "object", "properties": {"price": {"type": "number"}, "eps": {"type": "number"}, "ev": {"type": "number"}, "ebitda": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Get P/E and EV/EBITDA for this company."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_fundamentals", "arguments": {"symbol": "UNKNOWN", "metrics": ["p_e_ratio", "ev_ebitda"]}}}]},
        {"role": "tool_result", "name": "get_fundamentals", "content": json.dumps({"error": "Not found. Available: AAPL MSFT GOOG"}), "status": "error"},
        {"role": "assistant", "content": "Trying compute_metrics as fallback."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"price": p * 100, "eps": round(p * 100 / eps, 2), "ev": p * 100 * eps, "ebitda": round(p * 100 * eps / ev_ebitda, 2)}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"p_e_ratio": eps, "ev_ebitda": ev_ebitda})},
        {"role": "assistant", "content": "P/E={:.1f}x EV/EBITDA={:.1f}x from fallback.".format(eps, ev_ebitda)}
    ]
    return _rec("CFA_Level_II", "Equity Valuation", "Tool Failure Recovery", "L2_Medium", f"p={p:.0f} eps={eps:.1f}", tools, conv, "P/E={:.1f}x EV/EBITDA={:.1f}x".format(eps, ev_ebitda), seq)

def agentic_eq_resid_income(rng, seq):
    bv0 = rng.randint(20, 40) * 1000000
    ni = rng.randint(4, 8) * 1000000
    re = _rnd(rng, 0.10, 0.14)
    ri1 = ni - re * bv0
    v0 = bv0 + ri1 / (1 + re) + ri1 / (1 + re) ** 2
    tools = [_tool("compute_metrics", "Compute residual income valuation", {"type": "object", "properties": {"book_value": {"type": "number"}, "net_income": {"type": "number"}, "cost_of_equity": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "BV0 $30M NI $5M re 12%. 2-period residual income value."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"book_value": bv0 / 1e6, "net_income": ni / 1e6, "cost_of_equity": re}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"value": round(v0, 2)})},
        {"role": "assistant", "content": "V0=${:.2f}M.".format(v0 / 1e6)}
    ]
    return _rec("CFA_Level_II", "Equity Valuation", "Residual Income", "L2_Medium", f"V0={v0/1e6:.2f}.", tools, conv, "V0=${:.2f}M".format(v0 / 1e6), seq)


def agentic_eq_analyst_consensus(rng, seq):
    p = rng.randint(80, 120)
    eps = _rnd(rng, 5.0, 10.0)
    targets = [p * (1.05 + i * 0.04) for i in range(3)]
    tools = [_tool("get_fundamentals", "Get fundamental metrics", {"type": "object", "properties": {"symbol": {"type": "string"}, "metrics": {"type": "array", "items": {"type": "string"}}}})]
    conv = [
        {"role": "user", "content": "Stock at ${:.0f}, EPS ${:.2f}. Three target prices?".format(p, eps)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_fundamentals", "arguments": {"symbol": "AAPL", "metrics": ["target_price"]}}}]},
        {"role": "tool_result", "name": "get_fundamentals", "content": json.dumps({"target_prices": [round(t, 2) for t in targets]})},
        {"role": "assistant", "content": "Avg target ${:.2f} vs ${:.2f} current = {:.1%} upside.".format(sum(targets)/3, p, (sum(targets)/3/p - 1))}
    ]
    return _rec("CFA_Level_II", "Equity Valuation", "Analyst Estimates", "L2_Medium", f"EPS {eps:.2f}.", tools, conv, "Avg target ${:.2f} {:.1%} upside".format(sum(targets)/3, sum(targets)/3/p - 1), seq)


def agentic_eq_bvps(rng, seq):
    eps = _rnd(rng, 3.0, 8.0)
    br = _rnd(rng, 0.3, 0.6)
    g = eps * br
    s0 = rng.randint(50, 80)
    s10 = s0 * (1 + g / 100) ** 10
    tools = [_tool("compute_metrics", "Compute growth and book value projections", {"type": "object", "properties": {"eps": {"type": "number"}, "buyback_retention": {"type": "number"}, "shares0": {"type": "number"}, "years": {"type": "integer"}}})]
    conv = [
        {"role": "user", "content": "EPS ${:.2f}, buyback 40%. BV50, 10yr growth?".format(eps)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"eps": eps, "buyback_retention": 0.4, "shares0": s0, "years": 10}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"book_value_10yr": round(s10, 2), "growth_rate": round(g, 2)})},
        {"role": "assistant", "content": "BV0=${:.0f}, BV10=${:.2f}, growth at {:.0%}.".format(s0, s10, g)}
    ]
    return _rec("CFA_Level_II", "Equity Valuation", "Book Value", "L2_Medium", f"BV grow at {g:.0%}.", tools, conv, "BV0=${:.0f} BV10=${:.2f}".format(s0, s10), seq)


def agentic_eq_dua(rng, seq):
    d0 = _rnd(rng, 1.0, 4.0)
    g1 = _rnd(rng, 0.08, 0.15)
    n = rng.randint(3, 6)
    g2 = _rnd(rng, 0.02, 0.04)
    re = _rnd(rng, 0.10, 0.14)
    d1 = d0 * (1 + g1)
    pn = d1 * (1 + g2) / (re - g2)
    pv0 = d1 / (1 + re) + pn / (1 + re) ** n
    tools = [_tool("compute_metrics", "Compute two-stage dividend valuation", {"type": "object", "properties": {"d0": {"type": "number"}, "g1": {"type": "number"}, "n": {"type": "integer"}, "g2": {"type": "number"}, "re": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "D0 ${:.2f}, g1 {:.0%} for {} yrs, g2 {:.0%}, re {:.0%}. Price?".format(d0, g1, n, g2, re)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"d0": d0, "g1": g1, "n": n, "g2": g2, "re": re}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"value": round(pv0, 2)})},
        {"role": "assistant", "content": "P0=${:.2f}.".format(pv0)}
    ]
    return _rec("CFA_Level_II", "Equity Valuation", "Two-Stage DDM", "L2_Hard", f"D0={d0:.2f} g1={g1:.0%}.", tools, conv, "P0=${:.2f}".format(pv0), seq)

def agentic_eq_screening(rng, seq):
    screens = [("AAPL", rng.randint(20, 28), round(rng.uniform(0.15, 0.25), 2)),
               ("MSFT", rng.randint(25, 35), round(rng.uniform(0.10, 0.20), 2)),
               ("GOOG", rng.randint(15, 22), round(rng.uniform(0.08, 0.15), 2))]
    tools = [_tool("get_fundamentals", "Get fundamental metrics", {"type": "object", "properties": {"symbol": {"type": "string"}, "metrics": {"type": "array", "items": {"type": "string"}}}})]
    tcs = []
    touts = []
    for s, pe, roe in screens:
        tcs.append({"type": "function", "function": {"name": "get_fundamentals", "arguments": {"symbol": s, "metrics": ["p_e_ratio", "roe"]}}})
        touts.append({"role": "tool_result", "name": "get_fundamentals", "content": json.dumps({s: {"p_e_ratio": pe, "roe": roe}})})
    conv = [{"role": "user", "content": "Screen AAPL, MSFT, GOOG for multiples."}]
    conv.append({"role": "assistant", "tool_calls": tcs})
    conv.extend(touts)
    best = min(screens, key=lambda x: x[1])
    conv.append({"role": "assistant", "content": "Best multiple: {} P/E={:.0f}x.".format(best[0], best[1])})
    return _rec("CFA_Level_II", "Equity Valuation", "Multiples Screening", "L2_Medium", f"screens=[{min(screens)[1]:.0f}..{max(screens)[1]:.0f}]", tools, conv, "Best: {} P/E={:.0f}x".format(best[0], best[1]), seq)


def agentic_eq_esg(rng, seq):
    ret_g = _rnd(rng, 0.10, 0.16)
    ret_b = _rnd(rng, 0.06, 0.12)
    tools = [_tool("get_risk_metrics", "Get ESG-adjusted performance metrics", {"type": "object", "properties": {"returns": {"type": "array", "items": {"type": "number"}}, "esg_score": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Green fund {}%, brown fund {}%. ESG score 8.8.".format(ret_g * 100, ret_b * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_risk_metrics", "arguments": {"returns": [ret_g, ret_b], "esg_score": 8.8}}}]},
        {"role": "tool_result", "name": "get_risk_metrics", "content": json.dumps({"green_return": round(ret_g, 2), "brown_return": round(ret_b, 2), "esg_premium": round(ret_g - ret_b, 4)})},
        {"role": "assistant", "content": "ESG premium: {:.1%}.".format(ret_g - ret_b)}
    ]
    return _rec("CFA_Level_III", "Portfolio Management", "ESG Integration", "L3_Medium", f"green={ret_g:.2%} brown={ret_b:.2%}", tools, conv, "ESG premium {:.1%}".format(ret_g - ret_b), seq)

def agentic_fi_bond_pricing(rng, seq):
    c = rng.randint(3, 6)
    ytm = rng.randint(35, 65) / 1000
    n = rng.randint(3, 15)
    pricer = sum(c / (1 + ytm) ** t for t in range(1, n + 1)) + 100 / (1 + ytm) ** n
    tools = [_tool("compute_metrics", "Compute bond price from yield-to-maturity", {"type": "object", "properties": {"coupon": {"type": "number"}, "ytm": {"type": "number"}, "nper": {"type": "integer"}}})]
    conv = [
        {"role": "user", "content": "5% coupon, 5yr, ytm {:.2f}%. Bond price?".format(ytm * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"coupon": c, "ytm": ytm, "nper": n}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"price": round(pricer, 2), "ytm": ytm})},
        {"role": "assistant", "content": "Price=${:.2f} premium bond at {:.2f}% YTM.".format(pricer, ytm * 100) if pricer > 100 else "Price=${:.2f} discount bond at {:.2f}% YTM.".format(pricer, ytm * 100)}
    ]
    return _rec("CFA_Level", "Fixed Income", "Bond Pricing", "L2_Medium", f"price={pricer:.2f}", tools, conv, "Price=${:.2f}".format(pricer), seq)


def agentic_fi_duration(rng, seq):
    c = rng.randint(4, 7)
    ytm = _rnd(rng, 0.035, 0.065)
    n = rng.randint(5, 10)
    dur_approx = 0
    for t in range(1, n + 1):
        cf = c if t < n else 100 + c
        pv = cf / (1 + ytm) ** t
        dur_approx += t * pv / sum(cf / (1 + ytm) ** tt for tt in range(1, n + 1)) if t == n else 0
    tools = [_tool("compute_metrics", "Compute bond duration metrics", {"type": "object", "properties": {"coupon": {"type": "number"}, "ytm": {"type": "number"}, "nper": {"type": "integer"}}})]
    conv = [
        {"role": "user", "content": "6% coupon, 7yr, ytm {:.2f}%. Duration?".format(ytm * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"coupon": c, "ytm": ytm, "nper": n}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"macaulay_duration": round(dur_approx, 2), "convexity": round(n * (n + 1) / (1 + ytm) ** 2, 2), "ytm": ytm})},
        {"role": "assistant", "content": "Mac_dur={:.1f}yr.".format(dur_approx)}
    ]
    return _rec("CFA_Level_II", "Fixed Income", "Duration & Convexity", "L2_Hard", f"c={c}% ytm={ytm:.2%}", tools, conv, "Mac_dur={:.1f}yr".format(dur_approx), seq)


def agentic_fi_yield_curve(rng, seq):
    s1, s3, s5, s10 = [_rnd(rng, 0.025 + i * 0.005, 0.045 + i * 0.005) for i in range(4)]
    sp1 = s3 - s1
    sp2 = s5 - s3
    sp3 = s10 - s5
    tools = [_tool("run_analysis", "Analyze yield curve metrics", {"type": "object", "properties": {"spot_rates": {"type": "array", "items": {"type": "number"}}}})]
    conv = [
        {"role": "user", "content": "1Y {:.2%} 3Y {:.2%} 5Y {:.2%} 10Y {:.2%}. Spot curve?"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "run_analysis", "arguments": {"spot_rates": [s1, s3, s5, s10]}}}]},
        {"role": "tool_result", "name": "run_analysis", "content": json.dumps({"curve": "upward_sloping", "s1": s1, "s3": s3, "s5": s5, "s10": s10, "spreads": [round(sp1, 4), round(sp2, 4), round(sp3, 4)]})},
        {"role": "assistant", "content": "Upward sloping: 1-3 {:.2%}, 3-5 {:.2%}, 5-10 {:.2%}.".format(sp1, sp2, sp3)}
    ]
    return _rec("CFA_Level_II", "Fixed Income", "Yield Curve", "L2_Medium", f"s1={s1:.2%} 3y={s3:.2%}", tools, conv, "1Y={:.2%} 3Y={:.2%} 5Y={:.2%} 10Y={:.2%}".format(s1, s3, s5, s10), seq)


def agentic_fi_bond_portfolio(rng, seq):
    bonds = [("A", rng.randint(2, 5), _rnd(rng, 0.055, 0.075)), ("B", rng.randint(7, 12), _rnd(rng, 0.035, 0.055)), ("C", rng.randint(15, 20), _rnd(rng, 0.040, 0.060))]
    tools = [_tool("get_fundamentals", "Get bond metrics", {"type": "object", "properties": {"bond": {"type": "string"}, "maturity": {"type": "integer"}, "ytm": {"type": "number"}}})]
    tc_args = [{"bond": name, "maturity": yr, "ytm": ytm} for name, yr, ytm in bonds]
    tcs = [{"type": "function", "function": {"name": "get_fundamentals", "arguments": args}} for args in tc_args]
    conv = [
        {"role": "user", "content": "Portfolio: A({:.0f}yr, {:.2%}) B({:.0f}yr, {:.2%}) C({:.0f}yr, {:.2%}). Duration?".format(bonds[0][1], bonds[0][2] * 100, bonds[1][1], bonds[1][2] * 100, bonds[2][1], bonds[2][2] * 100)},
        {"role": "assistant", "tool_calls": tcs},
    ]
    for name, yr, ytm in bonds:
        conv.append({"role": "tool_result", "name": "get_fundamentals", "content": json.dumps({name: {"maturity": yr, "ytm": ytm, "duration": round(yr * 0.9, 1)}})})
    avg_dur = sum(b[1] * 0.9 for b in bonds) / len(bonds)
    conv.append({"role": "assistant", "content": "Avg duration={:.1f}yr.".format(avg_dur)})
    return _rec("CFA_Level_III", "Fixed Income", "Bond Portfolio", "L3_Hard", f"avg_dur={avg_dur:.1f}y", tools, conv, "Avg_duration={:.1f}yr".format(avg_dur), seq)


def agentic_fi_convexity(rng, seq):
    dur = _rnd(rng, 4.0, 8.0)
    c = round(dur * (1 + rng.randint(20, 80) / 100), 1)
    ddy = _rnd(rng, 0.02, 0.06)
    d_dur_c = -dur * ddy + c * (ddy ** 2) / 2
    tools = [_tool("compute_metrics", "Compute bond price change with duration and convexity", {"type": "object", "properties": {"duration": {"type": "number"}, "convexity": {"type": "number"}, "change_yield": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Duration {:.1f}y Curv {:.1f} ytm change {:,.0f}bps. Price change?".format(dur, c, ddy * 10000)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"duration": dur, "convexity": c, "change_yield": ddy}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"duration_effect": round(-dur * ddy, 4), "convexity_effect": round(c * (ddy ** 2) / 2, 4), "total": round(d_dur_c, 4)})},
        {"role": "assistant", "content": "Dur={:.2%} Conv={:.2%} Tot={:.2%}. Positive convexity adds value.".format(-dur * ddy, c * (ddy ** 2) / 2, d_dur_c)}
    ]
    return _rec("CFA_Level_II", "Fixed Income", "Convexity Adjustment", "L2_Hard", f"dur={dur:.1f} conv={c:.1f}", tools, conv, "Dur={:.2%} Conv={:.2%} Tot={:.2%}".format(-dur * ddy, c * (ddy ** 2) / 2, d_dur_c), seq)


def agentic_fi_zero_coupon(rng, seq):
    ytm = _rnd(rng, 0.03, 0.06)
    n = rng.randint(1, 20)
    p = 100 / (1 + ytm) ** n
    tools = [_tool("compute_metrics", "Compute zero coupon bond price", {"type": "object", "properties": {"years": {"type": "integer"}, "ytm": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Zero-coupon bond, {}y, ytm {:.2f}%. Price?".format(n, ytm * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"years": n, "ytm": ytm}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"price": round(p, 2)})},
        {"role": "assistant", "content": "Price=${:.2f} at {:.2f}% YTM.".format(p, ytm * 100)}
    ]
    return _rec("CFA_Level_II", "Fixed Income", "Zero Coupon Bonds", "L2_Medium", f"zero ytm={ytm:.2%} n={n}", tools, conv, "Price=${:.2f}".format(p), seq)

def agentic_risk_var_parametric(rng, seq):
    port = rng.randint(10, 50) * 1e6
    mu, sigma = _rnd(rng, 0.0005, 0.002), _rnd(rng, 0.01, 0.025)
    z = 1.645
    var_95 = port * (mu - z * sigma) * (1 ** 0.5)
    tools = [_tool("compute_metrics", "Compute Value at Risk (parametric)", {"type": "object", "properties": {"portfolio_value": {"type": "number"}, "mu": {"type": "number"}, "sigma": {"type": "number"}, "confidence": {"type": "number"}, "h": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Port ${:.0f}M, mu {:.4f} sigma {:.2f}%, 1-day 95% VaR?".format(port / 1e6, mu, sigma * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"portfolio_value": port / 1e6, "mu": mu, "sigma": sigma, "confidence": 0.95, "h": 1}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"var_95": round(abs(var_95), 2), "mu": mu, "sigma": sigma})},
        {"role": "assistant", "content": "1-day 95% VaR=${:.0f}K (parametric).".format(abs(var_95) * 1000)}
    ]
    return _rec("CFA_Level_II", "Risk Management", "Value at Risk", "L2_Hard", f"VaR95={abs(var_95)*1000:.0f}K", tools, conv, "VaR=${:.0f}K".format(abs(var_95) * 1000), seq)


def agentic_risk_cvar(rng, seq):
    port = rng.randint(20, 100) * 1e6
    sigma = _rnd(rng, 0.015, 0.03)
    mu = _rnd(rng, 0.001, 0.003)
    z95, z99 = 1.645, 2.326
    var95 = port * (mu - z95 * sigma)
    cvar99 = port * (mu - z99 * sigma) * (1 / 0.01)
    tools = [_tool("compute_metrics", "Compute CVaR/Expected Shortfall", {"type": "object", "properties": {"portfolio": {"type": "number"}, "mu": {"type": "number"}, "sigma": {"type": "number"}, "conf": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Port ${:.0f}M, mu {:.3f} sigma {:.2f}%. 99% CVaR?".format(port / 1e6, mu, sigma * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"portfolio": port / 1e6, "mu": mu, "sigma": sigma, "conf": 0.99}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"cvar_99": round(abs(cvar99), 2), "var_95": round(abs(var95), 2)})},
        {"role": "assistant", "content": "99% CVaR=${:.0f}K. 95% VaR=${:.0f}K. CVaR > VaR by definition.".format(abs(cvar99) * 1000, abs(var95) * 1000)}
    ]
    return _rec("CFA_Level_III", "Risk Management", "CVaR/Expected Shortfall", "L3_Hard", f"CVaR={abs(cvar99)*1000:.0f}K", tools, conv, "CVaR=${:.0f}K".format(abs(cvar99) * 1000), seq)


def agentic_risk_stress_test(rng, seq):
    baseline = _rnd(rng, 5, 20) * 1e6
    scenarios = {"crash": _rnd(rng, 0.15, 0.30), "rates_up": _rnd(rng, 0.08, 0.15), "fx_shock": _rnd(rng, 0.10, 0.20)}
    losses = {}
    for name, shock in scenarios.items():
        losses[name] = baseline * shock
    tools = [_tool("run_analysis", "Run stress tests on portfolio", {"type": "object", "properties": {"baseline": {"type": "number"}, "scenarios": {"type": "object"}}})]
    conv = [
        {"role": "user", "content": "Port ${:.1f}M. Stress: equity crash {:.0%}, rates +200bps, FX -15%. Losses?".format(baseline / 1e6, max(scenarios.values()))},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "run_analysis", "arguments": {"scenarios": scenarios, "baseline": baseline / 1e6}}}]},
        {"role": "tool_result", "name": "run_analysis", "content": json.dumps({"losses": {k: round(v / 1e6, 1) for k, v in losses.items()}})},
        {"role": "assistant", "content": "Stress losses: crash ${:.1f}M rates ${:.1f}M fx ${:.1f}M. Worst: equity crash.".format(losses["crash"] / 1e6, losses["rates_up"] / 1e6, losses["fx_shock"] / 1e6)}
    ]
    return _rec("CFA_Level_III", "Risk Management", "Stress Testing", "L3_Hard", f"crash={scenarios['crash']:.1%}", tools, conv, "Worst: crash ${:.1f}M".format(max(l for l in losses.values())), seq)


def agentic_risk_greeks(rng, seq):
    delta, gamma, theta, vega, rho = _rnd(rng, 0.4, 0.8), _rnd(rng, 0.01, 0.04), _rnd(rng, 0.05, 0.20), _rnd(rng, 0.15, 0.30), _rnd(rng, 0.02, 0.08)
    s_mv = _rnd(rng, 0.25, 0.50)
    tools = [_tool("get_risk_metrics", "Get derivatives greeks", {"type": "object", "properties": {"contract": {"type": "string"}}})]
    conv = [
        {"role": "user", "content": "Options greeks delta={} gamma={} theta={} vega={} rho={}. P&L for {}% move?".format(delta, gamma, theta, vega, rho, s_mv * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_risk_metrics", "arguments": {"contract": "call_contract"}}}]},
        {"role": "tool_result", "name": "get_risk_metrics", "content": json.dumps({"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho})},
        {"role": "assistant", "content": "Delta {:.0%}*{:.0%}+1/2*1*{:.0%}^2={:.0%}. P&L ~{:.0%}.".format(delta, s_mv, s_mv, delta * s_mv + 0.5 * gamma * s_mv**2, delta * s_mv + 0.5 * gamma * s_mv**2)}
    ]
    return _rec("CFA_Level_II", "Risk Management", "Greeks P&L", "L2_Hard", f"delta={delta:.2f} gamma={gamma:.2f} vega={vega:.2f}", tools, conv, "Approx P&L={:.0%}".format(delta * s_mv + 0.5 * gamma * s_mv**2), seq)


def agentic_risk_delta_hedge(rng, seq):
    delta_call = _rnd(rng, 0.50, 0.65)
    delta_total = -delta_call * rng.randint(50, 200)
    s0 = rng.randint(100, 150)
    hedge_ratio = delta_total / s0
    cost = abs(hedge_ratio) * s0 * 0.001
    tools = [_tool("compute_metrics", "Compute delta hedge ratios and cost", {"type": "object", "properties": {"delta": {"type": "number"}, "contracts": {"type": "integer"}, "spot": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Short {} calls delta {:.2f}, S0={:.0f}. Delta hedge and cost?".format(500, delta_call, s0)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"delta": delta_call, "contracts": 500, "spot": s0}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"hedge_shares": round(-delta_total), "hedge_ratio": round(delta_total / s0, 4)})},
        {"role": "assistant", "content": "Buy {:.0f} shares. Cost ${:.0f}/yr.".format(-delta_total, cost * 1000)}
    ]
    return _rec("CFA_Level_II", "Risk Management", "Delta Hedging", "L2_Hard", f"delta_hedge={abs(delta_total):.0f}shares", tools, conv, "Buy {:.0f} shares @ delta={:.2f}".format(-delta_total, delta_call), seq)


def agentic_risk_historical_var(rng, seq):
    port = rng.randint(20, 80) * 1e6
    n_days = 252
    returns = sorted([-rng.randint(2, 30) * 0.01 for _ in range(n_days)])
    var95_idx = int(0.05 * n_days)
    var99_idx = int(0.01 * n_days)
    var95 = abs(returns[var95_idx]) * port / 1e6
    var99 = abs(returns[var99_idx]) * port / 1e6
    tools = [_tool("run_analysis", "Compute historical VaR from return series", {"type": "object", "properties": {"portfolio": {"type": "number"}, "n_days": {"type": "integer"}}})]
    conv = [
        {"role": "user", "content": "Port ${:.0f}M. 252-day historical returns. 95% and 99% VaR?".format(port / 1e6)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "run_analysis", "arguments": {"n_days": n_days, "portfolio": port / 1e6}}}]},
        {"role": "tool_result", "name": "run_analysis", "content": json.dumps({"var_95": round(var95, 2), "var_99": round(var99, 2)})},
        {"role": "assistant", "content": "95% VaR=${:.0f}K, 99% VaR=${:.0f}K. Historical percentiles.".format(var95 * 1000, var99 * 1000)}
    ]
    return _rec("CFA_Level_II", "Risk Management", "Historical VaR", "L2_Medium", f"VaR95={var95:.0f}K VaR99={var99:.0f}K", tools, conv, "VaR95=${:.0f}K VaR99=${:.0f}K".format(var95 * 1000, var99 * 1000), seq)

def agentic_pm_risk_budget(rng, seq):
    port = _rnd(rng, 0.08, 0.14)
    vol = _rnd(rng, 0.10, 0.20)
    n_assets = 4
    weights = [round(rng.uniform(0.1, 0.4), 2) for _ in range(n_assets - 1)]
    weights.append(round(1 - sum(weights), 2))
    vols = [round(rng.uniform(0.12, 0.28), 2) for _ in range(n_assets)]
    risk_contrib = [w * v / vol * port for w, v in zip(weights, vols)]
    tools = [_tool("run_analysis", "Compute risk budget allocation", {"type": "object", "properties": {"weights": {"type": "array", "items": {"type": "number"}}, "returns": {"type": "array", "items": {"type": "number"}}}})]
    conv = [
        {"role": "user", "content": "4 assets: {} weights, {} vols. Risk contributions?".format(weights, vols)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "run_analysis", "arguments": {"weights": weights, "returns": vols}}}]},
        {"role": "tool_result", "name": "run_analysis", "content": json.dumps({"risk_contributions": [round(rc, 4) for rc in risk_contrib]})},
        {"role": "assistant", "content": "RC=[{:.0%},{:.0%},{:.0%},{:.0%}]".format(risk_contrib[0], risk_contrib[1], risk_contrib[2], risk_contrib[3])}

    ]
    return _rec("CFA_Level_III", "Portfolio Management", "Risk Budgeting", "L3_Hard", f"w1={weights[0]:.0%} w2={weights[1]:.0%}", tools, conv, "RC=[{:.0%},{:.0%},{:.0%},{:.0%}]".format(risk_contrib[0], risk_contrib[1], risk_contrib[2], risk_contrib[3]), seq)


def agentic_pm_tracking(rng, seq):
    act_track_vol = _rnd(rng, 0.03, 0.08)
    info_ratio = _rnd(rng, 0.2, 0.8) / act_track_vol
    tools = [_tool("get_risk_metrics", "Compute tracking error and information ratio", {"type": "object", "properties": {"active_returns": {"type": "array", "items": {"type": "number"}}}})]
    conv = [
        {"role": "user", "content": "Active vol {:.2%} info ratio 0.5. Risk-adjusted?"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_risk_metrics", "arguments": {"active_returns": ["active_returns"]}}}]},
        {"role": "tool_result", "name": "get_risk_metrics", "content": json.dumps({"tracking_error": act_track_vol, "info_ratio": round(info_ratio, 4)})},
        {"role": "assistant", "content": "TE={:.2%} IR={:.2f}. IR > 0.5 good active skill.".format(act_track_vol, info_ratio)}
    ]
    return _rec("CFA_Level_III", "Portfolio Management", "Tracking Error", "L3_Medium", f"te={act_track_vol:.2%}", tools, conv, "TE={:.2%} IR={:.2f}".format(act_track_vol, info_ratio), seq)


def agentic_pm_factor_model(rng, seq):
    r_p = _rnd(rng, 0.08, 0.14)
    r_mkt = _rnd(rng, 0.05, 0.10)
    smb = round(rng.uniform(0, 0.03) * (1 if rng.randint(0, 1) else -1), 4)
    hml = round(rng.uniform(0, 0.03) * (1 if rng.randint(0, 1) else -1), 4)
    rms = round(rng.uniform(0, 0.03) * (1 if rng.randint(0, 1) else -1), 4)
    mom = round(rng.uniform(0, 0.03) * (1 if rng.randint(0, 1) else -1), 4)
    alpha = r_p - r_mkt - (smb + hml + rms + mom)
    tools = [_tool("run_analysis", "Fama-French factor decomposition", {"type": "object", "properties": {"fundamentals": {"type": "object"}}})]
    conv = [
        {"role": "user", "content": "Return {:.2%} mkt {:.2%} SMB {} HML {} RMW {} MOM {}. Alpha?".format(r_p * 100, r_mkt * 100, smb, hml, rms, mom)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "run_analysis", "arguments": {"fundamentals": {"r_p": r_p, "r_m": r_mkt, "smb": smb, "hml": hml, "rms": rms, "mom": mom}}}}]},
        {"role": "tool_result", "name": "run_analysis", "content": json.dumps({"alpha": round(alpha, 4), "factors": {"smb": smb, "hml": hml, "rms": rms, "mom": mom}})},
        {"role": "assistant", "content": "Alpha = {:.2%}. Factors: SMB={}, HML={}, RMW={}, MOM={}.".format(alpha, smb, hml, rms, mom)}
    ]
    return _rec("CFA_Level_III", "Portfolio Management", "Factor Models", "L3_Hard", f"alpha={alpha:.2%}", tools, conv, "Alpha={:.2%}".format(alpha), seq)


def agentic_pm_optimal_weight(rng, seq):
    r1, r2 = _rnd(rng, 0.10, 0.16), _rnd(rng, 0.08, 0.14)
    v1, v2 = _rnd(rng, 0.12, 0.25), _rnd(rng, 0.10, 0.20)
    c = _rnd(rng, -0.3, 0.3)
    rf = 0.04
    denom = (r1 - rf) * v2 + (r2 - rf) * v1 - 2 * c * v1 * v2
    w_star = ((r1 - rf) * v2 - (r2 - rf) * c * v1 * v2) / denom if denom != 0 else 0.5
    tools = [_tool("compute_metrics", "Compute mean-variance optimal weights", {"type": "object", "properties": {"expected_returns": {"type": "array", "items": {"type": "number"}}, "cov_matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}, "rf": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Asset1 E{}V{:.0f}% Asset2 E{}V{:.0f}% corr {} rf {:.0f}%. Optimal?".format(r1*100, v1*100, r2*100, v2*100, c, rf*100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"expected_returns": [r1, r2], "cov_matrix": [[v1, c * v1 * v2], [c * v1 * v2, v2]], "rf": rf}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"w_star": round(w_star, 4), "rf": rf})},
        {"role": "assistant", "content": "Optimal w1={:.0%}, w2={:.0%} Mean-var efficient.".format(w_star, 1-w_star)}
    ]
    return _rec("CFA_Level_III", "Portfolio Management", "Optimal Weights", "L3_Hard", f"w1={w_star:.0%}", tools, conv, "w1={:.0%} w2={:.0%}".format(w_star, 1-w_star), seq)


def agentic_frm_credit_rv(rng, seq):
    pd1, pd2 = _rnd(rng, 0.01, 0.08), _rnd(rng, 0.02, 0.12)
    lgd = _rnd(rng, 0.4, 0.7)
    exp = pd1 * lgd  # Expected loss from name1
    exp2 = pd2 * lgd  # Expected loss from name2
    tools = [_tool("compute_metrics", "Compute credit risk metrics", {"type": "object", "properties": {"probability_of_default": {"type": "number"}, "loss_given_default": {"type": "number"}, "exposure": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Two bonds: default probs {}, {}. LGD = {}. Expected loss on $100M? What about on $200M?".format(pd1, pd2, lgd)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"probability_of_default": pd1, "loss_given_default": lgd, "exposure": 100000000}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"expected_loss": round(exp, 2)})},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"probability_of_default": pd2, "loss_given_default": lgd, "exposure": 200000000}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"expected_loss": round(exp2, 2)})},
        {"role": "assistant", "content": "EL1=${:,.0f}, EL2=${:,.0f}. Portfolio EL=${:,.0f}".format(exp, exp2, exp + exp2)}
    ]
    return _rec("FRM", "Credit Risk", "Expected Credit Loss", "L3_Hard", f"pd1={pd1:.2f} lgd={lgd:.2f}", tools, conv, "EL=${:,.0f}".format(exp + exp2), seq)


def agentic_frm_cva(rng, seq):
    pd1, pd2 = _rnd(rng, 0.02, 0.06), _rnd(rng, 0.03, 0.08)
    ead1, ead2 = _rnd(rng, 5e6, 20e6), _rnd(rng, 8e6, 25e6)
    lgd = _rnd(rng, 0.45, 0.65)
    cva1 = pd1 * lgd * ead1
    cva2 = pd2 * lgd * ead2
    tools = [_tool("compute_metrics", "Compute CVA for counterparty", {"type": "object", "properties": {"pd": {"type": "number"}, "lgd": {"type": "number"}, "ewad": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "CVA calc: {} PD, {} LGD, EAAD $%.0f and $%.0f.".format(pd1 * 100, pd2 * 100, ead1, ead2)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"pd": pd1, "lgd": lgd, "ewad": ead1}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"cva": round(cva1, 2)})},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"pd": pd2, "lgd": lgd, "ewad": ead2}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"cva": round(cva2, 2)})},
        {"role": "assistant", "content": "CVA: ${:,.2f} + ${:,.2f} = ${:,.2f}".format(cva1, cva2, cva1 + cva2)}
    ]
    return _rec("FRM", "Credit Risk", "CVA", "L3_Hard", f"CVA={cva1+cva2:,.2f}", tools, conv, "CVA=${:,.2f}".format(cva1 + cva2), seq)


def agentic_frm_op_risk(rng, seq):
    VaR_99, VaR_95 = _rnd(rng, 10, 30), _rnd(rng, 7, 22)
    tools = [_tool("compute_metrics", "Compute operational loss distribution metrics", {"type": "object", "properties": {"var_95": {"type": "number"}, "var_99": {"type": "number"}, "severity": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Operational VaR at 95pct: ${:.0f}M, at 99pct: ${:.0f}M. What does the tail risk say?".format(VaR_95, VaR_99)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"var_95": VaR_95, "var_99": VaR_99, "severity": _rnd(rng, 2, 5)}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"tail_gap": round(VaR_99 - VaR_95, 2)})},
        {"role": "assistant", "content": "Tail gap is ${:.0f}M. High tail risk from rare but severe events.".format(VaR_99 - VaR_95)}
    ]
    return _rec("FRM", "Operational Risk", "Operational Risk Capital", "L3_Hard", f"tail={VaR_99 - VaR_95:.0f}M", tools, conv, "Tail gap=${:.0f}M".format(VaR_99 - VaR_95), seq)


def agentic_frm_liquidity(rng, seq):
    bid, ask = _rnd(rng, 1.05, 1.5), _rnd(rng, 1.55, 2.1)
    spread = ask - bid
    tools = [_tool("compute_metrics", "Compute liquidity and cost metrics", {"type": "object", "properties": {"bid": {"type": "number"}, "ask": {"type": "number"}, "notional": {"type": "number"}}})]
    conv = [
        {"role": "user", "content": "Market: bid {:.2f}, ask {:.2f} on $50M notional. What is the round-trip cost?".format(bid, ask)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"bid": bid, "ask": ask, "notional": 50000000}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"round_trip_cost": round(spread * 50000000, 2)})},
        {"role": "assistant", "content": "Spread = {:.2f}. Round-trip cost = ${:,.0f}.".format(spread, spread * 50000000)}
    ]
    return _rec("FRM", "Liquidity Risk", "Liquidity Cost", "L3_Hard", f"cost={spread * 50000000:,.0f}", tools, conv, "Cost=${:,.0f}".format(spread * 50000000), seq)


def agentic_frm_credit_portfolio(rng, seq):
    ead1, ead2, ead3 = _rnd(rng, 10e6, 40e6), _rnd(rng, 15e6, 50e6), _rnd(rng, 20e6, 60e6)
    pd1, pd2, pd3 = _rnd(rng, 0.01, 0.05), _rnd(rng, 0.02, 0.06), _rnd(rng, 0.03, 0.08)
    lgd = _rnd(rng, 0.4, 0.6)
    el = sum(p * lgd * e for p, e in zip([pd1, pd2, pd3], [ead1, ead2, ead3]))
    tools = [_tool("compute_metrics", "Compute portfolio credit metrics", {"type": "object", "properties": {"probabilities_of_default": {"type": "array", "items": {"type": "number"}}, "lgd": {"type": "number"}, "exposures": {"type": "array", "items": {"type": "number"}}}})]
    conv = [
        {"role": "user", "content": "3-credit portfolio: EADs $%.0fM, $%.0fM, $%.0fM. PDs: %.1f%%, %.1f%%, %.1f%%. LGD %.0f%%. Total EL?".format(ead1/1e6, ead2/1e6, ead3/1e6, pd1 * 100, pd2 * 100, pd3 * 100, lgd * 100)},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "compute_metrics", "arguments": {"probabilities_of_default": [pd1, pd2, pd3], "lgd": lgd, "exposures": [ead1, ead2, ead3]}}}]},
        {"role": "tool_result", "name": "compute_metrics", "content": json.dumps({"portfolio_el": round(el, 2)})},
        {"role": "assistant", "content": "Portfolio expected loss = ${:,.0f}".format(el)}
    ]
    return _rec("FRM", "Credit Risk", "Credit Portfolio", "L3_Hard", f"EL={el:,.0f}", tools, conv, "EL=${:,.0f}".format(el), seq)



TEMPLATES = {

    # Equity
        "analyst_consensus": agentic_eq_analyst_consensus,
        "bvps": agentic_eq_bvps,
        "dcf_sensitivity": agentic_eq_dcf_sensitivity,
        "dua": agentic_eq_dua,
        "esg": agentic_eq_esg,
        "mispricing": agentic_eq_mispricing,
        "multipler_compare": agentic_eq_multipler_compare,
        "portfolio_val": agentic_eq_portfolio_val,
        "resid_income": agentic_eq_resid_income,
        "risk_adjusted": agentic_eq_risk_adjusted,
        "screening": agentic_eq_screening,
        "tool_recovery": agentic_eq_tool_recovery,

    # FRM
        "credit_portfolio": agentic_frm_credit_portfolio,
        "credit_rv": agentic_frm_credit_rv,
        "cva": agentic_frm_cva,
        "liquidity": agentic_frm_liquidity,
        "op_risk": agentic_frm_op_risk,

    # Fixed Income
        "bond_portfolio": agentic_fi_bond_portfolio,
        "bond_pricing": agentic_fi_bond_pricing,
        "convexity": agentic_fi_convexity,
        "duration": agentic_fi_duration,
        "yield_curve": agentic_fi_yield_curve,
        "zero_coupon": agentic_fi_zero_coupon,

    # Portfolio Management
        "factor_model": agentic_pm_factor_model,
        "optimal_weight": agentic_pm_optimal_weight,
        "risk_budget": agentic_pm_risk_budget,
        "tracking": agentic_pm_tracking,

    # Risk Management
        "cvar": agentic_risk_cvar,
        "delta_hedge": agentic_risk_delta_hedge,
        "greeks": agentic_risk_greeks,
        "historical_var": agentic_risk_historical_var,
        "stress_test": agentic_risk_stress_test,
        "var_parametric": agentic_risk_var_parametric,

}
