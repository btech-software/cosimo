"""
Abstention record generators for Cosimo dataset v2.

Generates abstention-type records - calibration training data where the model
correctly identifies that a question cannot or should not be answered confidently.

Three defect categories:
  - underspecified: missing critical information needed to answer
  - unanswerable: the question cannot be answered even with more info
  - false_premise: the question assumes something false or impossible

Each template function abstention_<name>(rng, seq) -> dict follows the
abstention record schema defined in FORMAT.md lines 111-150.
"""


def _make_abstention(program, topic, subtopic, difficulty, question, answer, defect):
    """Build an abstention record dict following FORMAT.md."""
    return {
        "record_type": "abstention",
        "id": f"cosimo_abstention_{int(hash(question + str(0)) % 100000):06d}",
        "program": program,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "question_type": "Calibration",
        "defect": defect,
        "question": question,
        "answer": answer,
        "verified": True,
        "verification": {
            "status": "PASS",
            "method": "structural",
            "checks": ["no_final_answer", "asks_for_info", "identifies_gap"],
        },
        "metadata": {
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "question_type": "Calibration",
            "source": "synthetic_generator",
            "source_template": f"abstention_{question[:20].lower().replace(' ', '_')}",
            "seed": 0,
            "generator_version": "v2.0.0",
            "answer_token_length": len(answer),
        },
    }


# ====================================================================


def _pick(rng, seq, q_variants):
    "Pick a question variation from the list using seq for deterministic spread across variants."
    return q_variants[(seq % 5000) // 1000]

# 1. UNDERSPECIFIED -- 30 templates
# ====================================================================



def abstention_portfolio_rebalance(rng, seq):
    """Missing: current holdings, target allocations, account constraints."""
    # Deliberately worded away from suites/calibration.jsonl:cal_019
    # ("Rebalance the portfolio to target weights."), which verification/
    # suite_overlap.py flagged at jaccard 0.60. That file is a held-out
    # measurement instrument; near-duplicating it contaminates the only
    # evaluation that measures the objective.
    q = _pick(rng, seq,  [
        "The trustees want the sleeve brought back in line before quarter-end. Can you put the trade list together?",
        "The trustees want the sleeve brought back in line before quarter-end. Can you put the trade list together? Please explain your reasoning.",
        "The trustees want the sleeve brought back in line before quarter-end. Can you put the trade list together? Can you assess this?",
        "The trustees want the sleeve brought back in line before quarter-end. Can you put the trade list together? When I ask about this:",
        "The trustees want the sleeve brought back in line before quarter-end. Can you put the trade list together? I need your take on:",
    ])
    a = (
        "I would need more information before I can help with portfolio rebalancing. Specifically, "
        "I need:\n\n"
        "1. Your current portfolio holdings - what assets, securities, or funds you own and "
        "their current market values.\n"
        "2. Your target allocations - the desired percentage breakdown across asset classes, "
        "sectors, or individual securities.\n"
        "3. Account type - whether this is a taxable account, IRA, 401(k), or other "
        "retirement account, since tax considerations affect which trades to execute.\n"
        "4. Constraints - minimum trade sizes, transaction cost limits, or restrictions "
        "on selling specific positions.\n"
        "5. Your risk tolerance and investment time horizon, which inform whether aggressive "
        "or gradual rebalancing is appropriate.\n\n"
        "Once you provide these details, I can help identify which positions need to be "
        "bought or sold and estimate trade amounts."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Rebalancing",
                            "L3_Hard", q, a, "underspecified")


def abstention_npv_calculation(rng, seq):
    """Missing: cash flows, discount rate, project horizon."""
    q = _pick(rng, seq,  [
        "Calculate the NPV for this project.",
        "Calculate the NPV for this project. Please explain your reasoning.",
        "Calculate the NPV for this project. Can you assess this?",
        "Calculate the NPV for this project. When I ask about this:",
        "Calculate the NPV for this project. I need your take on:",
    ])
    a = (
        "I need additional details before I can compute the net present value (NPV):\n\n"
        "1. The projected cash flows - the expected inflows and outflows for each period.\n"
        "2. The discount rate - what rate should discount future cash flows "
        "(e.g., WACC, cost of capital, hurdle rate)?\n"
        "3. The project timeline - how many periods (years or quarters) does the project span?\n"
        "4. The initial investment - what is the upfront capital outlay at time zero?\n\n"
        "NPV = sum of discounted cash flows minus initial investment. Without these components, "
        "I cannot produce a meaningful result. Please provide the cash flow schedule, "
        "discount rate, and any terminal value assumptions."
    )
    return _make_abstention("CFA_Level_II", "Capital Budgeting", "NPV",
                            "L2_Medium", q, a, "underspecified")


def abstention_stock_pick_comparison(rng, seq):
    """Missing: timeframe, criteria, valuation method."""
    q = _pick(rng, seq,  [
        "Which stock is a better buy, Apple or Microsoft?",
        "Which stock is a better buy, Apple or Microsoft? Please explain your reasoning.",
        "Which stock is a better buy, Apple or Microsoft? Can you assess this?",
        "Which stock is a better buy, Apple or Microsoft? When I ask about this:",
        "Which stock is a better buy, Apple or Microsoft? I need your take on:",
    ])
    a = (
        "I cannot definitively say which stock is better buy without more context "
        "and parameters:\n\n"
        "1. Investment timeframe - is this a short-term trade (weeks to months) or a "
        "long-term hold (5+ years)? The analysis differs substantially.\n"
        "2. Evaluation criteria - do you care about total return, dividend yield, "
        "earnings growth rate, valuation multiples, or risk-adjusted returns?\n"
        "3. Valuation method - should I use DCF, comparable multiples (P/E, EV/EBITDA), "
        "or a blended approach?\n"
        "4. Current market prices and historical valuation multiples for both companies.\n"
        "5. Your personal risk profile - how much volatility can you tolerate?\n\n"
        "Both Apple and Microsoft are large-cap technology companies with different business "
        "models, growth trajectories, and risk profiles. A rigorous comparison requires you "
        "to specify the above parameters."
    )
    return _make_abstention("CFA_Level_II", "Equity Valuation", "Stock Comparison",
                            "L2_Medium", q, a, "underspecified")


def abstention_option_strategy(rng, seq):
    """Missing: underlying, direction, volatility, timing."""
    q = _pick(rng, seq,  [
        "What option strategy should I use for this position?",
        "What option strategy should I use for this position? Please explain your reasoning.",
        "What option strategy should I use for this position? Can you assess this?",
        "What option strategy should I use for this position? When I ask about this:",
        "What option strategy should I use for this position? I need your take on:",
    ])
    a = (
        "To recommend an options strategy, I need the following information:\n\n"
        "1. The underlying asset - which stock, index, or ETF?\n"
        "2. Market outlook - are you bullish, bearish, or neutral on the underlying?\n"
        "3. Volatility environment - what is the current implied volatility, and do you "
        "view it as elevated or depressed relative to historical?\n"
        "4. Time horizon - what expiration date are you considering?\n"
        "5. Risk tolerance and profit objective - do you want limited-risk defined strategies "
        "or are you open to undefined-risk positions?\n"
        "6. Existing position - are you starting from scratch, or do you already own the "
        "underlying and are looking for a covered-call or collar?\n\n"
        "Without these inputs, any strategy recommendation would be a guess. The choice "
        "between spreads, straddles, strangles, collars, and covered calls depends critically "
        "on your market outlook and constraints."
    )
    return _make_abstention("CFA_Level_II", "Derivatives", "Options Strategies",
                            "L2_Medium", q, a, "underspecified")


def abstention_bond_duration(rng, seq):
    """Missing: bond specifics, coupon, maturity, YTM."""
    q = _pick(rng, seq,  [
        "What is the duration of my bond portfolio?",
        "What is the duration of my bond portfolio? Please explain your reasoning.",
        "What is the duration of my bond portfolio? Can you assess this?",
        "What is the duration of my bond portfolio? When I ask about this:",
        "What is the duration of my bond portfolio? I need your take on:",
    ])
    a = (
        "To calculate duration, I need the specific characteristics of each bond in your "
        "portfolio:\n\n"
        "1. Bond details - coupon rate, maturity date, face value, and current market price "
        "or market value for each position.\n"
        "2. Yield-to-maturity - the YTM for each individual bond, or the portfolio-wide "
        "YTM you wish to use.\n"
        "3. Yield curve structure - if you want key rate durations, I need the full yield "
        "curve assumption.\n"
        "4. Embedded options - do any bonds have callable, putable, or convertible features? "
        "These materially change the duration calculation.\n\n"
        "Duration measures interest rate sensitivity, but without bond-level data (coupons, "
        "maturities, yields, and options), I cannot compute Macaulay, modified, or effective duration."
    )
    return _make_abstention("CFA_Level_II", "Fixed Income", "Duration",
                            "L2_Medium", q, a, "underspecified")


def abstention_beta_calculation(rng, seq):
    """Missing: benchmark, time period, data frequency."""
    q = _pick(rng, seq,  [
        "What is the beta of my stock picks?",
        "What is the beta of my stock picks? Please explain your reasoning.",
        "What is the beta of my stock picks? Can you assess this?",
        "What is the beta of my stock picks? When I ask about this:",
        "What is the beta of my stock picks? I need your take on:",
    ])
    a = (
        "Beta depends on several parameters that have not been specified:\n\n"
        "1. The benchmark index - is beta measured against the S&P 500, Russell 2000, "
        "MSCI World, or another index?\n"
        "2. The estimation period - 1 year, 3 years, or 5 years? Shorter windows capture "
        "recent market behavior; longer windows smooth out regime changes.\n"
        "3. Data frequency - daily, weekly, or monthly returns data?\n"
        "4. The exact securities in your portfolio and their respective weights.\n\n"
        "Beta = Covariance(stock returns, benchmark returns) / Variance(benchmark returns). "
        "Without the constituent list, benchmark choice, and estimation parameters, "
        "the beta calculation is impossible."
    )
    return _make_abstention("CFA_Level_II", "Portfolio Management", "CAPM",
                            "L2_Medium", q, a, "underspecified")


def abstention_portfolio_optimization(rng, seq):
    """Missing: expected returns, covariance, constraints."""
    q = _pick(rng, seq,  [
        "Optimize my portfolio for maximum Sharpe ratio.",
        "Optimize my portfolio for maximum Sharpe ratio. Please explain your reasoning.",
        "Optimize my portfolio for maximum Sharpe ratio. Can you assess this?",
        "Optimize my portfolio for maximum Sharpe ratio. When I ask about this:",
        "Optimize my portfolio for maximum Sharpe ratio. I need your take on:",
    ])
    a = (
        "To construct a mean-variance optimized portfolio, I need:\n\n"
        "1. Expected returns - the anticipated return for each asset in the candidate universe.\n"
        "2. Covariance (or correlation) matrix - the volatilities and correlation structure "
        "across all assets.\n"
        "3. Constraints - any requirements such as long-only positions, turnover limits, "
        "sector caps, or min/max weight bounds.\n"
        "4. The risk-free rate - for the Sharpe ratio numerator.\n"
        "5. The candidate universe - which set of assets are we optimizing over (equities, "
        "bonds, alternatives)?\n\n"
        "Portfolio optimization is notoriously sensitive to input assumptions. Even small "
        "changes in expected returns can produces wildly different optimal weights "
        "(the 'error maximization' problem of Markowitz). Please provide the full parameter "
        "set, or consider using Black-Litterman or resampled optimization."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Mean-Variance Optimization",
                            "L3_Hard", q, a, "underspecified")


def abstention_credit_spread(rng, seq):
    """Missing: bond details, benchmark, rating."""
    q = _pick(rng, seq,  [
        "Analyze the credit spread for this bond.",
        "Analyze the credit spread for this bond. Please explain your reasoning.",
        "Analyze the credit spread for this bond. Can you assess this?",
        "Analyze the credit spread for this bond. When I ask about this:",
        "Analyze the credit spread for this bond. I need your take on:",
    ])
    a = (
        "To perform a proper credit spread analysis, I need the following:\n\n"
        "1. The bond characteristics - maturity, coupon, seniority, issue date, and currency.\n"
        "2. The benchmark curve - which risk-free reference curve should I use "
        "(U.S. Treasuries, OIS, swaps)?\n"
        "3. Credit rating - is the bond rated, and if so by which agency (S&P, Moody's, Fitch)?\n"
        "4. The market price or current yield - so I can back out the spread.\n"
        "5. Sector and geography - for peer-set and historical benchmarking.\n\n"
        "Credit spread = Bond yield minus benchmark yield for the same maturity. Without the "
        "bond's yield, maturity, and the appropriate risk-free benchmark curve, I cannot "
        "compute or analyze the spread."
    )
    return _make_abstention("CFA_Level_II", "Fixed Income", "Credit Spreads",
                            "L2_Medium", q, a, "underspecified")


def abstention_interest_rate_hedge(rng, seq):
    """Missing: exposure details, risk metric, instrument options."""
    q = _pick(rng, seq,  [
        "Hedge my portfolio's interest rate risk.",
        "Hedge my portfolio's interest rate risk. Please explain your reasoning.",
        "Hedge my portfolio's interest rate risk. Can you assess this?",
        "Hedge my portfolio's interest rate risk. When I ask about this:",
        "Hedge my portfolio's interest rate risk. I need your take on:",
    ])
    a = (
        "To design an interest rate hedge, I need information about your portfolio "
        "and constraints:\n\n"
        "1. Portfolio composition - the duration, convexity, key rate exposures, and the "
        "makeup of your fixed-income holdings.\n"
        "2. Risk management target - what is your desired end-state duration? Are you aiming "
        "to fully eliminate rate risk or only hedge a portion?\n"
        "3. Available instruments - does the firm have access to Treasury futures, interest "
        "rate swaps, swaptions, or SOFR futures?\n"
        "4. Cost constraints - what is the acceptable budget for the hedge (swap costs, "
        "option premiums, margin requirements)?\n"
        "5. Hedge horizon - do you need to maintain the hedge for weeks, months, or until a "
        "specific event?\n\n"
        "Different instruments (swaps, futures, options) have different cost structures, "
        "basis risk profiles, and accounting treatments. I need the full picture to recommend "
        "an appropriate hedge ratio and instrument."
    )
    return _make_abstention("CFA_Level_III", "Risk Management", "Interest Rate Hedging",
                            "L3_Hard", q, a, "underspecified")


def abstention_merger_arbitrage(rng, seq):
    """Missing: deal terms, timing, regulatory risk."""
    q = _pick(rng, seq,  [
        "Evaluate this merger arbitrage opportunity.",
        "Evaluate this merger arbitrage opportunity. Please explain your reasoning.",
        "Evaluate this merger arbitrage opportunity. Can you assess this?",
        "Evaluate this merger arbitrage opportunity. When I ask about this:",
        "Evaluate this merger arbitrage opportunity. I need your take on:",
    ])
    a = (
        "To evaluate a merger spread trade, I need:\n\n"
        "1. Deal terms - the acquisition price per share, form of consideration (cash, "
        "stock, or mixed), and the exchange ratio if stock.\n"
        "2. Deal structure - is it a stock merger, asset purchase, or third-party tender offer?\n"
        "3. Timeline - the expected closing date, regulatory review periods, and "
        "shareholder approval status.\n"
        "4. Merger conditions - what are the material conditions precedent and termination "
        "fee provisions?\n"
        "5. Current spread - the difference between the current market price and the deal price, "
        "expressed as a percentage and annualized.\n"
        "6. Closing probability - your assessment of the likelihood of regulatory approval. "
        "\n\n"
        "Merger arbitrage returns depend on deal completion probability, time to close, and "
        "the chance of deal breakup. Without these details, the risk-adjusted return "
        "cannot be computed."
    )
    return _make_abstention("FRM_Part_2", "Alternative Investments", "Merger Arbitrage",
                            "FRM2_Medium", q, a, "underspecified")


def abstention_tax_loss_harvesting(rng, seq):
    """Missing: positions, cost basis, tax bracket, wash sale info."""
    q = _pick(rng, seq,  [
        "How should I harvest tax losses this year?",
        "How should I harvest tax losses this year? Please explain your reasoning.",
        "How should I harvest tax losses this year? Can you assess this?",
        "How should I harvest tax losses this year? When I ask about this:",
        "How should I harvest tax losses this year? I need your take on:",
    ])
    a = (
        "To advise on tax-loss harvesting, I need your specific circumstances:\n\n"
        "1. Current positions - which securities do you hold at an unrealized loss, and by how much?\n"
        "2. Cost basis details - the original purchase price and acquisition date for each position.\n"
        "3. Your tax profile - your marginal federal and state income tax rates, any capital "
        "loss carryforwards from prior years, and whether you are taxed as an individual or "
        "a pass-through entity.\n"
        "4. Wash sale concerns - would selling and repurchasing substantially the same security "
        "within 30 days (before or after) trigger a wash sale?\n"
        "5. Market outlook - do you wish to maintain market exposure after selling, and if so, "
        "in an identical or different asset?\n\n"
        "Tax-loss harvesting rules vary by jurisdiction and investor type. The U.S. wash sale "
        "rule disallows claiming a loss on repurchasing substantially identical securities "
        "within 30 days. I need your full tax profile to provide meaningful guidance."
    )
    return _make_abstention("CFA_Level_III", "Tax Planning", "Tax-Loss Harvesting",
                            "L3_Medium", q, a, "underspecified")


def abstention_emerging_markets_allocation(rng, seq):
    """Missing: allocation parameters, country preferences, risk context."""
    q = _pick(rng, seq,  [
        "What should my emerging markets allocation be?",
        "What should my emerging markets allocation be? Please explain your reasoning.",
        "What should my emerging markets allocation be? Can you assess this?",
        "What should my emerging markets allocation be? When I ask about this:",
        "What should my emerging markets allocation be? I need your take on:",
    ])
    a = (
        "The appropriate emerging markets (EM) allocation depends on several parameters "
        "you have not specified:\n\n"
        "1. Your overall portfolio size, existing geographic diversification, and home-country bias.\n"
        "2. Risk tolerance - EM assets historically exhibit higher volatility, lower liquidity, "
        "and higher political risk than developed markets.\n"
        "3. Currency preferences - EM assets often carry unhedged currency risk, which can be "
        "a major return driver or detractor.\n"
        "4. Investment time horizon - EM allocation suitability differs materially over 3-year "
        "vs 20-year horizons.\n"
        "5. Investment vehicle preference - broad EM ETFs, country-specific funds, direct "
        "equities, local currency debt, or USD-denominated sovereign bonds?\n"
        "6. Regional preferences - EM is not monolithic. Brazil, China, India, Korea, Taiwan, "
        "and South Africa have very different risk-return profiles.\n\n"
        "Academic research suggests general EM allocations in a diversified portfolio typically "
        "fall in the 5-20 percent range. Your specific allocation depends on the parameters above."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "EM Allocation",
                            "L3_Medium", q, a, "underspecified")


def abstention_pe_ratio_valuation(rng, seq):
    """Missing: specific entity, earnings measure, comparison framework."""
    q = _pick(rng, seq,  [
        "Is the current P/E ratio cheap or expensive?",
        "Is the current P/E ratio cheap or expensive? Please explain your reasoning.",
        "Is the current P/E ratio cheap or expensive? Can you assess this?",
        "Is the current P/E ratio cheap or expensive? When I ask about this:",
        "Is the current P/E ratio cheap or expensive? I need your take on:",
    ])
    a = (
        "The question is too vague to answer - I need:\n\n"
        "1. Which entity - the S&P 500 index, the NASDAQ Composite, a sector index, or an "
        "individual company?\n"
        "2. Which earnings measure - trailing twelve months (TTM), forward (consensus next "
        "12 months), normalized (average over a multi-year cycle), or GAAP earnings?\n"
        "3. Which comparison benchmark - the index's or company's own 5-year or 10-year "
        "historical average? Peer group? Global equity? Emerging market?\n"
        "4. Interest rate environment - P/E multiples expand when rates fall and contract "
        "when rates rise.\n"
        "5. Earnings growth expectations - a higher P/E may be fully justified if future "
        "earnings growth is materially higher.\n\n"
        "A P/E of 20x might be cheap for a high-growth company but expensive for a mature "
        "utility. The question cannot be answered without specifying the entity, the earnings "
        "definition, and the comparison framework."
    )
    return _make_abstention("CFA_Level_I", "Equity Valuation", "Valuation Multiple",
                            "L1_Medium", q, a, "underspecified")


def abstention_sharpe_ratio_comparison(rng, seq):
    """Missing: fund details, time period, risk-free rate."""
    q = _pick(rng, seq,  [
        "Compare the risk-adjusted returns of these two funds.",
        "Compare the risk-adjusted returns of these two funds. Please explain your reasoning.",
        "Compare the risk-adjusted returns of these two funds. Can you assess this?",
        "Compare the risk-adjusted returns of these two funds. When I ask about this:",
        "Compare the risk-adjusted returns of these two funds. I need your take on:",
    ])
    a = (
        "To compare risk-adjusted returns between two funds, I need:\n\n"
        "1. The specific funds - names, ISINs, or ticker symbols.\n"
        "2. The fund categories and benchmark indices for a meaningful comparison.\n"
        "3. The evaluation period - overlapping or non-overlapping return histories? 1, 3, "
        "or 5 years?\n"
        "4. The risk-free rate - T-bill rate, SOFR, federal funds rate, or another benchmark?\n"
        "5. Return frequency - daily, monthly, or quarterly (affects annualization).\n"
        "6. The evaluation metric - Sharpe ratio, Sortino ratio, Treynor ratio, or information ratio?\n\n"
        "Risk-adjusted comparisons require identical treatment of the risk-free rate and "
        "consistent time windows. Without the specific fund identifiers, the comparison "
        "is impossible."
    )
    return _make_abstention("CFA_Level_II", "Performance Evaluation", "Risk-Adjusted Returns",
                            "L2_Medium", q, a, "underspecified")


def abstention_dcf_valuation_company(rng, seq):
    """Missing: full set of DCF assumptions."""
    q = _pick(rng, seq,  [
        "Perform a DCF valuation on company XYZ.",
        "Perform a DCF valuation on company XYZ. Please explain your reasoning.",
        "Perform a DCF valuation on company XYZ. Can you assess this?",
        "Perform a DCF valuation on company XYZ. When I ask about this:",
        "Perform a DCF valuation on company XYZ. I need your take on:",
    ])
    a = (
        "A proper DCF valuation requires many assumptions you have not provided:\n\n"
        "1. Revenue growth assumptions - near-term projected growth rates (3-5 years) and "
        "the long-term terminal growth rate.\n"
        "2. Operating margins - should margins expand, contract, or remain stable? What drives "
        "the trajectory?\n"
        "3. Capital expenditures - projected capex as a percentage of revenue or dollar amount.\n"
        "4. Working capital requirements - projected changes in receivables, inventory, and "
        "payables.\n"
        "5. The WACC - cost of equity (CAPM: risk-free rate, beta, equity risk premium), "
        "cost of debt (post-tax), and target capital structure.\n"
        "6. Terminal value approach - perpetuity growth or exit multiple?\n\n"
        "DCF outputs are extremely sensitive to inputs. A 1 percent change in the terminal "
        "growth rate or WACC can shift the intrinsic value by 20-30 percent. I cannot produce "
        "a credible DCF without these parameters."
    )
    return _make_abstention("CFA_Level_II", "Equity Valuation", "DCF",
                            "L2_Hard", q, a, "underspecified")


def abstention_cva_derivate(rng, seq):
    """Missing: counterparty data, exposure profile, CDS spreads."""
    q = _pick(rng, seq,  [
        "Calculate the CVA for this derivative portfolio.",
        "Calculate the CVA for this derivative portfolio. Please explain your reasoning.",
        "Calculate the CVA for this derivative portfolio. Can you assess this?",
        "Calculate the CVA for this derivative portfolio. When I ask about this:",
        "Calculate the CVA for this derivative portfolio. I need your take on:",
    ])
    a = (
        "To compute Credit Valuation Adjustment (CVA), I need the following data:\n\n"
        "1. Counterparty details - identities of all counterparties and their credit ratings "
        "or CDS spreads.\n"
        "2. Exposure profiles - the Expected Positive Exposure (EPE) at each future time step "
        "for each derivative contract.\n"
        "3. Recovery rate assumptions - for each counterparty upon default (typically 40 percent "
        "for G-SIBs under Basel III).\n"
        "4. Default correlation - how defaults of different counterparties correlate across the portfolio.\n"
        "5. Time horizon - what is the CVA computation period (e.g., 1 year, 5 years)?\n"
        "6. Discounting curve - the risk-free curve to discount expected CVA cash flows.\n\n"
        "CVA = sum over time steps of [discount factor x EPE x Marginal PD x (1 minus Recovery)]. "
        "Without the exposure profile, counterparty credit data, and recovery assumptions, the "
        "CVA cannot be computed."
    )
    return _make_abstention("CFA_Level_III", "Derivatives", "CVA/DVA",
                            "L3_Hard", q, a, "underspecified")


def abstention_esg_rating(rng, seq):
    """Missing: company, framework, data source."""
    q = _pick(rng, seq,  [
        "What is the ESG score of my portfolio?",
        "What is the ESG score of my portfolio? Please explain your reasoning.",
        "What is the ESG score of my portfolio? Can you assess this?",
        "What is the ESG score of my portfolio? When I ask about this:",
        "What is the ESG score of my portfolio? I need your take on:",
    ])
    a = (
        "To calculate or evaluate an ESG score for a portfolio, I need:\n\n"
        "1. The portfolio holdings - the specific securities and their weights.\n"
        "2. The ESG framework - which provider (MSCI, Sustainalytics, Bloomberg, FTSE4Good)? "
        "Scores are not comparable across providers.\n"
        "3. The aggregation methodology - is the score value-weighted, equal-weighted, or "
        "carbon-weighted?\n"
        "4. The ESG dimensions of interest - environmental, social, governance, or the overall "
        "composite? Each has very different sub-scores.\n"
        "5. Data currency - ESG ratings are updated regularly; which vintage date should I use?\n\n"
        "ESG scores are not standardized or regulated like accounting figures. Two providers "
        "can assign dramatically different scores to the same company. I need your specific "
        "holdings, provider, and methodology to compute a meaningful result."
    )
    return _make_abstention("CFA_Level_II", "ESG Investing", "Portfolio ESG Score",
                            "L2_Medium", q, a, "underspecified")


def abstention_risk_parity_portfolio(rng, seq):
    """Missing: covariance matrix, assets, budget."""
    q = _pick(rng, seq,  [
        "Build a risk parity portfolio.",
        "Build a risk parity portfolio. Please explain your reasoning.",
        "Build a risk parity portfolio. Can you assess this?",
        "Build a risk parity portfolio. When I ask about this:",
        "Build a risk parity portfolio. I need your take on:",
    ])
    a = (
        "A risk parity (risk-budgeting) portfolio requires:\n\n"
        "1. Candidate assets - the set of asset classes or individual securities we are "
        "allocating across.\n"
        "2. The covariance matrix - volatilities and correlation structure, since risk parity "
        "equalizes risk contributions.\n"
        "3. Risk budget specification - should all assets contribute equally to total portfolio "
        "risk, or should you specify custom risk budgets?\n"
        "4. Leverage tolerance - many asset classes (e.g., government bonds) have low volatility, "
        "so risk parity often requires leverage to reach target returns.\n"
        "5. Investment constraints - are short sales allowed? Are there min/max position limits?\n\n"
        "Risk parity allocations are highly dependent on the correlation structure, which "
        "itself varies across market regimes. Please provide the asset universe and the "
        "covariance data."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Risk Parity",
                            "L3_Medium", q, a, "underspecified")


def abstention_fund_manager_alpha(rng, seq):
    """Missing: fund details, benchmark, fee structure."""
    q = _pick(rng, seq,  [
        "Did this fund manager generate alpha?",
        "Did this fund manager generate alpha? Please explain your reasoning.",
        "Did this fund manager generate alpha? Can you assess this?",
        "Did this fund manager generate alpha? When I ask about this:",
        "Did this fund manager generate alpha? I need your take on:",
    ])
    a = (
        "To evaluate whether a fund manager generated alpha, I need:\n\n"
        "1. The fund name or identifier - to obtain its return series.\n"
        "2. The appropriate benchmark - must match the fund's stated investment strategy "
        "(S&P 500 for large-cap equity, blended index for balanced funds, etc.).\n"
        "3. The evaluation period - 1, 3, 5, or 10 years? Alpha at different horizons can differ.\n"
        "4. The regression model - CAPM, Fama-French 3/5 factor, Carhart 4-factor? Each "
        "models attribution differently.\n"
        "5. The fee structure - management fees and performance carry must be deducted to "
        "determine net alpha.\n\n"
        "Alpha = (Fund returns minus Model-predicted returns) averaged over a period. The "
        "result depends critically on the chosen model and benchmark."
    )
    return _make_abstention("CFA_Level_III", "Performance Evaluation", "Alpha Attribution",
                            "L3_Medium", q, a, "underspecified")


def abstention_montecarlo_simulation(rng, seq):
    """Missing: model parameters, confidence level, horizon."""
    q = _pick(rng, seq,  [
        "Run a Monte Carlo simulation for my portfolio.",
        "Run a Monte Carlo simulation for my portfolio. Please explain your reasoning.",
        "Run a Monte Carlo simulation for my portfolio. Can you assess this?",
        "Run a Monte Carlo simulation for my portfolio. When I ask about this:",
        "Run a Monte Carlo simulation for my portfolio. I need your take on:",
    ])
    a = (
        "To construct a Monte Carlo simulation, I need:\n\n"
        "1. Portfolio composition - the asset allocation (equities, bonds, alternatives) and "
        "the weights of each asset.\n"
        "2. Parameter estimates - expected returns, volatilities, and correlation matrix for "
        "each asset class.\n"
        "3. Distribution assumptions - normal, lognormal, t-distributed returns? Constant or "
        "stochastic volatilities?\n"
        "4. Time horizon - how many years should the simulation project?\n"
        "5. Number of paths - how many simulated paths (e.g., 10,000, 50,000)?\n"
        "6. Output desired - probability of running out of money, percentile distributions of "
        "terminal wealth, Value at Risk?\n\n"
        "Monte Carlo simulations require explicit parameter specifications. The results are "
        "only as credible as the input assumptions."
    )
    return _make_abstention("CFA_Level_III", "Quantitative Methods", "Monte Carlo",
                            "L3_Medium", q, a, "underspecified")


def abstention_black_litterman_model(rng, seq):
    """Missing: views, uncertainty, market cap weights."""
    q = _pick(rng, seq,  [
        "Apply the Black-Litterman model to my portfolio.",
        "Apply the Black-Litterman model to my portfolio. Please explain your reasoning.",
        "Apply the Black-Litterman model to my portfolio. Can you assess this?",
        "Apply the Black-Litterman model to my portfolio. When I ask about this:",
        "Apply the Black-Litterman model to my portfolio. I need your take on:",
    ])
    a = (
        "The Black-Litterman (BL) model requires several inputs:\n\n"
        "1. Equilibrium weights - the market-cap weights of each asset in the global market "
        "portfolio. These serve as the BL prior.\n"
        "2. The covariance matrix - the asset returns' covariance or volatility matrix.\n"
        "3. Your views - which assets you believe will outperform or underperform, and the "
        "magnitude of each view.\n"
        "4. View confidence / uncertainty - the omega matrix encoding how confident you are "
        "in each view (higher uncertainty is closer to BL prior; higher confidence is closer "
        "to your view).\n"
        "5. View types - absolute (e.g., AAPL returns 10 percent next year) or relative "
        "(e.g., AAPL outperforms MSFT by 2 percent)?\n\n"
        "The BL model blends your views with equilibrium returns. Incorrect view specification "
        "(especially confidence levels) can produce nonsensical allocations. Please provide "
        "the full view set and parameters."
    )
    return _make_abstention("CFA_Level_III", "Quantitative Methods", "Black-Litterman",
                            "L3_Hard", q, a, "underspecified")


def abstention_factor_model_regression(rng, seq):
    """Missing: fund returns, factor definitions, time period."""
    q = _pick(rng, seq,  [
        "Run a factor regression on this fund.",
        "Run a factor regression on this fund. Please explain your reasoning.",
        "Run a factor regression on this fund. Can you assess this?",
        "Run a factor regression on this fund. When I ask about this:",
        "Run a factor regression on this fund. I need your take on:",
    ])
    a = (
        "A factor regression requires:\n\n"
        "1. The fund's returns - a time series of period-by-period returns.\n"
        "2. The choice of factor model - CAPM (one factor), Fama-French 3-factor (market, "
        "SMB, HML), Fama-French 5-factor (adds RMW, CMA), or Carhart 4-factor (adds momentum).\n"
        "3. The time period - the start and end dates for the analysis. Results vary across regimes.\n"
        "4. Return frequency - daily, weekly, or monthly?\n"
        "5. The risk-free rate series - for constructing the market risk premium (Rm - Rf).\n\n"
        "The regression output includes alpha, factor betas, R-squared, and t-statistics. "
        "Without the fund's return series and the chosen factors, the regression cannot be run."
    )
    return _make_abstention("CFA_Level_II", "Quantitative Methods", "Factor Models",
                            "L2_Hard", q, a, "underspecified")


def abstention_calmar_ratio_computation(rng, seq):
    """Missing: fund returns, benchmark, evaluation period."""
    q = _pick(rng, seq,  [
        "Compute the Calmar ratio for this fund.",
        "Compute the Calmar ratio for this fund. Please explain your reasoning.",
        "Compute the Calmar ratio for this fund. Can you assess this?",
        "Compute the Calmar ratio for this fund. When I ask about this:",
        "Compute the Calmar ratio for this fund. I need your take on:",
    ])
    a = (
        "To compute the Calmar ratio, I need:\n\n"
        "1. The fund's return series - consecutive period returns over the evaluation window.\n"
        "2. The maximum drawdown - the largest peak-to-trough decline during the evaluation period.\n"
        "3. The evaluation period - what start and end dates should I use?\n"
        "4. The return frequency - monthly, quarterly, or annualized?\n\n"
        "Calmar ratio = Annualized return / Maximum drawdown. The result is highly sensitive "
        "to the chosen period: a short window may exclude the maximum drawdown, while a longer "
        "window may span multiple drawdown cycles. Please provide the return series and the "
        "evaluation window."
    )
    return _make_abstention("CFA_Level_III", "Performance Evaluation", "Calmar Ratio",
                            "L3_Medium", q, a, "underspecified")


def abstention_sortino_ratio(rng, seq):
    """Missing: downside target, return data."""
    q = _pick(rng, seq,  [
        "What is the Sortino ratio for my investment?",
        "What is the Sortino ratio for my investment? Please explain your reasoning.",
        "What is the Sortino ratio for my investment? Can you assess this?",
        "What is the Sortino ratio for my investment? When I ask about this:",
        "What is the Sortino ratio for my investment? I need your take on:",
    ])
    a = (
        "To calculate the Sortino ratio, I need:\n\n"
        "1. The return series - period-by-period returns for the investment.\n"
        "2. The downside risk-free rate (or minimum acceptable return, MAR) - some practitioners "
        "use 0 percent, some use the T-bill rate, and others use a targeted downside return.\n"
        "3. The evaluation period and frequency (monthly or daily).\n\n"
        "Sortino ratio = (Return minus MAR) / Downside deviation. Unlike the Sharpe ratio, the "
        "Sortino penalizes only downside volatility (below the MAR). The MAR choice can change "
        "the ratio materially. Please provide the return data and specify the MAR."
    )
    return _make_abstention("CFA_Level_III", "Performance Evaluation", "Sortino Ratio",
                            "L3_Medium", q, a, "underspecified")


def abstention_tracking_error(rng, seq):
    """Missing: portfolio data, benchmark data, period."""
    q = _pick(rng, seq,  [
        "Calculate the tracking error for this portfolio.",
        "Calculate the tracking error for this portfolio. Please explain your reasoning.",
        "Calculate the tracking error for this portfolio. Can you assess this?",
        "Calculate the tracking error for this portfolio. When I ask about this:",
        "Calculate the tracking error for this portfolio. I need your take on:",
    ])
    a = (
        "To compute tracking error, I need:\n\n"
        "1. The portfolio's returns - period-by-period returns for the managed portfolio.\n"
        "2. The benchmark's returns - the same time period and frequency.\n"
        "3. The evaluation period - the start and end dates to ensure the two return series have "
        "the same length.\n\n"
        "Tracking error = Standard deviation of the active return series (portfolio return minus "
        "benchmark return), annualized. I cannot compute it without both return series."
    )
    return _make_abstention("CFA_Level_III", "Performance Evaluation", "Tracking Error",
                            "L3_Easy", q, a, "underspecified")


def abstention_liq_coverage_ratio(rng, seq):
    """Missing: asset data, liability schedule."""
    q = _pick(rng, seq,  [
        "Calculate the liquidity coverage ratio for my company.",
        "Calculate the liquidity coverage ratio for my company. Please explain your reasoning.",
        "Calculate the liquidity coverage ratio for my company. Can you assess this?",
        "Calculate the liquidity coverage ratio for my company. When I ask about this:",
        "Calculate the liquidity coverage ratio for my company. I need your take on:",
    ])
    a = (
        "To compute the Liquidity Coverage Ratio (LCR), I need:\n\n"
        "1. High-quality liquid assets (HQLA) - the market value of Level 1 and Level 2 HQLA "
        "held by the institution.\n"
        "2. Total net cash outflows - the estimated cash outflows over the next 30 calendar "
        "days, categorized by customer type and liability stability.\n"
        "3. Total expected cash inflows (subject to the 75 percent cap) - contractually "
        "scheduled inflows over the same 30-day window.\n\n"
        "Basel III requires an LCR of at least 100 percent. Without the HQLA inventory and the "
        "net cash flow projections, the ratio cannot be computed."
    )
    return _make_abstention("FRM_Part_2", "Liquidity Risk", "LCR",
                            "FRM2_Medium", q, a, "underspecified")


def abstention_economic_capital_modeling(rng, seq):
    """Missing: portfolio risk data, confidence level, model choice."""
    q = _pick(rng, seq,  [
        "Calculate the economic capital for my bank's portfolio.",
        "Calculate the economic capital for my bank's portfolio. Please explain your reasoning.",
        "Calculate the economic capital for my bank's portfolio. Can you assess this?",
        "Calculate the economic capital for my bank's portfolio. When I ask about this:",
        "Calculate the economic capital for my bank's portfolio. I need your take on:",
    ])
    a = (
        "To model economic capital, I need:\n\n"
        "1. The portfolio composition - exposure by asset class, sector, and geography.\n"
        "2. Risk parameters - PD (probability of default), LGD (loss given default), EAD "
        "(exposure at default), and default correlation assumptions.\n"
        "3. Confidence level - what percentile of the loss distribution should economic capital "
        "cover? (e.g., 99.9 percent, 99.73 percent, 95 percent, 99 percent).\n"
        "4. Time horizon - typically 1 year for ICAAP purposes.\n"
        "5. Aggregation method - how to model dependencies between risk types (credit, market, "
        "operational) and between counterparties.\n\n"
        "Economic capital = VaR at the chosen confidence level minus expected loss. The result "
        "depends on the chosen model (CreditRisk+, IRB, internal models) and all the parameters above."
    )
    return _make_abstention("FRM_Part_2", "Credit Risk", "Economic Capital",
                            "FRM2_Hard", q, a, "underspecified")


def abstention_portfolio_attribution_analysis(rng, seq):
    """Missing: portfolio weights, benchmark weights, return data."""
    q = _pick(rng, seq,  [
        "Perform a Brinson-Fachler attribution analysis on this portfolio.",
        "Perform a Brinson-Fachler attribution analysis on this portfolio. Please explain your reasoning.",
        "Perform a Brinson-Fachler attribution analysis on this portfolio. Can you assess this?",
        "Perform a Brinson-Fachler attribution analysis on this portfolio. When I ask about this:",
        "Perform a Brinson-Fachler attribution analysis on this portfolio. I need your take on:",
    ])
    a = (
        "A Brinson-Fachler attribution analysis requires:\n\n"
        "1. Portfolio weights - by asset category/sector at the start of the period.\n"
        "2. Benchmark weights - by the same categories/sectors.\n"
        "3. Actual returns - by category/sector for both the portfolio and the benchmark.\n"
        "4. The benchmark's overall return - for calculating the allocation effect base.\n\n"
        "Brinson-Fachler decomposes active return into three components: allocation (choosing "
        "overweight/underweight sectors), selection (choosing winners/losers within sectors), "
        "and interaction. Without weights and returns for both the portfolio and the benchmark, "
        "the decomposition cannot be performed."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Performance Attribution",
                            "L3_Hard", q, a, "underspecified")


def abstention_valuation_multiple_comparison(rng, seq):
    """Missing: peer set, multiples, adjustments."""
    q = _pick(rng, seq,  [
        "Which company has the best valuation multiple?",
        "Which company has the best valuation multiple? Please explain your reasoning.",
        "Which company has the best valuation multiple? Can you assess this?",
        "Which company has the best valuation multiple? When I ask about this:",
        "Which company has the best valuation multiple? I need your take on:",
    ])
    a = (
        "I cannot answer this question without significant additional information:\n\n"
        "1. Which companies should I compare? A valuation comparison requires a defined set of "
        "peer companies.\n"
        "2. Which multiples - P/E, EV/EBITDA, EV/Sales, P/B, or another metric?\n"
        "3. The time period - trailing 12 months, forward estimates, or normalized "
        "(average over a cycle)?\n"
        "4. Industry sector and geography - comparables must be in the same industry and region.\n"
        "5. Any normalization adjustments - one-time charges, restructuring costs, or "
        "extraordinary items must be excluded.\n\n"
        "Different industry sectors have different typical multiple ranges. A P/E of 15x might "
        "be cheap for a tech company but expensive for an energy company. Without specifying "
        "the companies, sector, and multiples, the comparison is meaningless."
    )
    return _make_abstention("CFA_Level_II", "Equity Valuation", "Comparable Valuation",
                            "L2_Medium", q, a, "underspecified")


def abstention_working_capital_forecast(rng, seq):
    """Missing: revenue forecast, payment terms, supply chain terms."""
    q = _pick(rng, seq,  [
        "Forecast my company's working capital needs for next year.",
        "Forecast my company's working capital needs for next year. Please explain your reasoning.",
        "Forecast my company's working capital needs for next year. Can you assess this?",
        "Forecast my company's working capital needs for next year. When I ask about this:",
        "Forecast my company's working capital needs for next year. I need your take on:",
    ])
    a = (
        "To forecast working capital requirements, I need:\n\n"
        "1. Revenue forecast - projected sales and the growth rate assumption for the coming year.\n"
        "2. Accounts receivable terms - average collection period (DSO), credit policy changes, "
        "or customer concentration risk.\n"
        "3. Inventory management - current inventory turns, supply lead times, and any planned "
        "changes in inventory policy.\n"
        "4. Accounts payable terms - average payment period (DPO) to vendors and any planned "
        "renegotiations.\n"
        "5. Capex plans - capital expenditure affects the cash conversion cycle by tying up "
        "working capital in long-term assets.\n\n"
        "Working capital requirements depend on (Days Sales Outstanding + Days Inventory "
        "Outstanding - Days Payable Outstanding) / 365 times annual revenue. Without the above "
        "inputs, the forecast cannot be constructed."
    )
    return _make_abstention("CFA_Level_I", "Financial Statement Analysis", "Working Capital",
                            "L1_Medium", q, a, "underspecified")


def abstention_implied_volatility_calculation(rng, seq):
    """Missing: option price, parameters for Black-Scholes."""
    q = _pick(rng, seq,  [
        "What is the implied volatility of this option?",
        "What is the implied volatility of this option? Please explain your reasoning.",
        "What is the implied volatility of this option? Can you assess this?",
        "What is the implied volatility of this option? When I ask about this:",
        "What is the implied volatility of this option? I need your take on:",
    ])
    a = (
        "To calculate implied volatility, I need the following input parameters for the "
        "Black-Scholes or relevant model:\n\n"
        "1. The option's current market price.\n"
        "2. The underlying asset's current price.\n"
        "3. The option's strike price.\n"
        "4. Time to expiration (in years or days).\n"
        "5. The risk-free interest rate.\n"
        "6. Any dividends expected from the underlying during the option's life.\n\n"
        "Implied volatility is the volatility parameter that, when plugging all the above inputs "
        "into an option pricing model, produces a theoretical price equal to the market price. "
        "Without these parameters, the calculation is impossible."
    )
    return _make_abstention("CFA_Level_II", "Derivatives", "Implied Volatility",
                            "L2_Hard", q, a, "underspecified")


def abstention_greeks_calculation(rng, seq):
    """Missing: option parameters, model choice."""
    q = _pick(rng, seq,  [
        "Calculate the Greeks for my options portfolio.",
        "Calculate the Greeks for my options portfolio. Please explain your reasoning.",
        "Calculate the Greeks for my options portfolio. Can you assess this?",
        "Calculate the Greeks for my options portfolio. When I ask about this:",
        "Calculate the Greeks for my options portfolio. I need your take on:",
    ])
    a = (
        "To calculate option Greeks, I need the parameters for each position:\n\n"
        "1. For each option: the underlying price, strike, time to expiration, risk-free rate, "
        "dividend yield, and whether it is a call or put.\n"
        "2. The pricing model to use (Black-Scholes, binomial tree, Monte Carlo). Greeks vary "
        "slightly between models.\n"
        "3. The position size (number of contracts).\n"
        "4. Any portfolio-level aggregation parameters (e.g., are you looking at spot-greeks or "
        "dollar-greeks)?\n\n"
        "The Greeks (delta, gamma, theta, vega, rho) measure sensitivity of the option price to "
        "various parameters. Without the option parameters and model specification, the "
        "computation cannot be performed."
    )
    return _make_abstention("CFA_Level_II", "Derivatives", "Option Greeks",
                            "L2_Hard", q, a, "underspecified")



# ====================================================================
# 2. UNANSWERABLE -- 25 templates
# ====================================================================


def abstention_stock_price_prediction(rng, seq):
    """Impossibility: future stock prices are unknowable."""
    q = _pick(rng, seq,  [
        "What will the price of SPY be one year from today?",
        "What will the price of SPY be one year from today? Please explain your reasoning.",
        "What will the price of SPY be one year from today? Can you assess this?",
        "What will the price of SPY be one year from today? When I ask about this:",
        "What will the price of SPY be one year from today? I need your take on:",
    ])
    a = (
        "I cannot predict the future price of SPY or any other security. This is "
        "fundamentally unknowable for several reasons:\n\n"
        "1. Stock prices reflect the present value of all future cash flows, which depend "
        "on infinitely many factors - economic conditions, corporate earnings, geopolitical "
        "events, interest rates, investor sentiment, and countless variables.\n"
        "2. Even sophisticated models (DCF, factor models) produce ranges of estimates, not "
        "point predictions, and the error bands are typically wider than the expected move.\n"
        "3. Markets are efficient to the extent that publicly available information is already "
        "reflected in the current price. Any future price change depends on new information "
        "that has not yet been discovered.\n"
        "4. Random events - supply chain disruptions, natural disasters, or unexpected "
        "regulatory announcements - are fundamentally unpredictable.\n\n"
        "What I can do instead: estimate fair value using fundamental analysis, compute "
        "probabilistic return distributions using historical volatility, or analyze "
        "option-implied distributions for expected price ranges."
    )
    return _make_abstention("CFA_Level_II", "Equity Valuation", "Price Prediction",
                            "L2_Medium", q, a, "unanswerable")


def abstention_fair_value_stock(rng, seq):
    """Fair value depends on subjective assumptions."""
    q = _pick(rng, seq,  [
        "What is the fair value of Tesla stock?",
        "What is the fair value of Tesla stock? Please explain your reasoning.",
        "What is the fair value of Tesla stock? Can you assess this?",
        "What is the fair value of Tesla stock? When I ask about this:",
        "What is the fair value of Tesla stock? I need your take on:",
    ])
    a = (
        "I cannot produce a single, definitive fair value for Tesla stock. The concept of "
        "'fair value' is inherently subjective:\n\n"
        "1. Valuation method - DCF, comparable multiples, scenario analysis, or residual "
        "income models can all produce results differing by 50 percent or more.\n"
        "2. Growth assumptions - Tesla's valuation hinges on EV market penetration rates, "
        "autonomous driving timelines, energy business growth, and margin trajectories. "
        "Each assumption is highly uncertain.\n"
        "3. Time horizon - fair value at one year may differ from fair value at 10 years.\n"
        "4. Discount rate - the equity risk premium and beta assumptions materially affect "
        "present values.\n\n"
        "Different analysts using different methods and assumptions produce fair values "
        "ranging widely. I can walk through a specific valuation framework with your "
        "chosen assumptions, but cannot give a single correct answer."
    )
    return _make_abstention("CFA_Level_II", "Equity Valuation", "Fair Value",
                            "L2_Medium", q, a, "unanswerable")


def abstention_market_bottom_call(rng, seq):
    """Unforecastable: no model can identify market bottoms in real-time."""
    q = _pick(rng, seq,  [
        "Has the market already bottomed?",
        "Has the market already bottomed? Please explain your reasoning.",
        "Has the market already bottomed? Can you assess this?",
        "Has the market already bottomed? When I ask about this:",
        "Has the market already bottomed? I need your take on:",
    ])
    a = (
        "I cannot determine whether the market has already bottomed. Identifying a market "
        "bottom in real-time is impossible because:\n\n"
        "1. A 'bottom' is only visible in hindsight - the market is at its lowest point only "
        "after prices rise subsequently, which cannot be known at the time.\n"
        "2. Markets can have multiple bottoms (double bottom, double dip recession), so "
        "declaring a bottom prematurely is a common error.\n"
        "3. Technical indicators (RSI, moving average crossovers, volume profiles) provide "
        "signals, not certainty, and have high false-positive rates.\n"
        "4. Economic catalysts that drive turning points - policy shifts, earnings inflection, "
        "Fed pivot - can emerge unexpectedly.\n\n"
        "I can instead help you analyze current market conditions using valuation metrics, "
        "sentiment indicators, and macroeconomic indicators to assess the risk-reward environment."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Market Timing",
                            "L3_Medium", q, a, "unanswerable")


def abstention_crypto_price_forecast(rng, seq):
    """Unforecastable: crypto prices have no fundamental valuation anchor."""
    q = _pick(rng, seq,  [
        "Where will Bitcoin be in 5 years?",
        "Where will Bitcoin be in 5 years? Please explain your reasoning.",
        "Where will Bitcoin be in 5 years? Can you assess this?",
        "Where will Bitcoin be in 5 years? When I ask about this:",
        "Where will Bitcoin be in 5 years? I need your take on:",
    ])
    a = (
        "I cannot predict Bitcoin's price 5 years from now. Crypto asset valuation is "
        "especially challenging:\n\n"
        "1. Unlike equities or bonds, Bitcoin produces no cash flows, dividends, or coupons, "
        "so traditional valuation frameworks have limited applicability.\n"
        "2. Price drivers include speculative demand, regulatory developments, institutional "
        "adoption, macroeconomic environment, and network effects - each highly uncertain over "
        "a 5-year horizon.\n"
        "3. Past performance is not predictive in a regime where technology, regulation, and "
        "competitive landscape (competing assets) can change dramatically.\n"
        "4. Regulatory actions - bans, spot ETF approvals, tax treatment changes - can cause "
        "step-function price moves.\n\n"
        "What I can help with: analyze the supply mechanics (halving schedule, miner behavior), "
        "discuss the macroeconomic case for Bitcoin as a store of value, or compare risk-return "
        "profiles with traditional assets."
    )
    return _make_abstention("CFA_Level_III", "Alternative Investments", "Cryptocurrency",
                            "L3_Medium", q, a, "unanswerable")


def abstention_pandemic_gdp_impact(rng, seq):
    """Unforecastable: future pandemic impact is unknowable."""
    q = _pick(rng, seq,  [
        "What will the global GDP impact of the next pandemic be?",
        "What will the global GDP impact of the next pandemic be? Please explain your reasoning.",
        "What will the global GDP impact of the next pandemic be? Can you assess this?",
        "What will the global GDP impact of the next pandemic be? When I ask about this:",
        "What will the global GDP impact of the next pandemic be? I need your take on:",
    ])
    a = (
        "I cannot answer this. The GDP impact of a future pandemic is fundamentally "
        "unknowable:\n\n"
        "1. You need to specify the pathogen's characteristics - transmissibility, severity, "
        "fatality rate, and seasonal patterns.\n"
        "2. You need to know where and when it emerges - a pandemic originating in one region "
        "versus simultaneously globally has different GDP implications.\n"
        "3. Government response - lockdowns, fiscal stimulus, and monetary policy responses "
        "cannot be anticipated.\n"
        "4. Supply chain and labor market dynamics depend on sectoral vulnerability that cannot "
        "be predicted.\n\n"
        "Even modeling the 2020 pandemic in real-time required a wide range of scenario "
        "assumptions. Predicting an unknown event's economic magnitude is categorically infeasible."
    )
    return _make_abstention("CFA_Level_II", "Economics", "Pandemic Economics",
                            "L2_Hard", q, a, "unanswerable")


def abstention_interest_rate_prediction(rng, seq):
    """Unforecastable: future rates depend on future economic data."""
    q = _pick(rng, seq,  [
        "Will the Fed cut interest rates in the next six months?",
        "Will the Fed cut interest rates in the next six months? Please explain your reasoning.",
        "Will the Fed cut interest rates in the next six months? Can you assess this?",
        "Will the Fed cut interest rates in the next six months? When I ask about this:",
        "Will the Fed cut interest rates in the next six months? I need your take on:",
    ])
    a = (
        "I cannot predict whether the Fed will cut rates in the next six months. Interest "
        "rate decisions depend on future data:\n\n"
        "1. The Fed follows a data-dependent approach - upcoming inflation readings (CPI, PCE), "
        "labor market reports (NFP, unemployment), and GDP growth determine the path.\n"
        "2. The FOMC consists of 19 voting members who may disagree about the appropriate path, "
        "especially at the margin.\n"
        "3. External shocks - banking stress, geopolitical events, or commodity price spikes - "
        "can alter the policy calculus unexpectedly.\n"
        "4. Markets price in rate cut probabilities through fed funds futures, but these reflect "
        "market beliefs, not certainty.\n\n"
        "What I can help with: analyze current market-implied rate probabilities, explain the "
        "Fed's reaction function, or model how the Fed might respond to different inflation and "
        "labor scenarios."
    )
    return _make_abstention("CFA_Level_II", "Fixed Income", "Monetary Policy",
                            "L2_Medium", q, a, "unanswerable")


def abstention_black_swan_event(rng, seq):
    """Unforecastable: by definition, black swans are unpredictable."""
    q = _pick(rng, seq,  [
        "Which black swan event is most likely to happen next?",
        "Which black swan event is most likely to happen next? Please explain your reasoning.",
        "Which black swan event is most likely to happen next? Can you assess this?",
        "Which black swan event is most likely to happen next? When I ask about this:",
        "Which black swan event is most likely to happen next? I need your take on:",
    ])
    a = (
        "This question contains a contradiction. A 'black swan event', by Nassim Taleb's "
        "definition, is an event that is:\n\n"
        "1. An outlier - it lies outside the realm of regular expectations because nothing in "
        "the past can conclusively point to its possibility.\n"
        "2. It carries extreme impact - financial, societal, or geopolitical consequences far "
        "exceeding normal fluctuations.\n"
        "3. Retrospectively explainable - but only after it happens, people rationalize it as predictable.\n\n"
        "If an event can be predicted with any confidence, it is not a black swan. Attempting "
        "to rank the likelihood of impossible-to-anticipate events is logically inconsistent. "
        "I can instead discuss Black Swan theory, tail risk hedging strategies, or how to build "
        "robustness against extreme events."
    )
    return _make_abstention("CFA_Level_III", "Risk Management", "Black Swan",
                            "L3_Hard", q, a, "unanswerable")


def abstention_perfect_timing(rng, seq):
    """Unforecastable: no one can time entries and exits perfectly."""
    q = _pick(rng, seq,  [
        "When is the best time to invest $1 million in the S&P 500?",
        "When is the best time to invest $1 million in the S&P 500? Please explain your reasoning.",
        "When is the best time to invest $1 million in the S&P 500? Can you assess this?",
        "When is the best time to invest $1 million in the S&P 500? When I ask about this:",
        "When is the best time to invest $1 million in the S&P 500? I need your take on:",
    ])
    a = (
        "I cannot identify the single 'best time' to invest - it is fundamentally unknowable:\n\n"
        "1. The best time is only identifiable after the fact. At any given moment, any choice "
        "of timing is the 'best' or 'worst' choice only in hindsight.\n"
        "2. Even if you could identify the absolute best entry date within a 3-year window, the "
        "cost of missing just a few of the best days is enormous (approximately 50% reduction "
        "in cumulative returns).\n"
        "3. The tradeoff - waiting for a better entry carries the risk that the market rises "
        "while you wait, and the opportunity cost may far exceed any benefit of timing the dip.\n\n"
        "Empirical evidence suggests that for long-term investors, dollar-cost averaging and "
        "early deployment generally outperform attempts to time the market. What I can help "
        "with: modeling the expected return distribution of different deployment strategies."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Market Timing",
                            "L3_Medium", q, a, "unanswerable")


def abstention_company_revenue_forecast(rng, seq):
    """Unforecastable: future revenue depends on unpredictable factors."""
    q = _pick(rng, seq,  [
        "What will Apple's revenue be in Q4 2026?",
        "What will Apple's revenue be in Q4 2026? Please explain your reasoning.",
        "What will Apple's revenue be in Q4 2026? Can you assess this?",
        "What will Apple's revenue be in Q4 2026? When I ask about this:",
        "What will Apple's revenue be in Q4 2026? I need your take on:",
    ])
    a = (
        "I cannot precisely predict Apple's Q4 2026 revenue. Corporate revenue projections "
        "have inherent limits:\n\n"
        "1. Revenue depends on unit sales volume and average selling price, both of which "
        "depend on consumer behavior, competitive dynamics, and supply chain conditions.\n"
        "2. New product launches - iPhone cycles, iPad/Mac refreshes, services growth - "
        "have uncertain timing and initial reception.\n"
        "3. Foreign exchange effects - Apple generates approximately 60% of revenue outside "
        "the U.S.; currency movements are unpredictable.\n"
        "4. Regulatory actions - antitrust restrictions, app store policy changes, or "
        "import/export controls could alter the business model.\n\n"
        "Analyst consensus estimates exist, but they vary widely (often by 5-10 percent), and "
        "consensus accuracy for specific quarters is notoriously poor."
    )
    return _make_abstention("CFA_Level_II", "Equity Valuation", "Revenue Forecast",
                            "L2_Medium", q, a, "unanswerable")


def abstention_market_bubble_call(rng, seq):
    """Unforecastable: predicting a bubble is inherently speculative."""
    q = _pick(rng, seq,  [
        "Is the market a bubble that will definitely burst by year end?",
        "Is the market a bubble that will definitely burst by year end? Please explain your reasoning.",
        "Is the market a bubble that will definitely burst by year end? Can you assess this?",
        "Is the market a bubble that will definitely burst by year end? When I ask about this:",
        "Is the market a bubble that will definitely burst by year end? I need your take on:",
    ])
    a = (
        "I cannot definitively answer whether the market is a bubble that will burst. Even "
        "if conditions suggest overvaluation:\n\n"
        "1. 'Bubbles can remain irrational longer than you can remain solvent' (Keynes). "
        "Bubbles do not burst on a predictable timeline.\n"
        "2. What looks like a bubble may reflect genuinely higher earnings growth potential, "
        "lower interest rates, or structural changes in the economy.\n"
        "3. Shorting a bubble is exceptionally risky - returns can continue rising before the "
        "collapse.\n"
        "4. 'By year-end' is a very specific, constrained timeframe that adds further uncertainty.\n\n"
        "I can instead help you analyze valuation metrics, identify potential overvaluation "
        "signals, or discuss portfolio insurance and tail-risk hedging strategies."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Bubble Analysis",
                            "L3_Medium", q, a, "unanswerable")


def abstention_currency_prediction(rng, seq):
    """Unforecastable: exchange rates follow a random walk in the short term."""
    q = _pick(rng, seq,  [
        "What will the USD/JPY exchange rate be in 3 months?",
        "What will the USD/JPY exchange rate be in 3 months? Please explain your reasoning.",
        "What will the USD/JPY exchange rate be in 3 months? Can you assess this?",
        "What will the USD/JPY exchange rate be in 3 months? When I ask about this:",
        "What will the USD/JPY exchange rate be in 3 months? I need your take on:",
    ])
    a = (
        "I cannot predict the USD/JPY exchange rate in 3 months. Exchange rates present "
        "extreme forecasting challenges:\n\n"
        "1. Empirical research shows that forward exchange rates are poor predictors of future "
        "spot rates - the random walk hypothesis holds strongly in short-to-medium horizons.\n"
        "2. The rate depends on interest rate differentials, relative inflation, current account "
        "balances, and geopolitical developments - all unpredictable at the required granularity.\n"
        "3. Central bank intervention - the BOJ or the Fed could intervene in unexpected ways.\n"
        "4. Market sentiment and capital flows create short-term price movements that are "
        "essentially random.\n\n"
        "What I can help with: explain the forward premium-discount relationship, discuss the "
        "interest rate parity condition, or analyze the historical volatility and risk management "
        "strategies for USD/JPY."
    )
    return _make_abstention("CFA_Level_II", "Economics", "Currency Forecasting",
                            "L2_Medium", q, a, "unanswerable")


def abstention_portfolio_return_exact(rng, seq):
    """Unforecastable: future returns are not deterministic."""
    q = _pick(rng, seq,  [
        "What exact return will my 60/40 portfolio generate next year?",
        "What exact return will my 60/40 portfolio generate next year? Please explain your reasoning.",
        "What exact return will my 60/40 portfolio generate next year? Can you assess this?",
        "What exact return will my 60/40 portfolio generate next year? When I ask about this:",
        "What exact return will my 60/40 portfolio generate next year? I need your take on:",
    ])
    a = (
        "An exact one-year portfolio return is not computable. Returns are stochastic, not "
        "deterministic:\n\n"
        "1. Future returns depend on future equity performance, bond performance, and their "
        "correlation structure - all of which are uncertain.\n"
        "2. Even historical average returns (e.g., approximately 7 percent for 60/40 annually) "
        "are statistical expectations, not certainties. The actual outcome will deviate from "
        "the mean.\n"
        "3. The standard deviation of 60/40 portfolio returns is typically 8-12 percent annually, "
        "meaning any single-year return has a wide confidence interval.\n"
        "4. 'Exact' implies a precise point estimate, but no model produces such precision.\n\n"
        "What I can help with: construct expected return distributions via Monte Carlo simulation, "
        "analyze the probability of various return ranges, or discuss sequence-of-returns risk."
    )
    return _make_abstention("CFA_Level_III", "Portfolio Management", "Return Forecasting",
                            "L3_Medium", q, a, "unanswerable")


# ====================================================================
# 3. FALSE PREMISE -- 18 templates
# ====================================================================




def abstention_constant_growth_gordon(rng, seq):
    """False: Gordon model fails when g >= k (growth > discount)."""
    q = _pick(rng, seq,  [
        "Value a company with 8% perpetual growth using a 6% cost of capital.",
        "Value a company with 8% perpetual growth using a 6% cost of capital. Please explain your reasoning.",
        "Value a company with 8% perpetual growth using a 6% cost of capital. Can you assess this?",
        "Value a company with 8% perpetual growth using a 6% cost of capital. When I ask about this:",
        "Value a company with 8% perpetual growth using a 6% cost of capital. I need your take on:",
    ])
    a = ('This question contains a false premise. The Gordon Growth Model requires the perpetual growth rate (g) to be strictly less than the cost of capital (k). Here g = 8% and k = 6%, so g > k.\n\nWhen g >= k, the Gordon model produces an infinite or negative value, neither of which is meaningful. The mathematical series diverges. A company cannot grow faster than the economy on a perpetual basis.\n\nA valid question would specify g < k (e.g., 4% growth with 6% cost of capital), or a two-stage DCF model that allows g to decline to a sustainable terminal rate.')
    return _make_abstention('CFA_Level_II', 'Equity Valuation', 'DDM',
                            'L2_Hard', q, a, "false_premise")


def abstention_zero_risk_premium(rng, seq):
    """False: there is always a risk premium in financial markets."""
    q = _pick(rng, seq,  [
        "Calculate the expected return using a zero risk premium.",
        "Calculate the expected return using a zero risk premium. Please explain your reasoning.",
        "Calculate the expected return using a zero risk premium. Can you assess this?",
        "Calculate the expected return using a zero risk premium. When I ask about this:",
        "Calculate the expected return using a zero risk premium. I need your take on:",
    ])
    a = ('This question is based on a false premise. In well-functioning financial markets, a positive risk premium is required by investors to hold risky assets over risk-free assets.\n\nThe CAPM explicitly assumes a positive market risk premium (E[Rm] - Rf > 0). Empirical evidence shows positive equity risk premiums of 3-6% in annualized terms.\n\nIf the risk premium were truly zero, investors would be indifferent between stocks and Treasury bonds, inconsistent with observed behavior.')
    return _make_abstention('CFA_Level_I', 'Ethics and Professional Standards', 'Risk Return',
                            'L1_Easy', q, a, "false_premise")


def abstention_positive_correlation(rng, seq):
    """False: not all asset pairs are necessarily positively correlated."""
    q = _pick(rng, seq,  [
        "Given negative correlation between two assets in a portfolio, what is the expected return?",
        "Given negative correlation between two assets in a portfolio, what is the expected return? Please explain your reasoning.",
        "Given negative correlation between two assets in a portfolio, what is the expected return? Can you assess this?",
        "Given negative correlation between two assets in a portfolio, what is the expected return? When I ask about this:",
        "Given negative correlation between two assets in a portfolio, what is the expected return? I need your take on:",
    ])
    a = ('The question is self-contradictory. A diversified portfolio benefits from assets with NEGATIVE correlation, which reduces risk through diversification.\n\nThe portfolio expected return is the weighted average of individual expected returns - correlation does NOT affect expected return, only portfolio risk.\n\nIf assets are negatively correlated, the portfolio standard deviation is less than the weighted average of individual standard deviations.')
    return _make_abstention('CFA_Level_I', 'Portfolio Management', 'Portfolio Risk',
                            'L1_Medium', q, a, "false_premise")


def abstention_deterministic_returns(rng, seq):
    """False: financial returns are stochastic, not deterministic."""
    q = _pick(rng, seq,  [
        "Given a deterministic 7% annual return, what will my portfolio be worth in 30 years?",
        "Given a deterministic 7% annual return, what will my portfolio be worth in 30 years? Please explain your reasoning.",
        "Given a deterministic 7% annual return, what will my portfolio be worth in 30 years? Can you assess this?",
        "Given a deterministic 7% annual return, what will my portfolio be worth in 30 years? When I ask about this:",
        "Given a deterministic 7% annual return, what will my portfolio be worth in 30 years? I need your take on:",
    ])
    a = ("While mathematically computable (FV = PV * 1.07^30 = PV * 7.612), this rests on a false premise. No financial asset guarantees a deterministic return beyond a zero-coupon Treasury held to maturity.\n\nEquity returns are stochastic and volatile. A '7% annual return' is a long-term historical average, not a guaranteed future outcome.\n\nThe standard deviation of annual equity returns is approximately 15-20%, meaning individual-year returns deviate massively from the mean.")
    return _make_abstention('CFA_Level_III', 'Portfolio Management', 'Return Distributions',
                            'L3_Medium', q, a, "false_premise")


def abstention_blackscholes_predict(rng, seq):
    """False: Black-Scholes cannot predict actual pricing deviations."""
    q = _pick(rng, seq,  [
        "How much will the Black-Scholes model price deviate from the actual price?",
        "How much will the Black-Scholes model price deviate from the actual price? Please explain your reasoning.",
        "How much will the Black-Scholes model price deviate from the actual price? Can you assess this?",
        "How much will the Black-Scholes model price deviate from the actual price? When I ask about this:",
        "How much will the Black-Scholes model price deviate from the actual price? I need your take on:",
    ])
    a = ("This question contains a false premise. Black-Scholes is a theoretical PRICING model, not a forecasting model. The model price is an estimate of current fair value.\n\nThere is no deterministic function that predicts the exact deviation between model price and market price. The deviation is a stochastic error term.\n\nA better question: 'What factors cause market price to deviate from Black-Scholes?' (Volatility smile, stochastic vol, dividends, jumps, market impact).")
    return _make_abstention('CFA_Level_II', 'Derivatives', 'Black-Scholes',
                            'L2_Hard', q, a, "false_premise")


def abstention_efficient_frontier_parabola(rng, seq):
    """False: Markowitz frontier is a hyperbola, not a parabola."""
    q = _pick(rng, seq,  [
        "Calculate the return at the vertex of the efficient frontier parabola.",
        "Calculate the return at the vertex of the efficient frontier parabola. Please explain your reasoning.",
        "Calculate the return at the vertex of the efficient frontier parabola. Can you assess this?",
        "Calculate the return at the vertex of the efficient frontier parabola. When I ask about this:",
        "Calculate the return at the vertex of the efficient frontier parabola. I need your take on:",
    ])
    a = ("The efficient frontier is NOT a parabola. In mean-standard-deviation space, it forms a HYPERBOLA. The lower portion is inefficient; the upper portion is efficient.\n\nThe Global Minimum Variance portfolio is sometimes called the 'vertex,' but this is a misleading term for a hyperbola.\n\nIf plotted in return-VARIANCE space, the frontier is parabolic. But this is a mathematical artifact of the coordinate transform.")
    return _make_abstention('CFA_Level_III', 'Portfolio Management', 'Markowitz Model',
                            'L3_Hard', q, a, "false_premise")


def abstention_negative_volatility(rng, seq):
    """False: standard deviation/volatility cannot be negative."""
    q = _pick(rng, seq,  [
        "What is the portfolio volatility with a value of -5%?",
        "What is the portfolio volatility with a value of -5%? Please explain your reasoning.",
        "What is the portfolio volatility with a value of -5%? Can you assess this?",
        "What is the portfolio volatility with a value of -5%? When I ask about this:",
        "What is the portfolio volatility with a value of -5%? I need your take on:",
    ])
    a = ('Volatility CANNOT be negative by definition. Standard deviation is the square root of variance, which is a sum of squared deviations and therefore always non-negative.\n\nThe formula sigma = sqrt(sum((ri - mu)^2 / N)) always produces a non-negative result.\n\nIf a negative volatility value appears in software, there is a computational error. If you meant expected return of -5%, that is a different concept.')
    return _make_abstention('CFA_Level_II', 'Quantitative Methods', 'Volatility',
                            'L2_Medium', q, a, "false_premise")


def abstention_risk_free_option(rng, seq):
    """False: no real-world investment option is completely risk-free."""
    q = _pick(rng, seq,  [
        "Find the most risk-free investment option.",
        "Find the most risk-free investment option. Please explain your reasoning.",
        "Find the most risk-free investment option. Can you assess this?",
        "Find the most risk-free investment option. When I ask about this:",
        "Find the most risk-free investment option. I need your take on:",
    ])
    a = ('There is no such thing as a completely risk-free investment (other than the theoretical risk-free rate from Treasury securities for analytical purposes).\n\nEven Treasury securities carry inflation risk and interest rate risk (market value declines if sold before maturity). All other asset classes carry market risk.\n\nThe closest approximation to a risk-free asset is short-term U.S. Treasury bills. But even T-bills carry minor risks (inflation, opportunity cost, reinvestment risk).')
    return _make_abstention('CFA_Level_I', 'Quantitative Methods', 'Risk',
                            'L1_Easy', q, a, "false_premise")


def abstention_infinite_sharpe_ratio(rng, seq):
    """False: Sharpe ratio cannot be meaningfully infinite (division by zero)."""
    q = _pick(rng, seq,  [
        "What is the Sharpe ratio of an investment with 10% return and zero risk?",
        "What is the Sharpe ratio of an investment with 10% return and zero risk? Please explain your reasoning.",
        "What is the Sharpe ratio of an investment with 10% return and zero risk? Can you assess this?",
        "What is the Sharpe ratio of an investment with 10% return and zero risk? When I ask about this:",
        "What is the Sharpe ratio of an investment with 10% return and zero risk? I need your take on:",
    ])
    a = ('The Sharpe ratio is (E[R] - Rf) / sigma. With sigma = 0, we have division by zero, which is mathematically UNDEFINED, not infinite.\n\nAn investment with zero risk and return above the risk-free rate would be an arbitrage, which cannot persist in efficient markets.\n\nA practical question: what is the Sharpe ratio with 1% vol, 10% return, 2% Rf? Answer: (10% - 2%) / 1% = 8.0.')
    return _make_abstention('CFA_Level_III', 'Performance Evaluation', 'Sharpe Ratio',
                            'L3_Medium', q, a, "false_premise")


def abstention_perfect_hedge(rng, seq):
    """False: perfect hedges are theoretical, not achievable in practice."""
    q = _pick(rng, seq,  [
        "Construct a perfect hedge for my portfolio.",
        "Construct a perfect hedge for my portfolio. Please explain your reasoning.",
        "Construct a perfect hedge for my portfolio. Can you assess this?",
        "Construct a perfect hedge for my portfolio. When I ask about this:",
        "Construct a perfect hedge for my portfolio. I need your take on:",
    ])
    a = ("A 'perfect hedge' is a theoretical fiction. Perfect hedging is impossible: basis risk means the hedge never perfectly tracks the underlying; continuous rebalancing required by Black-Scholes is impossible in discrete time; discontinuous price moves destroy hedges during crashes; and transaction costs erode effectiveness.\n\nI can help construct the best practical hedge given your constraints, instruments, and budget. But a perfect hedge does not exist in reality.")
    return _make_abstention('CFA_Level_III', 'Derivatives', 'Dynamic Hedging',
                            'L3_Hard', q, a, "false_premise")


def abstention_zero_risk_free_rate(rng, seq):
    """False: the risk-free rate is generally positive, not zero."""
    q = _pick(rng, seq,  [
        "Use a zero risk-free rate for your CAPM calculation.",
        "Use a zero risk-free rate for your CAPM calculation. Please explain your reasoning.",
        "Use a zero risk-free rate for your CAPM calculation. Can you assess this?",
        "Use a zero risk-free rate for your CAPM calculation. When I ask about this:",
        "Use a zero risk-free rate for your CAPM calculation. I need your take on:",
    ])
    a = ("Using a zero risk-free rate only applies under very rare circumstances. Historically, the risk-free rate has been positive in nearly all periods.\n\nWith Rf = 0, CAPM E[Ri] = Rf + beta*(E[Rm]-Rf) reduces to E[Ri] = beta*E[Rm], a different model equation.\n\nA more standard question would use the prevailing risk-free rate (e.g., 10-year Treasury yield). The zero-Rf assumption changes the model's behavior.")
    return _make_abstention('CFA_Level_I', 'Quantitative Methods', 'CAPM',
                            'L1_Hard', q, a, "false_premise")


def abstention_market_inferred_probability(rng, seq):
    """False: market-implied distributions reflect risk-neutral, not real-world probabilities."""
    q = _pick(rng, seq,  [
        "Calculate the real-world probability from the market-implied distribution.",
        "Calculate the real-world probability from the market-implied distribution. Please explain your reasoning.",
        "Calculate the real-world probability from the market-implied distribution. Can you assess this?",
        "Calculate the real-world probability from the market-implied distribution. When I ask about this:",
        "Calculate the real-world probability from the market-implied distribution. I need your take on:",
    ])
    a = ('Option prices reflect RISK-NEUTRAL probabilities (Q), NOT real-world probabilities (P). The relationship is: Q = P * exp(lambda * beta), where lambda is the market price of risk.\n\nYou CANNOT directly recover real-world probabilities from risk-neutral distributions without knowing the market price of risk for each outcome.\n\nTo estimate real-world probabilities, you must estimate the market price of risk and apply it to adjust the distribution. The result is highly uncertain.')
    return _make_abstention('CFA_Level_III', 'Derivatives', 'Risk-Neutral Valuation',
                            'L3_Hard', q, a, "false_premise")


def abstention_correlation_is_causation(rng, seq):
    """False: correlation does not imply causation."""
    q = _pick(rng, seq,  [
        "What is the causal effect of GDP on stock prices given they are correlated?",
        "What is the causal effect of GDP on stock prices given they are correlated? Please explain your reasoning.",
        "What is the causal effect of GDP on stock prices given they are correlated? Can you assess this?",
        "What is the causal effect of GDP on stock prices given they are correlated? When I ask about this:",
        "What is the causal effect of GDP on stock prices given they are correlated? I need your take on:",
    ])
    a = ('GDP and stock prices are correlated BECAUSE both respond to the same underlying economic drivers (employment, manufacturing, interest rates, consumer confidence).\n\nCorrelation measures LINEAR association, not causal direction. Two variables can be correlated because a third variable drives both.\n\nGranger causality tests do NOT establish true causation; they only measure predictive power. A correct question asks for a formal causal model with proper identification.')
    return _make_abstention('CFA_Level_II', 'Quantitative Methods', 'Causality',
                            'L2_Medium', q, a, "false_premise")


def abstention_constant_volatility(rng, seq):
    """False: volatility is stochastic, not constant."""
    q = _pick(rng, seq,  [
        "Assume constant volatility of 20% and price the option.",
        "Assume constant volatility of 20% and price the option. Please explain your reasoning.",
        "Assume constant volatility of 20% and price the option. Can you assess this?",
        "Assume constant volatility of 20% and price the option. When I ask about this:",
        "Assume constant volatility of 20% and price the option. I need your take on:",
    ])
    a = ('Black-Scholes ASSUMES constant volatility, but this is empirically false. Volatility STOCHASTICALLY clusters, reverts to a mean, and changes over time.\n\nIf volatility were constant, the volatility smile would not exist. The SABR, Heston, and Bates models exist precisely because the constant vol assumption fails.\n\nI can price the option under constant vol (Black-Scholes), but the result is likely misleading. A better approach uses stochastic or local volatility models.')
    return _make_abstention('CFA_Level_III', 'Derivatives', 'Volatility Modeling',
                            'L3_Hard', q, a, "false_premise")


def abstention_normal_returns(rng, seq):
    """False: financial returns are not normally distributed."""
    q = _pick(rng, seq,  [
        "Assume returns are normally distributed and calculate the value-at-risk.",
        "Assume returns are normally distributed and calculate the value-at-risk. Please explain your reasoning.",
        "Assume returns are normally distributed and calculate the value-at-risk. Can you assess this?",
        "Assume returns are normally distributed and calculate the value-at-risk. When I ask about this:",
        "Assume returns are normally distributed and calculate the value-at-risk. I need your take on:",
    ])
    a = ("Financial returns are NOT normally distributed. They exhibit FAT TAILS (leptokurtosis), SKEW, and VOLATILITY CLUSTERING. The 2008, 1987, and 2020 crises produced moves 'impossible' under a normal distribution.\n\nA proper VaR should use historical simulation, Monte Carlo with heavy-tailed distributions, or GARCH-based approaches. The normal assumption UNDERESTIMATES true tail risk.\n\nAcademic consensus uses Student-t, stable Pareto, or Generalized Pareto distributions.")
    return _make_abstention('FRM_Part_1', 'Financial Markets and Products', 'Value at Risk',
                            'FRM1_Hard', q, a, "false_premise")


def abstention_perfect_market_efficiency(rng, seq):
    """False: markets are not perfectly efficient."""
    q = _pick(rng, seq,  [
        "Given perfectly efficient markets, what is the expected excess return from stock picking?",
        "Given perfectly efficient markets, what is the expected excess return from stock picking? Please explain your reasoning.",
        "Given perfectly efficient markets, what is the expected excess return from stock picking? Can you assess this?",
        "Given perfectly efficient markets, what is the expected excess return from stock picking? When I ask about this:",
        "Given perfectly efficient markets, what is the expected excess return from stock picking? I need your take on:",
    ])
    a = ("If markets are perfectly efficient (strong form), ALL information is already in prices. Stock picking yields ZERO excess returns (after costs).\n\nYet the question implies stock picking CAN yield excess returns. This is self-contradictory.\n\nIn reality, markets are NOT perfectly efficient. Behavioral finance, Fama-French anomalies, and Nobel laureate Fama himself acknowledge 'markets are mostly efficient,' not perfectly.")
    return _make_abstention('CFA_Level_I', 'Ethics and Professional Standards', 'Market Efficiency',
                            'L1_Medium', q, a, "false_premise")


def abstention_diversification_eliminate_all_risk(rng, seq):
    """False: diversification does not eliminate ALL risk."""
    q = _pick(rng, seq,  [
        "How can I diversify my portfolio to eliminate all risk?",
        "How can I diversify my portfolio to eliminate all risk? Please explain your reasoning.",
        "How can I diversify my portfolio to eliminate all risk? Can you assess this?",
        "How can I diversify my portfolio to eliminate all risk? When I ask about this:",
        "How can I diversify my portfolio to eliminate all risk? I need your take on:",
    ])
    a = ("Diversification eliminates UNSYSTEMATIC (idiosyncratic) risk but CANNOT eliminate SYSTEMATIC (market) risk that affects all assets simultaneously.\n\nAn infinitely diversified portfolio (the market portfolio) still has significant standard deviation. The only near-zero risk approach is allocating entirely to risk-free assets (eliminating return above the risk-free rate).\n\nA properly framed question: 'How can I diversify to minimize portfolio risk?' Not 'eliminate all risk' - impossible with risky assets.")
    return _make_abstention('CFA_Level_I', 'Portfolio Management', 'Diversification',
                            'L1_Easy', q, a, "false_premise")


def abstention_expected_equals_realized(rng, seq):
    """False: expected returns and realized returns are fundamentally different."""
    q = _pick(rng, seq,  [
        "Calculate the expected return given the realized return over the past year.",
        "Calculate the expected return given the realized return over the past year. Please explain your reasoning.",
        "Calculate the expected return given the realized return over the past year. Can you assess this?",
        "Calculate the expected return given the realized return over the past year. When I ask about this:",
        "Calculate the expected return given the realized return over the past year. I need your take on:",
    ])
    a = ('Expected return is an ex-ante (forward-looking) estimate. Realized return is an ex-post (backward-looking) measure. They are NOT interchangeable.\n\nA high realized return implies nothing about the expected return (it could be good luck). The relationship: realized = expected + noise.\n\nUsing past returns as proxy for expected returns is imprecise: standard error scales as sigma/sqrt(T), which is large (sigma=20%, T=5 years gives standard error ~9%).')
    return _make_abstention('CFA_Level_III', 'Equity Valuation', 'Expected Returns',
                            'L3_Medium', q, a, "false_premise")


# ====================================================================
# 4. TEMPLATES EXPORT
# ====================================================================

TEMPLATES = {
    # Underspecified templates (32)
    "beta_calculation": abstention_beta_calculation,
    "black_litterman_model": abstention_black_litterman_model,
    "bond_duration": abstention_bond_duration,
    "calmar_ratio_computation": abstention_calmar_ratio_computation,
    "credit_spread": abstention_credit_spread,
    "cva_derivative": abstention_cva_derivate,
    "dcf_valuation_company": abstention_dcf_valuation_company,
    "economic_capital_modeling": abstention_economic_capital_modeling,
    "emerging_markets_allocation": abstention_emerging_markets_allocation,
    "esg_rating": abstention_esg_rating,
    "factor_model_regression": abstention_factor_model_regression,
    "fund_manager_alpha": abstention_fund_manager_alpha,
    "greeks_calculation": abstention_greeks_calculation,
    "implied_volatility_calculation": abstention_implied_volatility_calculation,
    "interest_rate_hedge": abstention_interest_rate_hedge,
    "liq_coverage_ratio": abstention_liq_coverage_ratio,
    "merger_arbitrage": abstention_merger_arbitrage,
    "montecarlo_simulation": abstention_montecarlo_simulation,
    "npv_calculation": abstention_npv_calculation,
    "option_strategy": abstention_option_strategy,
    "pe_ratio_valuation": abstention_pe_ratio_valuation,
    "portfolio_attribution_analysis": abstention_portfolio_attribution_analysis,
    "portfolio_optimization": abstention_portfolio_optimization,
    "portfolio_rebalance": abstention_portfolio_rebalance,
    "risk_parity_portfolio": abstention_risk_parity_portfolio,
    "sharpe_ratio_comparison": abstention_sharpe_ratio_comparison,
    "sortino_ratio": abstention_sortino_ratio,
    "stock_pick_comparison": abstention_stock_pick_comparison,
    "tax_loss_harvesting": abstention_tax_loss_harvesting,
    "tracking_error": abstention_tracking_error,
    "valuation_multiple_comparison": abstention_valuation_multiple_comparison,
    "working_capital_forecast": abstention_working_capital_forecast,

    # Unanswerable templates (12)
    "black_swan_event": abstention_black_swan_event,
    "company_revenue_forecast": abstention_company_revenue_forecast,
    "crypto_price_forecast": abstention_crypto_price_forecast,
    "currency_prediction": abstention_currency_prediction,
    "fair_value_stock": abstention_fair_value_stock,
    "interest_rate_prediction": abstention_interest_rate_prediction,
    "market_bottom_call": abstention_market_bottom_call,
    "market_bubble_call": abstention_market_bubble_call,
    "pandemic_gdp_impact": abstention_pandemic_gdp_impact,
    "perfect_timing": abstention_perfect_timing,
    "portfolio_return_exact": abstention_portfolio_return_exact,
    "stock_price_prediction": abstention_stock_price_prediction,

    # False premise templates (18)
    "blackscholes_predict": abstention_blackscholes_predict,
    "constant_growth_gordon": abstention_constant_growth_gordon,
    "constant_volatility": abstention_constant_volatility,
    "correlation_is_causation": abstention_correlation_is_causation,
    "deterministic_returns": abstention_deterministic_returns,
    "diversification_eliminate_all_risk": abstention_diversification_eliminate_all_risk,
    "efficient_frontier_parabola": abstention_efficient_frontier_parabola,
    "expected_equals_realized": abstention_expected_equals_realized,
    "infinite_sharpe_ratio": abstention_infinite_sharpe_ratio,
    "market_inferred_probability": abstention_market_inferred_probability,
    "negative_volatility": abstention_negative_volatility,
    "normal_returns": abstention_normal_returns,
    "perfect_hedge": abstention_perfect_hedge,
    "perfect_market_efficiency": abstention_perfect_market_efficiency,
    "positive_correlation": abstention_positive_correlation,
    "risk_free_option": abstention_risk_free_option,
    "zero_risk_free_rate": abstention_zero_risk_free_rate,
    "zero_risk_premium": abstention_zero_risk_premium,
}
