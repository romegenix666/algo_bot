# Curated Research Papers

> These are the papers we are *grounded* in. Each one is either freely available
> on SSRN/arXiv, the publisher, or the authors’ sites. Each entry has a link
> you can verify.

---

## Why not “just Feynman AI” (or any one research bot)?

**You can use Feynman** (or ChatGPT, Perplexity, Elicit, etc.) as a *search and
summarisation assistant* — it is **not** a substitute for:

1. **Primary sources** — strategy claims should trace to a PDF you can open.
2. **Reproducibility** — the bot may hallucinate titles, years, or DOIs unless
   you verify every link.
3. **Your stack** — Feynman needs install + API keys + network; earlier we hit
   tooling friction (`npm`/auth). That is a **practical** barrier, not a
   theoretical one: wire it up if you want, and still keep this file as the
   canonical list.

**Google Scholar** is the right place to sort by citations. We do **not** rely on
live citation ranks inside the codebase (they change weekly). Instead we lock
in a **small set of seminal, heavily cited references** that the industry and
academia already treat as canonical for (a) edges and (b) failure modes.

**What “integrate into code” means here:** research does not become alpha by
magic import. Integration = **(1)** this document, **(2)** docstrings in
`src/strategies/`, **(3)** metrics you already implement (e.g. deflated Sharpe,
Lo-corrected Sharpe, walk-forward, lookahead audit). Below, **Traceability**
maps each cluster to modules.

---

## Traceability — papers → code

| Theme | Primary references | Where it lives |
|-------|-------------------|----------------|
| Cross-sectional momentum | Jegadeesh & Titman (1993, 2001); Carhart (1997) | `src/strategies/momentum.py` |
| Time-series / dual momentum | Moskowitz–Ooi–Pedersen (2012); Antonacci (2014 book); Faber (2007) | `src/strategies/dual_momentum.py` |
| Value + momentum + factors | Fama–French (1993, 2015); Asness et al. (2013) | `src/strategies/multi_factor.py` |
| Mean reversion / bands | Chan (2009 book); Kakushadze & Serur §3.9 | `src/strategies/mean_reversion.py` |
| Pairs / cointegration | Engle–Granger (1987); Avellaneda–Lee (2010) | `src/strategies/pairs.py` |
| Breakout / trend | Donchian; Turtle literature; Wilder ADX | `src/strategies/breakout.py` |
| Sector rotation | Faber (2010); Moskowitz–Grinblatt (1999) | `src/strategies/sector_rotation.py` |
| Sentiment as filter | Tetlock (2007); Loughran–McDonald (2011) | `src/strategies/sentiment_momentum.py`, `src/sentiment/` |
| Overfitting / multiple testing | Bailey et al.; Harvey–Liu–Zhu; PBO / DSR | `src/backtest/metrics.py`, `src/backtest/multiple_testing.py`, `src/backtest/benchmark.py`, walk-forward |
| Sharpe misuse | Lo (2002) | `src/backtest/metrics.py` (`lo_sharpe_ratio`) |
| Crisis / crowding | Khandani–Lo (2007) | `docs/DOS_AND_DONTS.md`, `src/risk/circuit_breaker.py` |

---

## Top 10 — Edges, risk-aware returns (seminal / very highly cited)

*Rough order: momentum & factors first (most replicated), then stat arb and
information / sentiment. Verify citation counts on [Google Scholar](https://scholar.google.com).*

### P1. Jegadeesh & Titman (1993) — Momentum
**"Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"**
*Journal of Finance, Vol. 48, No. 1*

- **Edge:** Stocks with strong 3–12-month past returns continue to outperform over the next 3–12 months.
- **Why it matters for us:** Foundation of strategy 2.1 (Cross-Sectional Momentum). Replicated globally including India (multiple NSE academic papers).
- **Implementation key:** Skip the most recent month (1-month reversal contaminates pure momentum).
- **Free PDF:** [Yale](https://depot.som.yale.edu/icf/papers/fileuploads/2387/original/93-04.pdf) · DOI: `10.1111/j.1540-6261.1993.tb04702.x`

### P2. Jegadeesh & Titman (2001) — Momentum robustness
**"Profitability of Momentum Strategies: An Evaluation of Alternative Explanations"**
*Journal of Finance, Vol. 56, No. 2*

- **Edge:** Follow-up: momentum profits persist post-1990 sample; debates risk vs behavioural explanations.
- **Why it matters:** Validates that 1993 was not a one-off; sets expectations that premia vary by era and implementation.
- **Link:** [NBER w7159](https://www.nber.org/papers/w7159) · DOI: `10.1111/0022-1082.00342`

### P3. Carhart (1997) — Momentum as a priced factor
**"On Persistence in Mutual Fund Performance"**
*Journal of Finance, Vol. 52, No. 1*

- **Edge:** Adds **momentum factor** to the Fama–French setup; framework for ranking assets on momentum *controlling* for other factors.
- **Why it matters:** Bridges single-sort momentum stories to portfolio construction and risk attribution.
- **Link:** [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb03813.x)

### P4. Fama & French (1993, 2015) — Multi-factor models
**"Common Risk Factors in the Returns on Stocks and Bonds"** (1993, *Journal of Financial Economics*)
**"A Five-Factor Asset Pricing Model"** (2015, *Journal of Financial Economics*)

- **Edge:** Size, value, profitability, investment (plus market) explain average returns; long–short factor portfolios are the workhorse of systematic equity.
- **Why it matters:** Foundation of `multi_factor.py` and any “quality / value / low vol” scoring.
- **Free PDFs:** [Chicago Booth copy of 1993](https://faculty.chicagobooth.edu/john.cochrane/teaching/35150_advanced_investments/Fama_French_Common_Risk_Factors.pdf), [SSRN 2287202](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2287202) (2015).

### P5. Asness, Moskowitz & Pedersen (2013) — Value and momentum “everywhere”
**"Value and Momentum Everywhere"**
*Journal of Finance, Vol. 68, No. 3*

- **Edge:** Value and momentum premia show up **across** asset classes and regions (stocks, bonds, FX, commodities in their setting).
- **Why it matters:** Justifies using momentum + value together and not treating India as a magical exception.
- **Link:** [DOI 10.1111/jofi.12021](https://doi.org/10.1111/jofi.12021) · [NYU working PDF](https://pages.stern.nyu.edu/~lpederse/papers/ValMomEverywhere.pdf)

### P6. Moskowitz, Ooi & Pedersen (2012) — Time-series momentum
**"Time Series Momentum"**
*Journal of Financial Economics, Vol. 104, No. 2*

- **Edge:** Trend-following on **futures** (12–1 style) works out-of-sample across horizons; complements cross-sectional stock momentum.
- **Why it matters:** Underpins **absolute** trend filters (e.g. dual momentum, defensive cash rules).
- **Link:** [AQR PDF](https://www.aqr.com/-/media/AQR/Documents/Insights/Research/Journal-Article/Time-Series-Momentum.pdf)

### P7. Engle & Granger (1987) — Cointegration
**"Co-Integration and Error Correction: Representation, Estimation, and Testing"**
*Econometrica, Vol. 55, No. 2*

- **Edge:** Formal foundation for **pairs** and mean-reversion in *spreads* between co-integrated series.
- **Why it matters:** Justifies Engle–Granger tests before trading a spread (see `pairs.py`).
- **Link:** [JSTOR](https://www.jstor.org/stable/1913236) · DOI: `10.2307/1913236`

### P8. Avellaneda & Lee (2010) — Statistical arbitrage
**"Statistical Arbitrage in the U.S. Equities Market"**
*Quantitative Finance, Vol. 10, No. 7*

- **Edge:** Residual returns from simple risk models mean-revert; z-score trading of residuals.
- **Why it matters:** Mathematical cousin of our cointegration / z-score pair engine (different estimation, same *idea*: trade stationary residuals).
- **Free PDF:** [arXiv:0902.0254](https://arxiv.org/abs/0902.0254)

### P9. Tetlock (2007) — Media tone and returns
**"Giving Content to Investor Sentiment: The Role of Media in the Stock Market"**
*Journal of Finance, Vol. 62, No. 3*

- **Edge:** Negative tone predicts downward pressure and reversals at short horizons.
- **Why it matters:** Supports using sentiment as a **filter**, not a standalone alpha (see `sentiment_momentum.py`).
- **Free PDF:** [author page](http://www.columbia.edu/~paul.tetlock/publications.htm) · DOI: `10.1111/j.1540-6261.2007.01232.x`

### P10. Loughran & McDonald (2011) — Finance-specific text
**"When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks"**
*Journal of Finance, Vol. 66, No. 1*

- **Edge:** Off-the-shelf “positive/negative” word lists **mislead** in finance; domain-specific lexicons matter.
- **Why it matters:** Motivates FinBERT / finance-tuned sentiment rather than generic VADER-only conclusions.
- **Link:** [DOI 10.1111/j.1540-6261.2010.01625.x](https://doi.org/10.1111/j.1540-6261.2010.01625.x)

---

## Top 10 — What NOT to do (pitfalls, overfitting, crowding)

### N1. Bailey, Borwein, López de Prado, Zhu (2014) — Backtest Overfitting
**"Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance"**
*Notices of the AMS, Vol. 61, No. 5*

- **Lesson:** With enough trial-and-error, you can produce a 4+ Sharpe in-sample backtest **for any random time series**. Most published / claimed Sharpe ratios are statistically meaningless.
- **What we do about it:** Walk-forward only, ≤5 params, sensitivity test, paper-trade buffer.
- **Free PDF:** [AMS](https://www.ams.org/journals/notices/201405/rnoti-p458.pdf)

### N2. Bailey & López de Prado (2014) — Deflated Sharpe Ratio
**"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"**
*Journal of Portfolio Management, Vol. 40, No. 5*

- **Lesson:** When you've tried N strategies and pick the best, your Sharpe is *deflated* by a known formula. Naïve Sharpe ratios overstate skill.
- **What we do:** Compute the deflated Sharpe before declaring any strategy "good".
- **Free PDF:** [SSRN 2460551](https://ssrn.com/abstract=2460551)

### N3. Harvey, Liu & Zhu (2016) — Factor Zoo
**"... and the Cross-Section of Expected Returns"**
*Review of Financial Studies, Vol. 29, No. 1*

- **Lesson:** 300+ factors have been published as "anomalies"; most do not survive multiple-testing corrections. Anything claimed without t-stat ≥ 3 (after multiple testing) is suspect.
- **What we do:** Use only well-replicated factors (momentum, value, low-vol, quality). Skip the exotic stuff.
- **Free PDF:** [SSRN 2249314](https://ssrn.com/abstract=2249314)

### N4. Khandani & Lo (2007) — The Quant Quake
**"What Happened to the Quants in August 2007?"**
*Journal of Investment Management, Vol. 5, No. 4*

- **Lesson:** When too many funds run similar strategies, simultaneous deleveraging causes catastrophic correlated losses (the "quant quake" of Aug 6–10 2007 wiped out years of alpha in 4 days).
- **What we do:** Run strategies that "fly under the radar" of large institutions. Diversify across uncorrelated strategies. Have a circuit breaker.
- **Free PDF:** [MIT](https://web.mit.edu/alo/www/Papers/august07.pdf) (this is the SAME paper Chan cites in §3.7!)

### N5. Lo (2002) — Statistics of Sharpe Ratios (Non-IID Returns)
**"The Statistics of Sharpe Ratios"**
*Financial Analysts Journal, Vol. 58, No. 4*

- **Lesson:** Standard Sharpe-ratio annualization (`Sharpe × √252`) **assumes returns are IID**. Real returns have autocorrelation; the naïve annualization can overstate Sharpe by 50%+.
- **What we do:** Use the autocorrelation-corrected Sharpe formula from this paper for any reported metric.
- **Free PDF:** [MIT](https://alo.mit.edu/wp-content/uploads/2017/06/The-Statistics-of-Sharpe-Ratios.pdf)

### N6. Bailey, Borwein, López de Prado & Zhu — Probability of Backtest Overfitting (PBO)
**"The Probability of Backtest Overfitting"**

- **Lesson:** Estimates how likely a “winning” backtest is **pure selection** when many configurations were tried (combinatorially symmetric cross-validation idea).
- **What we do:** Treat `run_sensitivity.py` / walk-forward / paper period as mandatory; prefer reporting **ranges** of outcomes, not one tuned peak.
- **Link:** [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)

### N7. Sullivan, Timmermann & White (1999) — Data-snooping in technical rules
**"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"**
*Journal of Finance, Vol. 54, No. 5*

- **Lesson:** Testing **many** rules on the same data inflates apparent significance; need bootstrap / reality-check style corrections.
- **What we do:** Limit parameters per strategy; run sensitivity ±20%; avoid “kitchen sink” indicator mining.
- **Link:** [DOI 10.1111/0022-1082.00163](https://doi.org/10.1111/0022-1082.00163)

### N8. White (2000) — Reality check for multiple models
**"A Reality Check for Data Snooping"**
*Econometrica, Vol. 68, No. 5*

- **Lesson:** Formal p-value for “best strategy under the null that none work” — sibling of Harvey-style multiple testing.
- **What we do:** Any “best” backtest must be compared to **deflated** stats and **out-of-sample** windows, not raw in-sample winners.
- **Link:** [DOI 10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152)

### N9. Romano & Wolf (2005) — Stepwise multiple testing
**"Stepwise Multiple Testing as Formalized Data Snooping"**
*Econometrica, Vol. 73, No. 4*

- **Lesson:** Standard single-hypothesis tests fail when you **choose** which factors to report after peeking at all of them.
- **What we do:** Stick to a **small** pre-declared factor set (momentum, value, quality, low vol) per `STRATEGY.md`.
- **Link:** [DOI 10.1111/j.1468-0262.2005.00615.x](https://doi.org/10.1111/j.1468-0262.2005.00615.x)

### N10. López de Prado (2018) — ML in finance done wrong vs right
**"Advances in Financial Machine Learning"** (Wiley)

- **Lesson:** Wrong labels (e.g. fixed-horizon returns), leakage in cross-validation, and meta-labelling mistakes create **fake** edges; triple-barrier and purged CV exist to reduce that.
- **What we do:** Until ML is added with those controls, keep rules simple; use **lookahead audit** and **point-in-time** data discipline from Chan / engine design.
- **SSRN hub:** [SSRN 3257420](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257420) (example chapter)

---

## Bonus — Indian-Market Specific Reading

These were not in the original "top 5/5" prompt but are critical because we're trading NSE:

- **Sehgal & Balakrishnan (2014)** — *"Robustness of Fama–French Three Factor Model"* (Indian market). Confirms factor anomalies in NSE.
- **Aggarwal & Gupta (2009)** — *"Empirical Evidence on the Efficiency of Indian Stock Market"*. Tests for predictability.
- **Saha (2017)** — *"Predicting Indian Stock Market using Twitter Sentiment"*. Methodology adaptable.
- **NSE Quantitative Research Reports** (free, [nseindia.com](https://www.nseindia.com/products/content/equities/equities/eq_research.htm)).

---

## How To Actually Read These (lazy mode)

You probably won't read 20 papers cover-to-cover. That's fine. **For each paper:**

1. Read the **abstract** (1 paragraph).
2. Read the **conclusion / discussion** section (~2 pages).
3. Look at the **headline tables/figures** (returns table, Sharpe table).
4. **Skip the math** unless you're implementing the formula.

This gives you 80% of the value in 20% of the time. The implementation detail
will be re-derived in our code anyway.

---

## When Feynman is Authenticated

Once you set up Feynman with an API key (Anthropic or OpenAI), run:

```bash
feynman lit "profitable algorithmic trading strategies for retail equities" --new-session
feynman lit "common pitfalls and overfitting in quantitative trading backtests" --new-session
feynman deepresearch "factor investing in Indian equities NSE" --new-session
feynman deepresearch "news sentiment analysis for stock prediction FinBERT" --new-session
```

Save the outputs into `docs/feynman/` for traceability.
