# Dos and Don'ts of Algorithmic Trading

> Distilled from Chan's *Quantitative Trading*, Kakushadze & Serur's *151 Trading Strategies*, the CFTC IRL paper, López de Prado's pitfalls papers, and the painful experiences of basically every retail algo trader who blew up.

---

## DOs

### Process & Discipline
1. **DO write down your trading rules before you trade.** If they aren't written, they don't exist when emotion takes over.
2. **DO paper-trade for at least 30 calendar days** before deploying any new strategy with real money.
3. **DO keep a trade journal.** For every override of the algo, log *why*. Review monthly.
4. **DO start with small capital.** Scale up only after 3 profitable live months.
5. **DO version-control everything** — strategy code, configs, even the seed for any randomness.
6. **DO assume your strategy will eventually stop working** (regime shift, alpha decay) and have a kill-switch.

### Backtesting
7. **DO use walk-forward validation.** Train on past, test on future. Never tune parameters on the test set.
8. **DO include realistic transaction costs** (Indian brokerage 0.03% + STT 0.025% + GST + slippage 5 bps). Many "profitable" backtests die after costs.
9. **DO test for survivorship bias.** Use point-in-time universes; include delisted stocks.
10. **DO run sensitivity analysis** — perturb every parameter ±20% and verify performance is robust.
11. **DO use the truncate-and-rerun look-ahead test** (Chan §3): run backtest with full data, save positions; rerun with last N days truncated; positions for overlapping period must be identical.
12. **DO Monte Carlo bootstrap** your trade returns to get a confidence interval on Sharpe and max DD.
13. **DO test on multiple market regimes** — bull (2014, 2017, 2021), bear (2008, 2020), sideways (2011, 2022).

### Risk Management
14. **DO use volatility-adjusted position sizes** (ATR or Kelly-based). Equal rupee allocation is wrong.
15. **DO set ATR-based stop-losses**, not fixed-percent stops.
16. **DO have a portfolio-level circuit breaker** (e.g., halt if drawdown > 12%).
17. **DO size positions using Half-Kelly, not full Kelly.** Full Kelly is theoretically optimal but practically lethal.
18. **DO diversify across uncorrelated strategies.** Mean-rev + momentum + pairs > any one of them alone.
19. **DO calculate the Sharpe ratio after fees** as your primary metric, not absolute return.

### Operations
20. **DO use idempotent order submission.** Retries should never double-place an order.
21. **DO have a manual kill-switch** that closes all positions with one command.
22. **DO monitor data quality.** If today's bar is stale or has nonsense values, don't trade.
23. **DO log every order, fill, and reason** in structured JSON. You will need this for debugging *and* tax filing.
24. **DO set up Telegram alerts** for: fills, stop-losses tripped, circuit-breaker fires, data outages, broker API errors.
25. **DO have a separate API key for live vs paper** and read-only for monitoring.

### Strategy Selection
26. **DO prefer simple strategies with few parameters** (≤5). Each parameter is an opportunity to overfit.
27. **DO have an economic rationale** for why your strategy works. "It backtested well" is not a reason — it's a *prerequisite*.
28. **DO favor strategies with a high Sharpe ratio over high absolute return.** Higher Sharpe + leverage > higher return + low Sharpe (Chan §6).
29. **DO trade strategies that institutional money can't run easily** (low capacity strategies — small caps, niche pairs, etc.).

### Sentiment / News
30. **DO use sentiment as a *filter*, not a primary signal.** News alone is too noisy.
31. **DO use FinBERT or similar finance-tuned models** rather than generic VADER for finance text.
32. **DO normalize sentiment scores cross-sectionally** — absolute scores are meaningless.

---

## DON'Ts

### Process & Discipline
1. **DON'T override your algo on a hunch.** If you must override, document why; if the override is profitable, treat it as data and retrain — don't keep overriding.
2. **DON'T trade with money you can't afford to lose.** Algo trading is *not* a guaranteed income source.
3. **DON'T add real capital after a losing week** "to make it back". This is gambler ruin.
4. **DON'T skip paper trading** because "the backtest looks great". Backtests lie. They always look great. That's why you keep refining them — survivorship bias of *backtests*.
5. **DON'T copy strategies from YouTube/Telegram channels** without independent backtesting. If they really worked, the creator wouldn't be selling them.

### Backtesting
6. **DON'T optimize parameters on your test set.** This is data-snooping bias and will produce backtests that are 5–10× better than reality.
7. **DON'T use ≥10 parameters in your strategy.** Chan's rule of thumb: with daily data, ≥252 data points per parameter; you'll run out of data fast.
8. **DON'T trust a backtest that uses high/low prices for entry.** Daily highs/lows are noisy and often unfillable in reality (Chan §3).
9. **DON'T forget transaction costs.** A 4.47 Sharpe before costs can become a -3.19 Sharpe after costs (Chan §3.7 example).
10. **DON'T use survivorship-biased data** for value strategies. The dead stocks were the cheap ones.
11. **DON'T assume the future looks like 2009–2021.** That was a free-money decade. Test on 2022 and 2008 too.
12. **DON'T rely on a single great backtest run.** Bootstrap, walk forward, sensitivity-test.

### Risk Management
13. **DON'T over-leverage.** Even a 60% win rate strategy with 2x leverage can blow up in a 4-loss streak.
14. **DON'T use a fixed-percent stop-loss across all stocks.** RELIANCE and a smallcap have totally different volatilities.
15. **DON'T put more than 20% of your portfolio in one stock**, no matter how confident the signal.
16. **DON'T trade options or F&O until you have 12 months of profitable equity-only live trading.** F&O leverage destroys retail accounts.
17. **DON'T ignore correlations.** "Diversifying" across 5 PSU banks is not diversification.
18. **DON'T move stops *farther* away when a position goes against you** ("hopium"). Move them closer (trailing) only when in profit.

### Operations
19. **DON'T put your API key in code or commit it to git.** Use `.env` + `.gitignore`. Use a secrets manager.
20. **DON'T trade right after a code change** without re-running all tests and a paper-trading day. Code regressions in trading systems are expensive.
21. **DON'T trust the broker's "got order" response blindly.** Verify via order status callback or polling.
22. **DON'T run live trading on your laptop.** It will close, sleep, run out of battery, lose Wi-Fi at the worst time. Cloud VM only.
23. **DON'T ignore data anomalies.** If a stock "moved 90% overnight" and the volume is normal, it's bad data, not a real move.
24. **DON'T forget to handle Indian market quirks**: ASM/GSM lists (price bands), circuit limits (5/10/20%), trading halts, special pre-open/post-close sessions, illiquid stocks with no fills.

### Strategy Selection
25. **DON'T trade strategies with capacities lower than your account size.** If a strategy only works with ₹1 cr but you have ₹10 cr, costs will eat profits.
26. **DON'T trade ML-based strategies you don't understand.** A random forest with 200 features is a black box; when it fails, you can't fix it. Prefer interpretable models.
27. **DON'T believe AI/ML will magically beat the market** without an economic rationale. Chan: *"financial predictive models based on AI ... inevitably performed miserably going forward."*
28. **DON'T trade illiquid stocks** (avg daily volume < ₹5 Cr). Slippage will dominate any edge.
29. **DON'T trade penny stocks under ₹50.** Bid-ask spreads are huge; manipulation is rampant.
30. **DON'T trade newly-listed stocks** in their first 6 months — no historical data, high volatility, weak signal.

### Sentiment / News
31. **DON'T act on a single news headline.** Aggregate; sentiment is statistical, not deterministic.
32. **DON'T trade earnings announcements** until you've specifically backtested an earnings strategy. They're nonlinear and option markets price them better than you can.
33. **DON'T scrape news sites without rate-limiting** — you'll get IP-banned and lose your data source.

### Mental
34. **DON'T tell friends "I have a profitable bot"** until you have 12 months live track record. They'll ask for tips, you'll feel pressure to perform, you'll deviate.
35. **DON'T watch live PnL throughout the day.** It's an emotion machine. Check at end of day.
36. **DON'T blame the algo when YOU disabled it / overrode it / didn't fund it.** Audit your *own* decisions, not the strategy's.
37. **DON'T quit your job to do this full-time** until you've made >2× your salary algorithmically for 12 consecutive months.

---

## The "If In Doubt" Rule

When facing any decision (deploy?, increase size?, trust this signal?):

```
if not_paper_traded_30_days:  return PAPER_TRADE_FIRST
if backtest_sharpe < 1.5:     return DO_NOT_DEPLOY
if max_drawdown > 15%:        return REDUCE_SIZE_OR_REJECT
if i_am_excited:              return WAIT_24_HOURS
if i_am_scared:               return REDUCE_SIZE
if losing_streak >= 3:        return PAUSE_AND_REVIEW
if just_changed_code:         return RUN_ALL_TESTS
```

---

## The Single Most Important Rule

> **You cannot win this game by trying to win it. You can only win by not losing.**
>
> — Roughly paraphrased from every legendary trader's autobiography.

Capital preservation > returns. Always.
