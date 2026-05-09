# Strategy Design

> Source-grounded strategy choices. Every algorithm here cites either the books
> we ingested (Chan, Kakushadze & Serur) or peer-reviewed research (see
> `RESEARCH_PAPERS.md`). No "ChatGPT-imagined" indicators.

---

## 1. Why Strategy Pattern?

We code each algorithm as a class implementing a common `Strategy` interface.
A `StrategySelector` (the regime detector) decides which strategy(ies) is/are
active at any given time. This means:

- New strategy = new class, zero changes to data/risk/order pipeline.
- Backtester treats every strategy identically.
- We can run a **portfolio of strategies** (each on a slice of capital) and
  the bad ones get auto-deallocated by the risk manager.

```python
# src/strategies/base.py (sketch)
from abc import ABC, abstractmethod

class Strategy(ABC):
    name: str
    timeframe: str   # "1d", "15m", etc.

    @abstractmethod
    def required_features(self) -> list[str]: ...

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,           # OHLCV per ticker, indexed by date
        features: pd.DataFrame,         # tech + fundamental
        sentiment: pd.DataFrame,        # daily sentiment score per ticker
    ) -> pd.DataFrame:                  # columns: ticker, side, conviction
        ...

    @abstractmethod
    def position_size(self, signal, equity, risk_params) -> float: ...

    @abstractmethod
    def exit_rules(self, position, market_state) -> ExitDecision: ...
```

---

## 2. Strategies (initial set of 5)

### 2.1 Cross-Sectional Momentum (12-1)
**Source:** Jegadeesh & Titman (1993); Kakushadze & Serur §3.1; Chan §7.

**Logic:** Rank stocks by past 12-month return *excluding* the most recent month
(the "skip month" avoids 1-month reversal). Long top decile, short bottom decile
(or long-only top decile if no shorting).

**Why for India:** Momentum is the most-replicated anomaly globally and persists
in NSE 500 (multiple Indian academic papers confirm).

**Params (≤4):**
- Lookback: `12 months`
- Skip: `1 month`
- Rebalance: `monthly`
- Top-N: `5` for our concentrated portfolio

**Exits:** Monthly rebalance. ATR-trailing stop in between (locks profit on bad months).

**Expected stats (5-yr Indian backtest, before costs):** Sharpe ≈ 1.2–1.6, max DD 18–22%.

---

### 2.2 Mean Reversion — Bollinger Band + RSI Filter
**Source:** Chan §7 ("Mean-Reverting versus Momentum"); Kakushadze §3.9.

**Logic:** When price drops > 2σ below 20-day MA *and* RSI(14) < 30 *and* the stock
is in a known mean-reverting regime → buy. Exit when price returns to MA or
RSI > 50.

**Why:** Chan: *"Mean-reverting regimes are more prevalent than trending regimes"*.
Mean-reversion has the highest Sharpe among simple strategies.

**Critical filter:** Only apply to stocks that pass an Augmented Dickey-Fuller
stationarity test on the spread vs. its index — most stocks are *not*
mean-reverting in price (only spreads are).

**Params (≤5):** BB period (20), σ multiplier (2.0), RSI period (14),
RSI threshold (30), exit RSI (50).

**Exits:** Time-based stop (max 10 trading days). No trailing stop —
mean-reversion does not benefit from it (Chan §7).

**Expected stats:** Sharpe 1.5–2.0, max DD 8–12%, hit rate 55–65%.

---

### 2.3 Pairs Trading (Cointegration)
**Source:** Chan §7 (cointegration); Kakushadze §3.8; Avellaneda & Lee (2010).

**Logic:**
1. Run cointegration tests (Engle-Granger) on all NSE pairs within same sector.
2. For pairs with p < 0.05, compute the spread `S = log(A) - β·log(B)`.
3. Trade the z-score of the spread: short spread when z > +2, long when z < -2,
   exit at z = 0.

**Why:** Market-neutral → near-zero correlation with Nifty → drawdowns stay
small even in crashes. Sharpe per pair is moderate but **portfolio Sharpe**
across many uncorrelated pairs is excellent.

**Indian candidates:** HDFCBANK/ICICIBANK, RELIANCE/ONGC, TCS/INFY, MARUTI/M&M, etc.

**Params (≤4):** Lookback for cointegration (1 yr), z-entry (2.0), z-exit (0.5),
half-life of mean-reversion (Ornstein-Uhlenbeck fit).

**Exits:** z-score crosses 0, OR cointegration breaks (rolling p > 0.10), OR
30-day timeout.

**Expected stats:** Sharpe 1.8–2.5, max DD 6–10%.

---

### 2.4 Multi-Factor Portfolio (Value + Momentum + Quality + LowVol)
**Source:** Fama & French (1993, 2015); Kakushadze §3.6; Asness et al. (2013).

**Logic:** For each stock, compute a Z-score on each factor:
- **Value:** P/E, P/B, EV/EBITDA (lower is better)
- **Momentum:** 12-1 return (higher is better)
- **Quality:** ROE, debt/equity, earnings stability (higher ROE / lower D/E is better)
- **Low Vol:** trailing 90-day realized volatility (lower is better)

Composite score = mean of the four Z-scores. Long top-quintile, optionally short
bottom-quintile.

**Why:** Each factor is independently documented (decades of research). Combining
them diversifies factor risk. Low-Vol + Quality together reduce drawdowns
significantly (the "defensive" tilt).

**Params (few):** Factor weights (default equal), rebalance freq (quarterly),
top-quintile cutoff (20%).

**Exits:** Quarterly rebalance. Stop-loss = -8% from cost basis (rare hit, as
this strategy holds quality names).

**Expected stats:** Sharpe 0.9–1.3 (lower than mean-rev but **massive capacity**
— scales to crores), max DD 15–20%, low turnover (~30%/yr → tax efficient).

---

### 2.5 Sentiment-Augmented Momentum
**Source:** Tetlock (2007); Loughran & McDonald (2011); Kakushadze §3.20 (alpha combos).

**Logic:** Take signals from strategy 2.1 (momentum). Filter out any long signal
where the rolling 7-day news sentiment for that ticker is negative.
Filter out any short signal where sentiment is positive.

**Why:** News sentiment alone is too noisy to trade, but as a **filter** on an
existing edge it removes catastrophic shocks (e.g., fraud allegations, regulatory
news) that pure momentum would walk straight into.

**Sentiment data flow:**
```
Moneycontrol/ET/LiveMint scrape (hourly)
         │
         ▼
Dedupe + clean → store raw articles
         │
         ▼
FinBERT scoring (HuggingFace `ProsusAI/finbert`)
         │
         ▼
Aggregate per ticker per day → score ∈ [-1, +1]
         │
         ▼
7-day EMA → final sentiment feature
```

**LLM enhancement (optional, phase 7+):** Pass top headlines to Claude/GPT for
nuanced "what does this mean for the stock" reasoning. Use only as a tiebreaker.

**Expected stats:** Sharpe of underlying momentum + 0.2–0.4, drawdowns reduced ~20%.

---

## 3. Regime Detection — When to Run Which Strategy

A simple but battle-tested regime classifier:

```
Compute on Nifty 50 (rolling 60 days):
  realized_vol  = std of daily log returns × √252
  trend_score   = (price_today - SMA_200) / ATR_60

Regime decisions:
  if realized_vol > 25%  AND |trend_score| > 1.5  → "trending high-vol"
       → Allocate 60% to Momentum, 20% to Multi-Factor, 20% cash
  elif realized_vol > 25%                           → "choppy high-vol"
       → Allocate 50% to Pairs, 50% cash
  elif |trend_score| > 1.0                          → "trending low-vol"
       → Allocate 50% Momentum, 50% Multi-Factor
  else                                              → "range-bound low-vol"
       → Allocate 50% Mean Reversion, 30% Pairs, 20% Multi-Factor
```

These weights are themselves parameters subject to walk-forward validation.

---

## 4. Risk Management — The Real Edge

Most retail bots die not from bad strategies but from **bad risk management**.

### 4.1 Position Sizing — Half-Kelly

Full Kelly: `f* = (p·b - q) / b` where p=win prob, b=win/loss ratio.
**We use half-Kelly** because (a) we don't know `p` and `b` precisely and
(b) full-Kelly has unacceptable drawdowns (Chan §6).

```python
def position_size(signal, equity, win_rate, win_loss_ratio):
    full_kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    half_kelly = max(0, full_kelly / 2)
    half_kelly = min(half_kelly, 0.20)   # never > 20% in one stock
    return equity * half_kelly * signal.conviction
```

### 4.2 Stop-Loss — ATR-Based (Volatility-Adjusted)

Fixed-percent stops are wrong because volatility differs across stocks.

```
ATR(14) = Average True Range over 14 days
Initial stop  = entry_price - k * ATR(14)
   k = 2.0 for momentum
   k = 1.5 for mean-reversion
   k = 1.0 for pairs (tighter, since spread is bounded)

Trailing stop activates after price moves +1.5*ATR in our favor:
   trail_stop = max(trail_stop, current_price - k * ATR(14))
```

### 4.3 Portfolio-Level Circuit Breakers

| Trigger | Action |
|---------|--------|
| Daily loss > 3% of equity | Stop new entries for the day |
| 5-day rolling loss > 7% | Cut all positions to 50% size |
| Drawdown > 12% from peak | Halt all new trading; manual review |
| 3 consecutive stop-loss hits in same strategy | Disable that strategy for 5 days |

### 4.4 Concentration Limits

- Max 20% of equity in any single stock.
- Max 35% of equity in any single sector.
- Max 60% net long exposure (when using shorts/hedges).

---

## 5. The Top-5 Picker

Once strategies generate ranked candidates from the 500-stock universe:

```
For each strategy s in active_strategies:
    ranked_signals = s.generate_signals(...)            # all candidates
    s_top5 = ranked_signals.head(5)
    
final_top5 = blend(strategy_top5_lists, weights=regime_weights)
            .deduplicate()
            .filter(risk_constraints)
            .head(5)
```

The "top 5 stocks that will make a lot of money" emerges from this process,
**not** from a single magical model.

---

## 6. Backtest Discipline

Every strategy MUST pass these checks before paper trading:

1. **Walk-forward validation** (rolling, 3-yr train / 6-mo test).
2. **Monte Carlo bootstrap** of trade returns → 95% CI on Sharpe and max DD.
3. **Sensitivity test** — perturb each parameter ±20%, performance must stay
   in same ballpark.
4. **Look-ahead audit** (the "truncate-and-rerun" check from Chan §3).
5. **Transaction costs and slippage** included realistically (5 bps + brokerage).
6. **Survivorship bias** check — re-run with delisted stocks included.
7. **Regime stress test** — does it survive 2008/2020 simulation?

If a strategy fails any of these, it does not graduate.

---

## 7. References (numbered for traceability)

See `RESEARCH_PAPERS.md` for the full curated list of supporting research.
