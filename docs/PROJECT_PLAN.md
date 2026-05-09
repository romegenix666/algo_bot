# Algo Bot — Master Project Plan

> **Goal:** Personal algorithmic trading bot for Indian equities (NSE/BSE).
> Pick 500 candidate stocks → rank using 5-year backtest → trade top 5 with a
> minimum-risk strategy and accurate, volatility-adjusted stop-loss.
> Demo / paper trading first. Real money only after the system has proven
> itself for at least 30 trading days in paper mode.

---

## 1. Mission Statement

Build a **disciplined, risk-first** Indian equity trading bot that:

1. Survives drawdowns (capital preservation > maximizing returns).
2. Trades only when statistical edge + sentiment confirm.
3. Stays brutally honest with backtests (no look-ahead, no survivorship-bias-driven hopium).
4. Is built modularly using the **Strategy Pattern** so we can swap algos without rewriting plumbing.
5. Is testable end-to-end on paper before a single rupee of real capital is risked.

---

## 2. The "Earn Money" Reality Check (read this twice)

From the books we ingested (Chan, Kakushadze & Serur, the CFTC IRL paper) and decades of public quant research:

- **Most retail algos lose money** because of: ignored transaction costs, overfit backtests, look-ahead bias, survivorship bias, over-leverage, and emotional override of stop-losses.
- "Picking 5 stocks that will make a lot of money" is **not** a guarantee — it's a *probabilistic edge*. We're targeting ~15–25% CAGR with <15% max drawdown, **not** "doubling money in a month".
- A high **Sharpe ratio** (>1.5) with smaller absolute return is mathematically better than a high-return / high-volatility strategy because we can lever it up safely later.
- **Never** trade a strategy that has not been paper-traded for at least 30 days post-backtest.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Algo Bot System                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────┐   ┌────────────┐   ┌─────────────┐   ┌───────────┐ │
│  │   Data     │──▶│ Universe   │──▶│  Feature    │──▶│ Strategy  │ │
│  │ Ingestion  │   │ Selection  │   │ Engineering │   │  Engine   │ │
│  │ (yfinance, │   │ (500 picks)│   │ (technical, │   │ (Strategy │ │
│  │  nsepy,    │   │            │   │  fundamental│   │  Pattern) │ │
│  │  Kite API) │   │            │   │  sentiment) │   │           │ │
│  └────────────┘   └────────────┘   └─────────────┘   └─────┬─────┘ │
│                                                            │       │
│  ┌────────────┐   ┌────────────┐   ┌─────────────┐   ┌────▼─────┐ │
│  │ Live       │◀──│  Order     │◀──│  Risk       │◀──│ Backtest │ │
│  │ Broker     │   │  Manager   │   │  Manager    │   │ Engine   │ │
│  │ (Kite/     │   │ (smart     │   │ (Kelly,     │   │ (5-yr    │ │
│  │  Upstox)   │   │  routing)  │   │  ATR stops) │   │  vector- │ │
│  └────────────┘   └────────────┘   └─────────────┘   │  ized)   │ │
│                                                       └──────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Sentiment Layer (parallel pipeline)                        │   │
│  │  News scraper (Moneycontrol, ET, LiveMint) + FinBERT/LLM    │   │
│  │  → daily sentiment score per ticker → feature for strategy │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Monitoring (Streamlit/Grafana dashboard, Telegram alerts)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Breakdown

### 4.1 Data Layer (`src/data/`)
- **Free historical data:** `yfinance` (5+ yrs daily, has `.NS` for NSE), `nsepy` / `nsepython` (NSE official).
- **Live data (paper):** `yfinance` 15-min delayed.
- **Live data (real):** Zerodha Kite Connect (₹2000/month) **or** Upstox API (free) **or** Angel SmartAPI (free).
- **Storage:** SQLite for now (Postgres if we scale). Parquet files for OHLCV bars.
- **Refresh cadence:** Daily after market close (16:00 IST).

### 4.2 Universe Selection (`src/universe/`)
Pick 500 candidate stocks **monthly** using:
- Market-cap filter: top 500 by free-float market cap (≈ Nifty 500).
- Liquidity filter: average daily turnover > ₹5 Cr (last 30 days).
- Price filter: ₹50 ≤ price ≤ ₹10,000 (avoid penny stocks & illiquid high-priced).
- Survivorship-bias guard: include delisted/suspended in **historical** backtests.

Output: `data/universe/YYYY-MM.csv` with 500 tickers + metadata.

### 4.3 Strategy Engine (`src/strategies/`)
Uses the **Strategy design pattern**. Each strategy implements:
```python
class Strategy(ABC):
    def generate_signals(self, prices, features, sentiment) -> Signals
    def position_sizing(self, signals, portfolio, risk_params) -> Positions
    def exit_rules(self, position, market_state) -> ExitDecision
```

Initial strategies (detailed in `STRATEGY.md`):
1. **Momentum (12-1)** — Jegadeesh & Titman, modified for Indian markets.
2. **Mean Reversion (Bollinger + RSI)** — for sideways markets.
3. **Pairs Trading (cointegration)** — market-neutral, low drawdown.
4. **Multi-factor (Value + Momentum + Quality + LowVol)** — robust core.
5. **Sentiment-augmented Momentum** — only buy if news sentiment ≥ neutral.

A **regime detector** picks which strategy to run based on volatility/trend.

### 4.4 Risk Manager (`src/risk/`)
- **Position sizing:** Half-Kelly fraction (full Kelly is too aggressive).
- **Per-trade risk cap:** 1% of equity (max).
- **Per-strategy risk cap:** 5% of equity per day.
- **Portfolio drawdown circuit-breaker:** auto-halt if total DD > 12%.
- **Stop-loss:** ATR-based (2× ATR(14) for momentum, 1.5× for mean-reversion).
- **Trailing stop:** activates after 1.5× ATR profit, locks 50% of gain.
- **Max position concentration:** 20% of portfolio in any one stock.

### 4.5 Backtest Engine (`src/backtest/`)
- **Vectorized** (pandas/numpy) for speed; event-driven (`backtrader` / custom) for realism on top-5.
- **Walk-forward validation:** train on 2020–2023, test on 2024–2025, *no* peeking.
- **Realistic costs:** 0.03% brokerage + 0.025% STT + 0.00325% exchange + GST + slippage (5 bps).
- **Performance metrics:** CAGR, Sharpe, Sortino, max drawdown, max DD duration, Calmar, hit rate, profit factor.
- **Monte Carlo:** 1000 bootstrapped equity curves to estimate worst-case.

### 4.6 Sentiment Layer (`src/sentiment/`)
- Sources: Moneycontrol, Economic Times, LiveMint, Reddit (r/IndianStockMarket), Twitter/X (Nitter scrape).
- Pipeline: scrape → dedupe → FinBERT scoring → aggregate per-ticker daily score in `[-1, +1]`.
- Optional LLM layer (Claude / GPT / local Llama) for nuanced summary.
- Used as **feature** (not standalone signal) — sentiment alone is too noisy.

### 4.7 Order Manager (`src/orders/`)
- Idempotent order submission with retry + dedupe.
- Three modes: `paper` (in-memory ledger), `demo` (broker sandbox), `live`.
- Broker abstraction layer so we can swap Kite ↔ Upstox ↔ Angel without strategy changes.

### 4.8 Monitoring (`src/monitor/`)
- Streamlit dashboard: live PnL, positions, drawdown, signals.
- Telegram bot: alerts on fills, stop-losses, circuit-breaker trips, daily summary.
- Logs: structured JSON to `logs/` + rotation.

---

## 5. Phases (see `ROADMAP.md` for granular tasks)

| Phase | Duration | Goal | Capital |
|-------|----------|------|---------|
| 0. Setup | Week 1 | Repo, env, broker accounts (paper), data pipeline | ₹0 |
| 1. Data + Universe | Week 2 | 5-yr OHLCV stored, monthly 500-stock list working | ₹0 |
| 2. Backtest Engine | Week 3–4 | Vectorized backtester w/ realistic costs, walk-forward | ₹0 |
| 3. Strategies v1 | Week 5–6 | All 5 strategies coded + backtested + sensitivity analysis | ₹0 |
| 4. Sentiment Pipeline | Week 7 | Daily sentiment scores per ticker integrated as feature | ₹0 |
| 5. Risk + Orders | Week 8 | Risk manager + paper trading executor | ₹0 |
| 6. Paper Trading | Week 9–12 | **30+ days live paper trading**. No real money. | ₹0 |
| 7. Demo (broker sandbox) | Week 13 | Connect to Kite/Upstox sandbox, validate fills | ₹0 |
| 8. Live (small) | Week 14+ | ₹50k–1L real capital, scale only after 3 profitable months | ₹50k–1L |

**Hard rule:** No phase advances until the previous phase passes its acceptance criteria (defined in `ROADMAP.md`).

---

## 6. Success Criteria

A strategy is **graduated to live trading** only when:

| Metric | Threshold |
|--------|-----------|
| Backtest Sharpe (out-of-sample) | ≥ 1.5 |
| Max drawdown (backtest) | ≤ 15% |
| Max DD duration | ≤ 90 trading days |
| Paper-trading Sharpe (30+ days) | ≥ 1.0 (lower OK because sample is small) |
| Paper vs backtest divergence | within 25% on key metrics |
| Hit rate | ≥ 45% (combined with profit factor ≥ 1.5) |
| Sensitivity (parameter ±20%) | profitable in ≥ 80% of variations |

---

## 7. What Could Kill This Project (and our defenses)

| Risk | Defense |
|------|---------|
| Overfitting backtests | Walk-forward, ≤5 parameters per strategy, sensitivity analysis, paper-trading buffer |
| Survivorship bias | Use point-in-time universe, include delisted in historical |
| Black swan | Circuit-breakers, max position size, no naked options |
| Broker API outage | Idempotent orders, graceful degradation, manual override panel |
| Data error | Anomaly detection on incoming bars, fallback data source |
| Overconfidence after paper wins | Hard rule: 30 days paper minimum, then start with ₹50k max |
| Tax/compliance | Track every trade, generate P&L report monthly, consult CA for STCG/LTCG |

---

## 8. Project Layout (proposed)

```
Algo_bot/
├── docs/                       # Planning .md files (this file etc.)
├── src/
│   ├── data/                   # Ingestion + storage
│   ├── universe/               # 500-stock selection
│   ├── features/               # Technical / fundamental / sentiment features
│   ├── strategies/             # Strategy pattern implementations
│   │   ├── base.py
│   │   ├── momentum.py
│   │   ├── mean_reversion.py
│   │   ├── pairs.py
│   │   ├── multi_factor.py
│   │   └── sentiment_momentum.py
│   ├── backtest/               # Backtest engine
│   ├── risk/                   # Position sizing + stops
│   ├── sentiment/              # News + scoring
│   ├── orders/                 # Broker abstraction
│   ├── monitor/                # Dashboard + alerts
│   └── utils/
├── notebooks/                  # Exploratory analysis (Jupyter)
├── tests/                      # Pytest unit + integration
├── data/                       # OHLCV, universe, features (gitignored)
├── logs/                       # Runtime logs (gitignored)
├── config/                     # YAML configs per environment
├── scripts/                    # CLI entrypoints
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 9. Next Immediate Step

Read `STRATEGY.md` for the algorithm details, `REQUIREMENTS.md` for what to install,
`DOS_AND_DONTS.md` for the rules we will *never* break, and `ROADMAP.md` for the
checklist of tasks for Phase 0 (Setup).
