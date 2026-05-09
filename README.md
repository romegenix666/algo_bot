# Algo Bot — Indian Equities Algorithmic Trading

> Personal algorithmic trading bot for NSE/BSE. Built risk-first, paper-trading-first, money-second.

**Status:** Core pipeline implemented — data ingestion, universe, strategies, backtest, paper trading, optional Zerodha Kite (live / dry-run), sentiment, risk + circuit breakers, Telegram, Streamlit dashboard, scheduler, and tests. Operational hardening (30-day paper gauntlet, keys, cron) is on you before real capital.

---

## What this is

A modular, broker-agnostic algorithmic trading system for Indian equities, built to:

- Pick from a monthly universe of 500 liquid NSE stocks.
- Backtest multi-year data with realistic costs and validation helpers.
- Trade ranked signals via documented strategies (momentum, mean reversion, pairs, multi-factor, sentiment-augmented).
- Manage risk via Half-Kelly sizing, ATR-based stops, configurable circuit breakers (YAML-driven thresholds), and portfolio limits.
- Run paper trading (or sync + execute via Kite when `ALGO_MODE=live`).

This is not a get-rich-quick project. It is a discipline project that may, if executed well, produce sensible risk-adjusted returns with bounded drawdowns.

## Read these first (in order)

1. **`docs/PROJECT_PLAN.md`** — the master plan, architecture, phases.
2. **`docs/STRATEGY.md`** — algorithm details and Strategy Pattern design.
3. **`docs/REQUIREMENTS.md`** — accounts to open, libraries to install, capital needed.
4. **`docs/DOS_AND_DONTS.md`** — the rules we will not break.
5. **`docs/ROADMAP.md`** — phase-by-phase tasks and acceptance criteria.
6. **`docs/RESEARCH_PAPERS.md`** — curated supporting research.

## Quick start

```bash
# Python 3.11+
python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

cp .env.example .env
# Edit .env: ALGO_MODE, database URL, optional Kite / Telegram keys

# Initialise DB and fetch data (see scripts in scripts/)
python -m scripts.init_db
python -m scripts.fetch_history   # after universe is built

# Tests
pytest
```

**Paper vs live:** `ALGO_MODE=paper` (default) uses simulated execution and `data/paper/state.json`. `ALGO_MODE=live` (or `--broker kite`) uses Kite; use `--dry-run` to log orders without sending. Circuit breaker thresholds come from `config/default.yaml` under `risk` (e.g. `portfolio_max_drawdown`, `circuit_daily_loss_pause`).

```bash
python -m scripts.paper_trade
python -m scripts.paper_trade --as-of 2026-04-30
python -m scripts.paper_trade --broker kite --dry-run
```

## Project layout

```
Algo_bot/
├── docs/                       # Planning + research documents
├── src/
│   ├── data/                   # OHLCV + corporate actions ingestion
│   ├── universe/               # Monthly 500-stock selection
│   ├── features/               # Technical/fundamental/sentiment features
│   ├── strategies/             # Strategy Pattern implementations
│   ├── backtest/               # Vectorized backtester
│   ├── risk/                   # Position sizing + stops + circuit breakers
│   ├── sentiment/              # News sources + scoring
│   ├── orders/                 # Broker abstraction (Kite, paper, dry-run)
│   ├── monitor/                # Streamlit dashboard + Telegram alerts
│   └── utils/
├── notebooks/                  # Exploratory analysis
├── tests/                      # pytest
├── data/                       # OHLCV, sentiment, paper/live state (gitignored)
├── logs/                       # Runtime logs (gitignored)
├── config/                     # YAML configs per environment
├── scripts/                    # CLI entrypoints
├── requirements.txt
├── .env.example                # Template (copy to .env)
└── .gitignore
```

## The Hard Rules (read `docs/DOS_AND_DONTS.md` for the full list)

1. No real money before 30 days of paper trading.
2. No leverage / F&O for the first 12 months of profitable live trading.
3. No discretionary overrides without 24-hour cooling-off + journal entry.
4. Maximum 5% of net worth in this trading account, ever.
5. Capital preservation > returns. Always.

## Disclaimer

Algorithmic trading involves substantial risk. Past performance does not guarantee future returns. This software is for personal/educational use. The authors are not SEBI-registered investment advisors. Trade at your own risk.
