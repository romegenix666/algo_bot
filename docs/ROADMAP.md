# Roadmap — Phased Build & Acceptance Criteria

> Each phase has **specific tasks** and **acceptance criteria** (what must be true
> before the next phase starts). **Tasks** are now marked for **code in this repo**
> (`[x]`) vs **still on you** (`[ ]`) — accounts, capital, cloud, and multi-week
> paper runs are never “done” by code alone.

---

## Repository snapshot (implementation vs operations)

**Implemented in `src/`, `scripts/`, `config/`, tests**

| Area | Status |
|------|--------|
| Config + settings merge (`default` + `ALGO_MODE` overlay) | Done |
| Data: yfinance + NSE helpers, SQLite schema, refresh orchestrator, anomalies | Done |
| Universe selector + point-in-time replay (`src/universe/`) | Done |
| Backtest: vectorized engine, costs, metrics, Monte Carlo, lookahead audit, sensitivity | Done |
| Strategies: base + registry; momentum, mean reversion, pairs, multi-factor, sentiment_momentum, regime, others | Done |
| Risk: sizer (Half-Kelly + ATR stops), portfolio, manager, **YAML-driven** circuit breaker | Done |
| Orders: `Broker` ABC, paper broker, Kite, dry-run, router, live portfolio sync | Done |
| Sentiment: scrapers, storage (SQLite), aggregator, scorer (VADER + lazy FinBERT) | Done — *no separate `dedupe.py` / `finbert.py`; logic lives in storage + `scorer.py`* |
| Monitor: Telegram, Streamlit dashboard, APScheduler wrapper (`run_scheduler`) | Done |
| CLIs: `paper_trade`, `paper_replay`, `fetch_history`, `refresh_sentiment`, `kite_login`, backtest/universe demos | Done |

**Not implemented (or explicitly deferred)**

| Item | Notes |
|------|--------|
| `src/orders/upstox.py` | Not present; Kite + paper only |
| Standalone `src/sentiment/dedupe.py` | Optional enhancement |
| Dedicated backtest plotting module | Partially replaced by dashboard / notebooks |
| `ruff` / `mypy` clean on *entire* `scripts/` tree | A few legacy `type: ignore` warnings may remain |

**Operational (checklist for you, not the codebase)**

- Private Git remote, Zerodha **live** API subscription when you go live, Upstox backup account, 30+ day paper gauntlet on a VM, reconciliation discipline, capital deployment.

---

## Phase 0 — Setup (Week 1)

### Tasks
- [ ] `git init` / remote on GitHub (private) — *your account*
- [ ] Python 3.11+ venv (uv/poetry optional) — *your machine*
- [x] `requirements.txt` + `pyproject.toml` (ruff, mypy, pytest)
- [x] Folder structure (per `PROJECT_PLAN.md`)
- [x] `.env.example` + `.gitignore` (secrets not committed)
- [x] `ruff`, `mypy`, `.pre-commit-config.yaml` (+ `scripts/setup_dev.sh`)
- [ ] Open Zerodha account — *needed before funded live*
- [ ] Open Upstox developer account — *backup broker; no code yet*
- [ ] NewsAPI / other news keys — *optional per scraper config*
- [ ] Telegram bot via BotFather — *fill `.env`*
- [x] GitHub Actions CI (lint + test) — `.github/workflows/ci.yml`
- [ ] Feynman / research API — *optional*

### Acceptance criteria
- [x] `pytest` runs with a broad suite (see `tests/`)
- [x] `ruff` + `mypy` wired in CI / pre-commit (*local runs: your responsibility*)
- [ ] Pre-commit installed on your clone (`pre-commit install`) — *one-time local*
- [x] `.env` not tracked (verify with `git check-ignore .env` or templates only)

---

## Phase 1 — Data Layer & Universe Selection (Week 2)

### Tasks
- [x] `src/data/yfinance_client.py` — daily OHLCV
- [x] `src/data/nse_client.py` (`nsepython`) — corporate actions, sector mapping
- [x] SQLite schema (tickers, prices, corporate actions, anomalies, etc.)
- [x] `src/universe/selector.py` — filters aligned with config (cap, turnover, price band)
- [x] Point-in-time universe replay — `src/universe/replay.py`
- [x] Daily refresh pipeline — `src/data/refresh.py` + `scripts/fetch_history.py`; scheduler hooks via `scripts/run_scheduler.py` / `src/monitor/scheduler.py`
- [x] Data anomaly detector — `src/data/anomaly.py`

### Acceptance criteria
- [ ] *You* populate ≥5y history for ~500 names (run fetch jobs; depends on DB)
- [ ] Current-month universe CSV on disk — *run `build_universe` workflow*
- [x] Anomaly + loader tests exist (`tests/`)
- [x] Loader / anomaly unit coverage in test suite

---

## Phase 2 — Backtest Engine (Week 3–4)

### Tasks
- [x] Vectorized backtester — `src/backtest/engine.py` + costs (`costs.py`)
- [x] Walk-forward support (config + engine paths)
- [x] Performance metrics — `src/backtest/metrics.py`
- [x] Monte Carlo — `src/backtest/monte_carlo.py`
- [ ] Standalone plotting package (equity, DD heatmap) — *use dashboard/notebooks or add later*
- [x] Look-ahead-bias auditor — `src/backtest/lookahead.py`
- [x] Sensitivity runner — `src/backtest/sensitivity.py`

### Acceptance criteria
- [ ] Nifty buy-and-hold baseline vs TRI — *run benchmark script with your data*
- [x] Look-ahead auditor covered by tests
- [x] Cost model tests / hand-check fixtures where present
- [x] Metrics + MC + sensitivity tests in `tests/`

---

## Phase 3 — Strategy Implementations v1 (Week 5–6)

### Tasks
- [x] `src/strategies/base.py` — Strategy interface + signals
- [x] `src/strategies/momentum.py`
- [x] `src/strategies/mean_reversion.py`
- [x] `src/strategies/pairs.py`
- [x] `src/strategies/multi_factor.py`
- [x] `src/strategies/sentiment_momentum.py`
- [x] Regime detection — `src/strategies/regime.py` + selector/registry
- [x] Backtest / benchmark harness — `scripts/benchmark_strategies.py`, `src/backtest/benchmark.py`
- [x] Walk-forward + sensitivity + Monte Carlo + lookahead — modules wired; *re-run on your data*

### Acceptance criteria
*(Statistical gates below are **targets** — validate on your OOS windows; not auto-enforced in CI.)*

For each strategy:
- [ ] OOS Sharpe ≥ 1.0 after costs — *empirical*
- [ ] Max DD ≤ 20% after costs — *empirical*
- [ ] Sensitivity: profitable in ≥80% of ±20% perturbations — *run sensitivity*
- [ ] No look-ahead — *audit + tests*
- [ ] ≥100 OOS trades — *empirical*

Portfolio of strategies:
- [ ] Sharpe ≥ 1.5, max DD ≤ 15%, pairwise corr < 0.6 — *empirical*

---

## Phase 4 — Sentiment Pipeline (Week 7)

### Tasks
- [x] News / RSS ingestion — `src/sentiment/scrapers.py` (*paths differ from original “scrapers/” folder spec*)
- [ ] Dedicated `dedupe.py` — *optional; improve if duplicate articles bite*
- [x] FinBERT-capable scoring — `src/sentiment/scorer.py` (lazy model load)
- [x] Aggregation — `src/sentiment/aggregator.py` + `storage.py`
- [ ] `llm_curator.py` — *not built*
- [x] Daily refresh job — `scripts/refresh_sentiment.py` + scheduler slot
- [ ] Output as daily parquet under `data/sentiment/` — *currently SQLite-first; parquet optional*

### Acceptance criteria
- [ ] ≥30 days backfilled sentiment — *run jobs*
- [ ] FinBERT vs manual spot-check — *your holdout*
- [ ] Sentiment strategy vs plain momentum — *backtest compare*
- [x] Throttling / polite defaults in scraper code — *respect site ToS in production*

---

## Phase 5 — Risk Manager & Order Executor (Week 8)

### Tasks
- [x] `src/risk/sizer.py` — Half-Kelly + ATR stops
- [x] Stop logic — sizing + `paper_trade` ATR exits (no separate `stops.py` file)
- [x] `src/risk/circuit_breaker.py` — halts, daily pause, 5d throttle, strategy streaks; **thresholds from YAML**
- [x] Concentration / caps — `src/risk/manager.py` (`RiskLimits`)
- [x] Abstract broker — `src/orders/base.py` (*roadmap name `broker.py`*)
- [x] `src/orders/paper.py`
- [x] `src/orders/kite.py` + `live_sync.py` + `dry_run.py`
- [ ] `src/orders/upstox.py`
- [x] Idempotent refresh / de-duplicated state patterns — *data refresh; order idempotency relies on broker + careful retries — tighten as needed*
- [x] Kill-switch CLI — `python -m scripts.paper_trade --kill-switch`

### Acceptance criteria
- [x] Risk + router tests stress rejections / caps (`tests/test_risk_manager.py`, etc.)
- [x] Paper broker + full-day path exercised in tests / replay harness
- [ ] Kite **sandbox** place/cancel/modify — *your API keys + manual verification*
- [x] Kill-switch path closes positions in paper driver (see `paper_trade`)

---

## Phase 6 — Paper Trading (Week 9–12) — **THE 30-DAY GAUNTLET**

### Tasks
- [ ] Deploy on cloud VM (Mumbai region recommended) — *ops*
- [x] Streamlit dashboard code — `src/monitor/dashboard.py` + `scripts/run_dashboard.sh`
- [x] Telegram alerts — `src/monitor/telegram.py` wired in `paper_trade`
- [x] Daily PnL summary hook — end-of-run notifier
- [ ] Weekly self-review / journal — *discipline*

### Acceptance criteria — STRICT
*(None of these are “green in CI” — they require a real paper run.)*

- [ ] **30+ trading days** continuous paper (no overrides)
- [ ] Paper Sharpe ≥ 1.0 (lower bar than backtest)
- [ ] Paper max DD ≤ 18%
- [ ] Paper vs backtest within ~25% on Sharpe / DD
- [ ] Zero unexplained errors (every incident root-caused)
- [ ] Breakers + stops behaved when hit
- [ ] Overrides documented (ideally none)

**If any criterion fails: do NOT proceed. Debug, fix, restart the 30 days.**

---

## Phase 7 — Demo (Broker Sandbox) (Week 13)

### Tasks
- [ ] Kite Connect sandbox / demo round-trip — *your keys*
- [ ] Reconcile broker log vs internal log
- [ ] Corner cases: open, ASM/GSM, halts, partial fills — *manual QA*

### Acceptance criteria
- [ ] 100% log match on demo
- [ ] ≥5 days all strategies without errors
- [ ] Kill-switch in demo closes all, no orphans

---

## Phase 8 — Live (Small Capital) (Week 14+)

### Tasks
- [ ] Kite Connect paid subscription when you commit
- [ ] Fund ₹50k–1L (cap per `DOS_AND_DONTS`)
- [ ] `ALGO_MODE=live` + `--broker kite` (no `--dry-run`)
- [ ] Two-week watch period
- [ ] Daily contract-note reconciliation
- [ ] Monthly tax / PnL export — *script TBD or manual*

### Acceptance criteria for **scaling beyond ₹1L**
- [ ] 3 consecutive profitable months
- [ ] Live Sharpe ≥ ~1.0
- [ ] No catastrophic risk failures
- [ ] Overrides documented

---

## Phase 9 — Live (Scaled) (Month 4+)

Scale capital incrementally: ₹1L → ₹3L → ₹5L → larger only after each tier proves itself for 2+ months.

**Hard caps until 12 months of live track record:**
- No leverage / no margin / no F&O.
- No discretionary overrides without 24-hour cooling-off.
- Maximum 5% of net worth in algo trading account.

---

## Ongoing — Maintenance & Research

- Monthly: review performance vs backtest expectation. Investigate divergence.
- Quarterly: re-run universe selection, re-fit parameters via walk-forward.
- Quarterly: archive 1 strategy that isn't performing; research 1 new strategy.
- Annually: re-read Chan + Kakushadze. Update the dos/donts based on hard lessons.

---

## Anti-Goals (things we will explicitly NOT do)

- Trade options or futures in any form before 12 months of profitable equity track record.
- Trade intraday (sub-day holding period) before Phase 9. Daily bars only.
- Use leverage / margin / MTF until 12 months of profitable live trading.
- Run the bot from a laptop. Cloud only.
- Promise returns to anyone. Not even ourselves.
- Take on outside investors / friends' money. Ever, until SEBI-registered.
