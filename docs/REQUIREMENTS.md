# System Requirements & Toolchain

> **Mac (Apple Silicon) is the assumed dev machine.** Cloud server (DigitalOcean / AWS Lightsail / Hetzner) for live deployment in Phase 7+.

---

## 1. What You Need (the human checklist)

### 1.1 Accounts to open *now* (all free for paper)

| Service | Why | Cost | Link |
|---------|-----|------|------|
| **Zerodha Kite Connect** | Real broker API, best Indian retail liquidity. **Required for live trading**. | ₹2,000/month *only* when going live. Demo/historical is free. | [kite.trade](https://kite.trade) |
| **Upstox Developer** | Free alternative API, good for paper/dev. | Free | [upstox.com/developer](https://upstox.com/developer) |
| **Angel One SmartAPI** | Free API alternative, decent docs. | Free | [smartapi.angelbroking.com](https://smartapi.angelbroking.com) |
| **Yahoo Finance** | Free historical OHLCV (via `yfinance`). Has Indian stocks with `.NS` suffix. | Free | n/a |
| **NSE India** | Official source via `nsepython`. Bhavcopy + corporate actions. | Free | [nseindia.com](https://www.nseindia.com) |
| **NewsAPI** | Free 100 req/day for news. | Free tier | [newsapi.org](https://newsapi.org) |
| **HuggingFace** | FinBERT model + datasets. | Free | [huggingface.co](https://huggingface.co) |
| **GitHub** | Repo + secret manager (Actions). | Free | [github.com](https://github.com) |
| **Telegram Bot** | Cheap reliable alerting. | Free | BotFather |

### 1.2 Optional / Phase-dependent

| Service | Why | Cost |
|---------|-----|------|
| Anthropic / OpenAI API | LLM-curated sentiment & research summaries (Feynman needs this) | ~$5–20/month |
| AlphaXiv subscription | Research-paper firehose for Feynman | Free (signup) |
| Hetzner CPX21 / DO droplet | 24/7 cloud runner | ~₹400–800/month |
| Postgres on Supabase | If we outgrow SQLite | Free tier or ~₹500/month |
| Streak / Sensibull | Reference / sanity-check our signals | Free tier |

### 1.3 What to NOT pay for (yet)

- Bloomberg Terminal — overkill, costs ₹2L/month.
- Tickertape Pro / Smallcase Premium — we are *building* this, not consuming pre-baked signals.
- Algorithmic-trading "courses" on Udemy — the books we have already cover it.
- Paid Telegram tip channels — actively harmful (most are pump-and-dump).

---

## 2. Software Stack

### 2.1 Language: **Python 3.11**

Yes, we know C++/Rust would be faster. But:
- 99% of quant libraries are Python.
- Vectorized NumPy/pandas is fast enough for daily-bar strategies.
- Time-to-market matters more than 5 ms latency for our timeframes.
- We can rewrite hot paths in Numba/Cython if needed.

### 2.2 Key Libraries (will be in `requirements.txt`)

| Category | Library | Why |
|----------|---------|-----|
| Data | `yfinance`, `nsepython`, `pandas-datareader` | Free Indian stock data |
| Broker | `kiteconnect`, `upstox-python-sdk` | Official broker SDKs |
| DataFrames | `pandas`, `polars` (later) | Standard + 10× faster alternative |
| Numerics | `numpy`, `scipy`, `numba` | Vectorization + JIT |
| Backtest | `vectorbt` (or custom), `backtesting.py` | Fast vectorized backtester |
| Stats | `statsmodels` | ADF tests, cointegration, regressions |
| ML | `scikit-learn`, `lightgbm` | Classical ML for feature ranking |
| Risk | (custom in `src/risk/`) | Kelly, ATR stops, etc. |
| Sentiment | `transformers`, `torch`, `vaderSentiment` | FinBERT + rule-based fallback |
| Scraping | `httpx`, `beautifulsoup4`, `feedparser`, `playwright` | News scraping |
| LLM | `anthropic`, `openai`, `langchain` (optional) | Sentiment curation |
| Storage | `sqlalchemy`, `duckdb`, `pyarrow` | SQLite + Parquet |
| Scheduling | `apscheduler`, `prefect` (later) | Cron-like daily jobs |
| Dashboard | `streamlit`, `plotly` | Live PnL + positions |
| Alerts | `python-telegram-bot` | Telegram notifications |
| Testing | `pytest`, `pytest-cov`, `hypothesis` | Unit + property-based tests |
| Quality | `ruff`, `mypy`, `pre-commit` | Linting + type-checking |
| Config | `pydantic`, `pyyaml` | Type-safe config |
| Logging | `structlog`, `loguru` | Structured logs |

### 2.3 Dev Tools
- VS Code / Cursor (the latter, since you're using it).
- `uv` or `poetry` for dependency management (we'll use `uv` — it's faster).
- `git` (you should `git init` this repo soon — it isn't one yet).
- Pre-commit hooks for `ruff`/`mypy`.

---

## 3. Hardware

### Dev (right now)
- Your MacBook is fine. ≥16 GB RAM strongly preferred when we hit ML training.

### Live deployment (Phase 7+)
- **Cloud VM** in Mumbai region (lower latency to NSE) — Hetzner CCX13 or DO droplet (Mumbai).
- 2 vCPU, 4 GB RAM, 80 GB SSD = enough for daily-bar strategies.
- 99.9% uptime SLA.
- Snapshot before every market open.

---

## 4. Data Requirements

### 4.1 Historical (5-year backtest)
- **OHLCV daily** for ~500 stocks × 5 years = ~625K bars. Trivial size (~50 MB Parquet).
- **Corporate actions** — splits, dividends, bonus, mergers (NSE bhavcopy).
- **Index data** — Nifty 50, Nifty 500, sectoral indices.
- **Fundamentals** — quarterly P/E, P/B, ROE, D/E, etc. *Free sources are limited*; we may need:
  - `screener.in` scraping (legal grey area, do politely w/ rate limits).
  - `tickertape` API (unofficial).
  - For MVP: P/E + market cap from yfinance is OK.

### 4.2 Live (during trading)
- Real-time quotes for top-5 active positions.
- 1-min OHLCV for risk monitoring.
- Order book depth (Kite provides level-2).

### 4.3 Sentiment
- News articles: ~50–200 articles/day across our 500 universe.
- Storage: 1 GB / year is plenty.

---

## 5. Capital Requirements

### Phase-by-phase

| Phase | What | Capital |
|-------|------|---------|
| 0–6 (paper) | Setup → 30-day paper trading | ₹0 |
| 7 (demo) | Broker sandbox | ₹0 |
| 8 (live small) | Real money trial | **₹50,000 minimum**, ₹1,00,000 recommended |
| 9 (live scaled) | After 3 profitable months only | ₹3–5 L |

### Why ₹50k minimum?
- Indian brokerage + STT + GST is significant for small accounts.
- Need ≥10 stocks of meaningful size (₹3–5k each) to diversify.
- Below ₹50k, costs eat 1–2% per trade → backtest results invalid in live.

### Margin / Leverage
- **Phase 8:** **No leverage**. Spot delivery only.
- **Phase 9+:** Maximum 1.5× intraday only, after 6 months of profitable live.
- Never overnight margin until 12 months of consistent live performance.

---

## 6. Compliance & Tax

- **All algo trading via API requires a SEBI-registered broker** — Kite, Upstox, Angel are fine.
- Profits are taxable: STCG (15%) for <1 yr, LTCG (10% over ₹1L) for >1 yr.
- Intraday is treated as **speculative business income** — different ITR form.
- Maintain a trade log with timestamps for audit.
- Consider talking to a CA before Phase 8 (live trading).

---

## 7. Security

- **Never commit API keys to git.** Use `.env` + `.gitignore`.
- Store production keys in macOS Keychain or 1Password CLI.
- Two-factor auth on every brokerage account.
- Separate API keys for paper vs live; rotate quarterly.
- Read-only API key for monitoring; trade-enabled key only on the trading server.

---

## 8. Logistics

| Task | Tool |
|------|------|
| Issue tracker | GitHub Issues |
| Documentation | Markdown in `docs/` (this folder) |
| Secrets | `.env` + macOS Keychain |
| CI | GitHub Actions (lint + test) |
| Deployment | Docker + `docker-compose` on cloud VM |
| Backups | Daily snapshot of SQLite DB to S3 / Backblaze |

---

## 9. The "What Else Do I Need?" Answer

You asked. Here's the honest list:

1. **₹2000/month** for Kite Connect when going live.
2. **₹500–800/month** for cloud VM during live phase.
3. **One CA consultation** (~₹2000) before Phase 8 to set up tax tracking.
4. **30+ days of patience** in the paper phase (this is the hardest one).
5. **A Telegram account + bot** for alerts (free, 5-min setup).
6. **Optional: $5–20/month** for LLM API (Claude/GPT) — useful but not required.
7. **A separate Demat account** for algo trading (don't mix with your manual investments). Can be the same Zerodha or different broker.
8. **Two-factor auth on all broker accounts** — non-negotiable.
9. **A small notebook (paper one)** to journal *why* you overrode the algo, every time you do (you will be tempted; the journal is your accountability).
