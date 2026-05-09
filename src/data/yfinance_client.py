"""Wrapper around ``yfinance`` for Indian (NSE) equities.

Why not call ``yfinance.download`` directly everywhere?

1. **Consistent column names** — yfinance returns ``Adj Close`` (with a
   space + capital), or ``Close`` only, depending on version and whether
   you ``auto_adjust``. We always return ``adj_close`` lowercase.
2. **Symbol coercion** — Indian stocks need ``.NS`` suffix, indices use
   ``^`` prefix. Caller passes plain symbols; we add the suffix.
3. **Retries with backoff** — yfinance is rate-limited and occasionally
   returns empty frames. We retry with jitter.
4. **Batched downloads** — ``yfinance.download`` accepts space-separated
   tickers and returns a wide multi-column frame; we chunk + reshape.
5. **Failure tolerance** — one bad ticker shouldn't kill a batch of 500.
   We log failures and return whatever succeeded.

The bot's only entry point: ``YFinanceClient.fetch_history(symbols, period)``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dep
    yf = None

from src.utils.logging import logger

# yfinance returns columns with capital letters and spaces; we normalise.
_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


@dataclass(frozen=True)
class FetchResult:
    """One ticker's outcome from a fetch."""

    symbol: str
    yf_symbol: str
    bars: pd.DataFrame   # may be empty on failure
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.bars.empty


@dataclass
class YFinanceClient:
    """Thin retry-aware wrapper around ``yfinance``."""

    chunk_size: int = 50          # tickers per yfinance.download call
    max_retries: int = 3
    backoff_seconds: float = 2.0
    request_timeout: float = 30.0

    def __post_init__(self) -> None:
        if yf is None:
            raise ImportError(
                "yfinance is not installed. Run `pip install yfinance` "
                "or install with the dev requirements."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_history(
        self,
        symbols: Iterable[str],
        start: date | None = None,
        end: date | None = None,
        period: str | None = "5y",
    ) -> list[FetchResult]:
        """Fetch daily OHLCV for many symbols.

        Either ``start``/``end`` OR ``period`` must be provided. If both,
        ``start``/``end`` win.

        Returns a list of ``FetchResult`` — one per requested symbol,
        in input order. Caller decides what to do with failures.
        """
        symbols_list = [s.strip().upper() for s in symbols if s and s.strip()]
        if not symbols_list:
            return []

        if start is not None and end is None:
            end = datetime.utcnow().date()

        results: list[FetchResult] = []
        for chunk in _chunked(symbols_list, self.chunk_size):
            yf_chunk = [_to_yf_symbol(s) for s in chunk]
            sym_map = dict(zip(yf_chunk, chunk, strict=True))
            try:
                df = self._download_with_retries(
                    yf_chunk, start=start, end=end, period=period
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("yfinance batch failed for {}: {}", chunk[:3], exc)
                for sym in chunk:
                    results.append(
                        FetchResult(symbol=sym, yf_symbol=_to_yf_symbol(sym),
                                    bars=pd.DataFrame(), error=str(exc))
                    )
                continue

            results.extend(_unpack_batch(df, sym_map))

        return results

    # ------------------------------------------------------------------
    def fetch_one(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        period: str | None = "5y",
    ) -> FetchResult:
        return self.fetch_history([symbol], start=start, end=end, period=period)[0]

    # ------------------------------------------------------------------
    def fetch_actions(self, symbol: str) -> pd.DataFrame:
        """Fetch dividend + split history for one ticker.

        Returns a DataFrame with columns ``[ex_date, action_type,
        ratio, dividend_amount]``.
        """
        yf_sym = _to_yf_symbol(symbol)
        try:
            ticker = yf.Ticker(yf_sym)
            actions = ticker.actions  # DataFrame indexed by date with Dividends, Stock Splits
        except Exception as exc:
            logger.warning("Failed to fetch actions for {}: {}", symbol, exc)
            return pd.DataFrame()

        if actions is None or actions.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for idx, row in actions.iterrows():
            ex_date = idx.date() if hasattr(idx, "date") else idx
            div = float(row.get("Dividends", 0) or 0)
            split = float(row.get("Stock Splits", 0) or 0)
            if div > 0:
                rows.append(
                    {
                        "ex_date": ex_date,
                        "action_type": "dividend",
                        "ratio": None,
                        "dividend_amount": div,
                    }
                )
            if split > 0:
                rows.append(
                    {
                        "ex_date": ex_date,
                        "action_type": "split",
                        "ratio": split,
                        "dividend_amount": None,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _download_with_retries(
        self,
        yf_symbols: list[str],
        *,
        start: date | None,
        end: date | None,
        period: str | None,
    ) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "tickers": " ".join(yf_symbols),
                    "auto_adjust": False,
                    "progress": False,
                    "threads": True,
                    "group_by": "ticker",
                    "timeout": self.request_timeout,
                }
                if start is not None:
                    kwargs["start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
                    if end is not None:
                        kwargs["end"] = (
                            pd.Timestamp(end) + pd.Timedelta(days=1)
                        ).strftime("%Y-%m-%d")
                else:
                    kwargs["period"] = period or "5y"

                df = yf.download(**kwargs)
                if df is None or df.empty:
                    raise RuntimeError("yfinance returned empty")
                return df
            except Exception as exc:
                last_exc = exc
                wait = self.backoff_seconds * (2 ** attempt)
                logger.warning(
                    "yfinance attempt {}/{} failed: {} — retrying in {:.1f}s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_exc or RuntimeError("yfinance failed for unknown reason")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _to_yf_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.startswith("^"):
        return s
    if "." in s:
        return s
    return f"{s}.NS"


def _unpack_batch(
    df: pd.DataFrame, sym_map: dict[str, str]
) -> list[FetchResult]:
    """yfinance.download with multiple tickers returns a multi-column frame.

    Top level: ticker → second level: column. We split into per-ticker frames.
    """
    results: list[FetchResult] = []

    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker download.
        for yf_sym, sym in sym_map.items():
            if yf_sym not in df.columns.get_level_values(0):
                results.append(
                    FetchResult(
                        symbol=sym, yf_symbol=yf_sym, bars=pd.DataFrame(),
                        error="not in response",
                    )
                )
                continue
            sub = df[yf_sym].copy()
            sub = sub.rename(columns=_COLUMN_MAP)
            sub = sub.dropna(how="all")
            if sub.empty:
                results.append(
                    FetchResult(
                        symbol=sym, yf_symbol=yf_sym, bars=pd.DataFrame(),
                        error="empty after dropna",
                    )
                )
                continue
            results.append(
                FetchResult(symbol=sym, yf_symbol=yf_sym, bars=_clean(sub))
            )
    else:
        # Single-ticker download (rare in our chunked path, but handle it).
        sub = df.rename(columns=_COLUMN_MAP).dropna(how="all")
        only_yf, only_sym = next(iter(sym_map.items()))
        if sub.empty:
            results.append(
                FetchResult(
                    symbol=only_sym, yf_symbol=only_yf, bars=pd.DataFrame(),
                    error="empty",
                )
            )
        else:
            results.append(
                FetchResult(symbol=only_sym, yf_symbol=only_yf, bars=_clean(sub))
            )

    return results


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup: ensure required columns, integer volume, sorted index."""
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype("int64")
    needed = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0 if col == "volume" else float("nan")
    df = df[needed].sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def lookback_window(years: int) -> tuple[date, date]:
    """Convenience: produce a ``(start, end)`` pair covering ``years`` back."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(years * 365.25) + 5)
    return start, end


__all__ = ["FetchResult", "YFinanceClient", "lookback_window"]
