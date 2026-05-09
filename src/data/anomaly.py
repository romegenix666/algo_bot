"""Bar-quality anomaly detection.

Bad data poisons everything downstream — strategies, backtests, risk
calculations. We screen incoming OHLCV before storing.

Categories of anomalies we catch:

1. **Stale bars** — yesterday's close == today's open == today's close, with
   zero volume. Common when an exchange holiday wasn't filtered and the
   data vendor forward-filled.
2. **Zero / impossibly small volume** — illiquid trading day; the bar is
   technically real but unreliable as a fill price.
3. **Big unexplained gap** — close-to-close return > 20% with no
   corresponding split / dividend on file. Likely an unadjusted split or a
   bad tick that wasn't filtered.
4. **OHLC inconsistency** — high < low, close outside [low, high], etc.
   Pure data corruption.
5. **NaN / negative prices** — the obvious ones.

The detector returns ``Anomaly`` rows with a severity. Callers decide
whether to drop, flag, or pass through.

References:
    - Chan (2009) §3 — "After retrieving the data from a database, it is
      often advisable to do a quick error check."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Iterable

import pandas as pd


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Anomaly:
    symbol: str
    bar_date: date
    code: str
    severity: Severity
    detail: str


# ---------------------------------------------------------------------------
# Detection rules — each takes a per-ticker DataFrame and yields anomalies.
# ---------------------------------------------------------------------------


def _check_ohlc_consistency(symbol: str, df: pd.DataFrame) -> list[Anomaly]:
    out: list[Anomaly] = []
    bad = df[
        (df["high"] < df["low"])
        | (df["close"] < df["low"])
        | (df["close"] > df["high"])
        | (df["open"] < df["low"])
        | (df["open"] > df["high"])
    ]
    for ts, row in bad.iterrows():
        out.append(
            Anomaly(
                symbol=symbol,
                bar_date=_to_date(ts),
                code="ohlc_inconsistent",
                severity=Severity.ERROR,
                detail=(
                    f"O={row['open']:.2f} H={row['high']:.2f} "
                    f"L={row['low']:.2f} C={row['close']:.2f}"
                ),
            )
        )
    return out


def _check_negative_or_nan(symbol: str, df: pd.DataFrame) -> list[Anomaly]:
    cols = ["open", "high", "low", "close"]
    bad = df[df[cols].isna().any(axis=1) | (df[cols] <= 0).any(axis=1)]
    out: list[Anomaly] = []
    for ts, _row in bad.iterrows():
        out.append(
            Anomaly(
                symbol=symbol,
                bar_date=_to_date(ts),
                code="bad_price",
                severity=Severity.ERROR,
                detail="NaN or non-positive OHLC",
            )
        )
    return out


def _check_zero_volume(symbol: str, df: pd.DataFrame) -> list[Anomaly]:
    out: list[Anomaly] = []
    zero = df[df["volume"] == 0]
    for ts, _row in zero.iterrows():
        out.append(
            Anomaly(
                symbol=symbol,
                bar_date=_to_date(ts),
                code="zero_volume",
                severity=Severity.WARN,
                detail="volume=0",
            )
        )
    return out


def _check_stale(symbol: str, df: pd.DataFrame) -> list[Anomaly]:
    """3+ consecutive days with identical OHLC and zero volume → stale feed."""
    out: list[Anomaly] = []
    if len(df) < 3:
        return out
    same_ohlc = (
        (df["open"] == df["close"])
        & (df["high"] == df["close"])
        & (df["low"] == df["close"])
    )
    streak = 0
    for ts, is_stale in zip(same_ohlc.index, same_ohlc.values, strict=True):
        if is_stale and df.loc[ts, "volume"] == 0:
            streak += 1
            if streak >= 3:
                out.append(
                    Anomaly(
                        symbol=symbol,
                        bar_date=_to_date(ts),
                        code="stale_feed",
                        severity=Severity.WARN,
                        detail=f"{streak} consecutive stale bars",
                    )
                )
        else:
            streak = 0
    return out


def _check_unexplained_gap(
    symbol: str,
    df: pd.DataFrame,
    actions: pd.DataFrame | None,
    threshold: float = 0.20,
) -> list[Anomaly]:
    """Close-to-close return > threshold with no split / dividend on that date."""
    out: list[Anomaly] = []
    if len(df) < 2:
        return out

    close = df["close"].astype(float)
    rets = close.pct_change().abs()
    gaps = df[rets > threshold]
    if gaps.empty:
        return out

    action_dates: set[date] = set()
    if actions is not None and not actions.empty and "ex_date" in actions.columns:
        action_dates = {_to_date(d) for d in actions["ex_date"]}

    for ts, row in gaps.iterrows():
        d = _to_date(ts)
        if d in action_dates:
            continue
        out.append(
            Anomaly(
                symbol=symbol,
                bar_date=d,
                code="unexplained_gap",
                severity=Severity.WARN,
                detail=f"close-to-close move {rets.loc[ts]:.1%} (>{threshold:.0%})",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_anomalies(
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    gap_threshold: float = 0.20,
) -> list[Anomaly]:
    """Run every check on one ticker's bars and concatenate results.

    Args:
        symbol: NSE ticker (without ``.NS``).
        bars: DataFrame indexed by date with at least
            ``open, high, low, close, volume`` columns.
        actions: Optional corporate-actions frame with column ``ex_date``.
            Used to suppress "unexplained_gap" warnings on legitimate
            split / dividend days.
        gap_threshold: |return| above which a single bar is flagged.
    """
    if bars.empty:
        return []

    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    out: list[Anomaly] = []
    out.extend(_check_negative_or_nan(symbol, df))
    out.extend(_check_ohlc_consistency(symbol, df))
    out.extend(_check_zero_volume(symbol, df))
    out.extend(_check_stale(symbol, df))
    out.extend(_check_unexplained_gap(symbol, df, actions, gap_threshold))
    return sorted(out, key=lambda a: (a.bar_date, a.code))


def summarise_anomalies(anomalies: Iterable[Anomaly]) -> dict[str, int]:
    """Count by code → useful for log lines."""
    counts: dict[str, int] = {}
    for a in anomalies:
        counts[a.code] = counts.get(a.code, 0) + 1
    return counts


def _to_date(value) -> date:
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    return pd.Timestamp(value).date()


__all__ = ["Anomaly", "Severity", "detect_anomalies", "summarise_anomalies"]
