"""Technical indicators — vectorised, pandas-native, look-ahead-safe.

Every function here:

1. Operates on a ``pandas.Series`` or ``pandas.DataFrame`` indexed by a
   ``DatetimeIndex`` (ascending).
2. Uses ONLY data on or before the index of each output row. No future leakage.
3. Returns the same index as the input — ``NaN`` where the rolling window has
   insufficient data. Never silently zero-fills.
4. Is unit-tested in ``tests/test_indicators.py``.

References:
    - Wilder, J. W. Jr. (1978), *New Concepts in Technical Trading Systems*.
      Original definitions of RSI, ATR, ADX.
    - Bollinger, J. (2002), *Bollinger on Bollinger Bands*.
    - Murphy, J. (1999), *Technical Analysis of the Financial Markets*.
    - Kakushadze & Serur (2018), *151 Trading Strategies* §3.11–3.15.
    - Chan (2009), *Quantitative Trading*, Ch. 7.

A note on Wilder smoothing:
    Wilder's original RSI/ATR/ADX use a recursive smoother equivalent to an EMA
    with ``alpha = 1/n``. We expose this via ``ewm(alpha=1/n, adjust=False)``
    rather than the simple ``rolling(n).mean()`` to match the book exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Returns & momentum
# ---------------------------------------------------------------------------


def simple_returns(close: pd.Series) -> pd.Series:
    """Period-over-period simple returns: ``(P_t - P_{t-1}) / P_{t-1}``."""
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns: ``ln(P_t / P_{t-1})``. Additive over time, IID-friendlier."""
    return np.log(close / close.shift(1))


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """Cumulative return over the last ``window`` bars: ``P_t / P_{t-window} - 1``."""
    if window <= 0:
        raise ValueError("window must be positive")
    return close / close.shift(window) - 1.0


def momentum_12_1(close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Jegadeesh-Titman style momentum.

    Cumulative return from ``t - lookback`` to ``t - skip``, deliberately
    skipping the most recent ``skip`` bars to avoid the 1-month reversal
    contamination (Lehmann 1990, Jegadeesh 1990).

    Defaults: 12-month lookback, 1-month skip on daily bars (≈252/21).
    """
    if lookback <= skip:
        raise ValueError("lookback must be greater than skip")
    return close.shift(skip) / close.shift(lookback) - 1.0


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=1)


def rolling_volatility(close: pd.Series, window: int = 60, annualize: bool = True) -> pd.Series:
    """Annualised realised volatility of *log* returns over ``window`` bars.

    Daily bars: annualise by ``sqrt(252)``.
    """
    rets = log_returns(close)
    vol = rets.rolling(window=window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: ``(x - mean) / std`` over the trailing ``window``."""
    mean = rolling_mean(series, window)
    std = rolling_std(series, window)
    # Avoid div-by-zero (constant windows): emit NaN instead of inf.
    z = (series - mean) / std.replace(0.0, np.nan)
    return z


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score across columns (tickers) for each row (date).

    Use this to neutralise factor exposures before ranking. Robust to NaNs.
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=1).replace(0.0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index) — Wilder
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, in [0, 100]. Standard period is 14."""
    if window < 2:
        raise ValueError("window must be >= 2")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing == EMA with alpha = 1/window, adjust=False.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is zero and avg_gain is positive → RSI = 100. Handle that:
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


# ---------------------------------------------------------------------------
# True Range / ATR — Wilder
# ---------------------------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: ``max(H-L, |H - C_prev|, |L - C_prev|)``."""
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    tr.name = "true_range"
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range, Wilder smoothing. The standard volatility-aware
    distance metric used for ATR-based stops."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Returns DataFrame with columns ``[mid, upper, lower, pct_b, bandwidth]``.

    - ``pct_b`` (%B) is the position of the close inside the band, in roughly
      ``[0, 1]`` when inside the bands.
    - ``bandwidth`` measures volatility — useful for "Bollinger squeeze" entry.
    """
    mid = rolling_mean(close, window)
    sd = rolling_std(close, window)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth}
    )


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Standard MACD with EMAs. Columns: ``[macd, signal, hist]``."""
    if fast >= slow:
        raise ValueError("fast period must be < slow period")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


# ---------------------------------------------------------------------------
# Donchian Channel (Turtle / Breakout)
# ---------------------------------------------------------------------------


def donchian_channel(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Donchian channel: rolling max-high & rolling min-low.

    The ceiling/floor used by Turtle Traders and most channel-breakout systems.
    The window we use **excludes the current bar** — using the trailing
    ``window`` bars as of yesterday — so a "new 20-day high today" is a real
    breakout signal, not a tautology (which would always be true if the
    current bar is included).
    """
    upper = high.shift(1).rolling(window=window, min_periods=window).max()
    lower = low.shift(1).rolling(window=window, min_periods=window).min()
    mid = (upper + lower) / 2.0
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid})


# ---------------------------------------------------------------------------
# ADX (Average Directional Index) — trend strength
# ---------------------------------------------------------------------------


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """Wilder's ADX with +DI / -DI. ADX in ``[0, 100]``.

    Rule of thumb (Wilder): ADX > 25 indicates a meaningful trend.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    tr = true_range(high, low, close)
    atr_w = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        / atr_w.replace(0.0, np.nan)
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        / atr_w.replace(0.0, np.nan)
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_line = dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line})


# ---------------------------------------------------------------------------
# Internal Bar Strength (Pagonidis 2014, used in 151 Trading Strategies §4.4)
# ---------------------------------------------------------------------------


def internal_bar_strength(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """``IBS = (Close - Low) / (High - Low)``, in [0, 1].

    Close to 0 → bar closed near the low (potentially "cheap").
    Close to 1 → bar closed near the high (potentially "rich").
    """
    span = (high - low).replace(0.0, np.nan)
    return (close - low) / span


__all__ = [
    "adx",
    "atr",
    "bollinger_bands",
    "cross_sectional_zscore",
    "donchian_channel",
    "internal_bar_strength",
    "log_returns",
    "macd",
    "momentum_12_1",
    "rolling_mean",
    "rolling_return",
    "rolling_std",
    "rolling_volatility",
    "rsi",
    "simple_returns",
    "true_range",
    "zscore",
]
