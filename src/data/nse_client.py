"""NSE ticker-list provider.

Tries multiple sources in order of preference:

1. ``nsepython`` (if installed) — official-style API to NSE.
2. The bundled seed CSV at ``data/seed/nifty_seed.csv``.

Why a fallback?
    ``nsepython`` makes live HTTP calls to nseindia.com and intermittently
    breaks (anti-bot guards, header changes, network blocks). For a personal
    bot we'd rather have a usable seed list than a hard dependency on a
    third-party scraper.

When the live source works, we update the seed CSV via
``scripts/refresh_universe.py`` so the bundled list stays current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.logging import logger
from src.utils.settings import PROJECT_ROOT

DEFAULT_SEED_PATH = PROJECT_ROOT / "data" / "seed" / "nifty_seed.csv"


@dataclass
class TickerMeta:
    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None


@dataclass
class NSEClient:
    """Fetches the candidate-ticker list from NSE (or seed CSV fallback)."""

    seed_path: Path = field(default_factory=lambda: DEFAULT_SEED_PATH)
    use_nsepython: bool = True

    # ------------------------------------------------------------------
    def list_nifty_500(self) -> list[TickerMeta]:
        """Return the Nifty 500 component list, or our seed fallback."""
        if self.use_nsepython:
            try:
                tickers = self._from_nsepython()
                if tickers:
                    logger.info("Loaded {} tickers from nsepython", len(tickers))
                    return tickers
            except Exception as exc:  # pragma: no cover - network dep
                logger.warning(
                    "nsepython failed ({}); falling back to seed CSV", exc
                )
        return self._from_seed()

    # ------------------------------------------------------------------
    def list_seed(self) -> list[TickerMeta]:
        """Always read from the local seed (deterministic — used in tests)."""
        return self._from_seed()

    # ------------------------------------------------------------------
    def _from_nsepython(self) -> list[TickerMeta]:
        """Try nsepython. Multiple of its functions exist; we try the
        commonly-stable ones in order."""
        try:
            import nsepython
        except ImportError:
            return []

        # Try several likely entry points; nsepython's API has changed.
        candidates = []
        for func_name in (
            "nse_eq_symbols",
            "nse_get_index_constituent_list",
            "nse_get_index_quote",
        ):
            fn = getattr(nsepython, func_name, None)
            if fn is None:
                continue
            try:
                if func_name == "nse_get_index_constituent_list":
                    raw = fn("NIFTY 500")
                else:
                    raw = fn()
                if raw:
                    candidates.append(raw)
                    break
            except Exception:  # pragma: no cover - network dep
                continue

        if not candidates:
            return []

        # nsepython output shape varies — handle both lists of strings and
        # lists of dicts.
        out: list[TickerMeta] = []
        raw = candidates[0]
        if isinstance(raw, pd.DataFrame):
            sym_col = next(
                (c for c in raw.columns if c.upper() in {"SYMBOL", "TICKER"}),
                None,
            )
            if sym_col is None:
                return []
            for _, row in raw.iterrows():
                out.append(
                    TickerMeta(
                        symbol=str(row[sym_col]).upper(),
                        name=str(row.get("Company Name", row.get("NAME", ""))).strip() or None,
                        sector=str(row.get("Sector", row.get("Industry", ""))).strip() or None,
                    )
                )
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    out.append(TickerMeta(symbol=item.upper()))
                elif isinstance(item, dict):
                    out.append(
                        TickerMeta(
                            symbol=str(item.get("symbol", item.get("Symbol", ""))).upper(),
                            name=item.get("name") or item.get("Company Name"),
                            sector=item.get("sector") or item.get("Industry"),
                        )
                    )
        return [t for t in out if t.symbol]

    # ------------------------------------------------------------------
    def _from_seed(self) -> list[TickerMeta]:
        if not self.seed_path.exists():
            logger.error("Seed ticker CSV not found at {}", self.seed_path)
            return []
        df = pd.read_csv(self.seed_path)
        cols = {c.lower(): c for c in df.columns}
        sym_col = cols.get("symbol")
        if sym_col is None:
            raise ValueError(f"{self.seed_path}: missing 'symbol' column")
        out: list[TickerMeta] = []
        for _, row in df.iterrows():
            symbol = str(row[sym_col]).strip().upper()
            if not symbol:
                continue
            out.append(
                TickerMeta(
                    symbol=symbol,
                    name=_optional_str(row.get(cols.get("name", "name"))),
                    sector=_optional_str(row.get(cols.get("sector", "sector"))),
                    industry=_optional_str(row.get(cols.get("industry", "industry"))),
                )
            )
        logger.info("Loaded {} tickers from seed CSV {}", len(out), self.seed_path.name)
        return out


def _optional_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


__all__ = ["NSEClient", "TickerMeta"]
