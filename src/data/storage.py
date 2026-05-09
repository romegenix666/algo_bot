"""SQLAlchemy storage layer.

One SQLite database, four tables:

    tickers              — symbol metadata (name, sector, listing date, status)
    prices               — adjusted OHLCV bars per ticker per day
    corporate_actions    — splits + dividends + bonus issues per ex-date
    universe_snapshots   — which tickers were "tradeable" on each rebalance
                           date (this is what makes survivorship-bias-free
                           backtests possible)

Why SQLite?
    For a single-machine bot tracking ~500 stocks × 5 years (~625K rows),
    SQLite is more than enough. Migrating to Postgres later is a one-line
    config change since we use SQLAlchemy.

Why SQLAlchemy and not raw sql?
    - Vendor-agnostic: SQLite today, Postgres tomorrow.
    - Type safety + auto-migrations via Alembic later.
    - We can still drop to raw SQL when needed (e.g. bulk inserts).

Conventions:
    - All timestamps stored UTC (we convert to IST only at presentation).
    - Prices stored as Decimal(20, 6)? — No. We use Float here for speed.
      Indian equity prices have at most 2 decimal places, well within float
      precision; the alternative (DECIMAL) is 5–10× slower for analytics.
    - PKs are integers (auto-increment) for fast joins; symbol/date are
      indexed but not the PK.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.utils.settings import PROJECT_ROOT, settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base — one for the whole project."""


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Ticker(Base):
    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)
    yf_symbol: Mapped[str] = mapped_column(String(48), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(64), index=True)
    industry: Mapped[str | None] = mapped_column(String(96))
    isin: Mapped[str | None] = mapped_column(String(16))
    listing_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    prices: Mapped[list[Price]] = relationship(
        "Price", back_populates="ticker", cascade="all, delete-orphan", lazy="dynamic"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Ticker {self.symbol} ({self.yf_symbol})>"


class Price(Base):
    __tablename__ = "prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_id: Mapped[int] = mapped_column(
        ForeignKey("tickers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bar_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    ticker: Mapped[Ticker] = relationship("Ticker", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("ticker_id", "bar_date", name="uq_prices_ticker_date"),
        Index("ix_prices_date", "bar_date"),
    )


class CorporateAction(Base):
    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_id: Mapped[int] = mapped_column(
        ForeignKey("tickers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ex_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    action_type: Mapped[str] = mapped_column(String(16), nullable=False)  # 'split', 'dividend', 'bonus'
    ratio: Mapped[float | None] = mapped_column(Float)        # e.g. split 2:1 → 2.0; bonus 1:5 → 1.2
    dividend_amount: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint(
            "ticker_id", "ex_date", "action_type", name="uq_ca_ticker_date_type"
        ),
    )


class UniverseSnapshot(Base):
    """Which tickers belonged to the tradeable universe on a given snapshot date.

    For survivorship-bias-free backtests: at each rebalance date in the past,
    we only allow strategies to *see* the snapshot as it was that day, not as
    it is today (which would silently drop bankrupt names).
    """

    __tablename__ = "universe_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker_id: Mapped[int] = mapped_column(
        ForeignKey("tickers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int | None] = mapped_column(Integer)            # ranking within snapshot
    market_cap_cr: Mapped[float | None] = mapped_column(Float)   # market cap in INR Crore
    avg_turnover_cr: Mapped[float | None] = mapped_column(Float) # 30-day avg turnover INR Cr
    notes: Mapped[str | None] = mapped_column(String(255))

    __table_args__ = (
        UniqueConstraint(
            "snapshot_date", "ticker_id", name="uq_snapshot_date_ticker"
        ),
    )


# ---------------------------------------------------------------------------
# DAO — the only API the rest of the codebase uses
# ---------------------------------------------------------------------------


@dataclass
class DataStore:
    """High-level database access. Constructed once, passed around."""

    database_url: str

    @classmethod
    def from_settings(cls) -> DataStore:
        return cls(database_url=settings.env.database_url)

    @classmethod
    def in_memory(cls) -> DataStore:
        """For tests."""
        return cls(database_url="sqlite:///:memory:")

    def __post_init__(self) -> None:
        # Ensure parent directory exists for file-backed SQLite URLs.
        if self.database_url.startswith("sqlite:///") and not self.database_url.endswith(":memory:"):
            db_path = self.database_url[len("sqlite:///") :]
            full = (PROJECT_ROOT / db_path).resolve() if not Path(db_path).is_absolute() else Path(db_path)
            full.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            self.database_url,
            future=True,
            connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {},
        )
        self._SessionLocal = sessionmaker(
            bind=self.engine, expire_on_commit=False, autoflush=False, future=True
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:  # pragma: no cover - destructive
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional session context manager."""
        sess = self._SessionLocal()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    # ------------------------------------------------------------------
    # Tickers
    # ------------------------------------------------------------------
    def upsert_tickers(self, rows: Iterable[dict]) -> int:
        """Insert tickers that don't exist; update name/sector/etc. on existing.

        Returns the count of rows touched. Idempotent.
        """
        n = 0
        with self.session() as sess:
            for row in rows:
                symbol = row["symbol"].strip().upper()
                yf_symbol = row.get("yf_symbol") or _to_yf_symbol(symbol)
                existing = sess.scalar(
                    select(Ticker).where(Ticker.symbol == symbol)
                )
                if existing is None:
                    sess.add(
                        Ticker(
                            symbol=symbol,
                            yf_symbol=yf_symbol,
                            name=row.get("name"),
                            sector=row.get("sector"),
                            industry=row.get("industry"),
                            isin=row.get("isin"),
                            listing_date=row.get("listing_date"),
                            status=row.get("status", "active"),
                        )
                    )
                else:
                    existing.yf_symbol = yf_symbol
                    if row.get("name"):
                        existing.name = row["name"]
                    if row.get("sector"):
                        existing.sector = row["sector"]
                    if row.get("industry"):
                        existing.industry = row["industry"]
                    if row.get("isin"):
                        existing.isin = row["isin"]
                    if row.get("listing_date"):
                        existing.listing_date = row["listing_date"]
                    if row.get("status"):
                        existing.status = row["status"]
                n += 1
        return n

    def get_ticker(self, symbol: str) -> Ticker | None:
        symbol = symbol.strip().upper()
        with self.session() as sess:
            return sess.scalar(select(Ticker).where(Ticker.symbol == symbol))

    def list_tickers(self, status: str | None = "active") -> list[Ticker]:
        with self.session() as sess:
            stmt = select(Ticker)
            if status is not None:
                stmt = stmt.where(Ticker.status == status)
            return list(sess.scalars(stmt.order_by(Ticker.symbol)).all())

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    def insert_prices(
        self,
        symbol: str,
        bars: pd.DataFrame,
        replace_overlapping: bool = True,
    ) -> int:
        """Bulk-insert OHLCV bars for one ticker.

        Args:
            symbol: NSE symbol (without ``.NS`` suffix).
            bars: DataFrame indexed by date with columns
                ``open, high, low, close, adj_close, volume``.
            replace_overlapping: if True, existing rows for the same dates
                are deleted and replaced (idempotent re-import).

        Returns the number of rows inserted.
        """
        if bars.empty:
            return 0

        bars = _normalise_bars(bars)
        with self.session() as sess:
            ticker = sess.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
            if ticker is None:
                raise ValueError(f"Unknown ticker '{symbol}' — upsert it first")

            if replace_overlapping:
                dates = list(bars.index.date) if hasattr(bars.index, "date") else list(bars.index)
                if dates:
                    sess.query(Price).filter(
                        Price.ticker_id == ticker.id,
                        Price.bar_date.in_(dates),
                    ).delete(synchronize_session=False)

            payload = [
                Price(
                    ticker_id=ticker.id,
                    bar_date=_to_date(idx),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    adj_close=float(row["adj_close"]),
                    volume=int(row["volume"]),
                )
                for idx, row in bars.iterrows()
            ]
            sess.add_all(payload)
            return len(payload)

    def fetch_prices(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Read OHLCV for one ticker as a DatetimeIndex DataFrame."""
        symbol = symbol.upper()
        with self.session() as sess:
            ticker = sess.scalar(select(Ticker).where(Ticker.symbol == symbol))
            if ticker is None:
                return pd.DataFrame()

            stmt = select(Price).where(Price.ticker_id == ticker.id)
            if start is not None:
                stmt = stmt.where(Price.bar_date >= start)
            if end is not None:
                stmt = stmt.where(Price.bar_date <= end)
            stmt = stmt.order_by(Price.bar_date)

            rows = sess.scalars(stmt).all()
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(
                [
                    {
                        "date": r.bar_date,
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                        "adj_close": r.adj_close,
                        "volume": r.volume,
                    }
                    for r in rows
                ]
            )
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date").sort_index()

    def fetch_prices_panel(
        self,
        symbols: Iterable[str],
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Read OHLCV for many tickers, returned as MultiIndex (date, ticker).

        The shape strategies expect: ``prices.index = MultiIndex(date, ticker)``
        with columns ``open, high, low, close, adj_close, volume``.
        """
        frames = []
        for sym in symbols:
            df = self.fetch_prices(sym, start, end)
            if df.empty:
                continue
            df = df.copy()
            df["ticker"] = sym.upper()
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).reset_index()
        out["date"] = pd.to_datetime(out["date"])
        return out.set_index(["date", "ticker"]).sort_index()

    # ------------------------------------------------------------------
    # Corporate actions
    # ------------------------------------------------------------------
    def insert_actions(
        self,
        symbol: str,
        actions: list[dict],
        replace: bool = True,
    ) -> int:
        if not actions:
            return 0
        with self.session() as sess:
            ticker = sess.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
            if ticker is None:
                raise ValueError(f"Unknown ticker '{symbol}'")
            if replace:
                sess.query(CorporateAction).filter(
                    CorporateAction.ticker_id == ticker.id
                ).delete(synchronize_session=False)
            payload = [
                CorporateAction(
                    ticker_id=ticker.id,
                    ex_date=_to_date(a["ex_date"]),
                    action_type=a["action_type"],
                    ratio=a.get("ratio"),
                    dividend_amount=a.get("dividend_amount"),
                )
                for a in actions
            ]
            sess.add_all(payload)
            return len(payload)

    def fetch_actions(self, symbol: str) -> list[CorporateAction]:
        with self.session() as sess:
            ticker = sess.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
            if ticker is None:
                return []
            return list(
                sess.scalars(
                    select(CorporateAction)
                    .where(CorporateAction.ticker_id == ticker.id)
                    .order_by(CorporateAction.ex_date)
                ).all()
            )

    # ------------------------------------------------------------------
    # Universe snapshots
    # ------------------------------------------------------------------
    def save_universe_snapshot(
        self,
        snapshot_date: date,
        rows: list[dict],
        replace: bool = True,
    ) -> int:
        with self.session() as sess:
            if replace:
                sess.query(UniverseSnapshot).filter(
                    UniverseSnapshot.snapshot_date == snapshot_date
                ).delete(synchronize_session=False)

            sym_to_id = {
                t.symbol: t.id for t in sess.scalars(select(Ticker)).all()
            }
            payload: list[UniverseSnapshot] = []
            for r in rows:
                tid = sym_to_id.get(r["symbol"].upper())
                if tid is None:
                    continue
                payload.append(
                    UniverseSnapshot(
                        snapshot_date=snapshot_date,
                        ticker_id=tid,
                        rank=r.get("rank"),
                        market_cap_cr=r.get("market_cap_cr"),
                        avg_turnover_cr=r.get("avg_turnover_cr"),
                        notes=r.get("notes"),
                    )
                )
            sess.add_all(payload)
            return len(payload)

    def fetch_universe_as_of(self, as_of: date) -> list[str]:
        """Return the symbols of the most recent snapshot on or before ``as_of``."""
        with self.session() as sess:
            latest = sess.scalar(
                select(UniverseSnapshot.snapshot_date)
                .where(UniverseSnapshot.snapshot_date <= as_of)
                .order_by(UniverseSnapshot.snapshot_date.desc())
                .limit(1)
            )
            if latest is None:
                return []
            stmt = (
                select(Ticker.symbol)
                .join(UniverseSnapshot, UniverseSnapshot.ticker_id == Ticker.id)
                .where(UniverseSnapshot.snapshot_date == latest)
                .order_by(UniverseSnapshot.rank.asc().nulls_last())
            )
            return list(sess.scalars(stmt).all())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_yf_symbol(symbol: str) -> str:
    """NSE symbol → Yahoo Finance symbol. ``RELIANCE`` → ``RELIANCE.NS``.

    Indices stay as-is (already prefixed with ``^``).
    """
    s = symbol.strip().upper()
    if s.startswith("^"):
        return s
    if "." in s:
        return s
    return f"{s}.NS"


def _to_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        return pd.Timestamp(value).date()
    raise TypeError(f"Cannot coerce {type(value)} to date")


def _normalise_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are lowercase and an ``adj_close`` column exists."""
    df = bars.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            df["adj_close"] = df["adjclose"]
        else:
            df["adj_close"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0
    needed = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    return df[list(needed)]


__all__ = [
    "Base",
    "CorporateAction",
    "DataStore",
    "Price",
    "Ticker",
    "UniverseSnapshot",
]
