"""Market-data access layer.

Two interchangeable providers sit behind one interface:

``SnapshotProvider``
    Reads the reproducible OHLCV extract that ships with the repository. It
    needs no network access, so the published demo and the test-suite always
    behave identically.

``YFinanceProvider``
    Downloads fresh data from Yahoo Finance via ``yfinance``. Used when the
    application runs somewhere with outbound network access.

``load_prices`` picks a provider according to :class:`DataMode` and falls back
to the snapshot whenever a live download fails, so the UI never dead-ends.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from enum import Enum
from functools import lru_cache
from typing import Protocol

import pandas as pd

from config.settings import (
    DEFAULT_LIVE_PERIOD_YEARS,
    LIVE_MARKET_SYMBOL,
    MARKET_SYMBOL,
    SNAPSHOT_FILE,
    SNAPSHOT_META_FILE,
)
from src.data.validation import DataValidationError, validate_price_frame
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")


class DataMode(str, Enum):
    """How price data should be sourced."""

    SNAPSHOT = "snapshot"
    LIVE = "live"


class PriceProvider(Protocol):
    """Minimal interface every price source must implement."""

    def available_symbols(self) -> list[str]:
        """Return the symbols this provider can serve."""

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Return an OHLCV frame indexed by date for ``symbol``."""


# ---------------------------------------------------------------------------
# Snapshot provider
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _read_snapshot() -> pd.DataFrame:
    """Load and cache the bundled snapshot as a tidy long frame."""
    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(
            f"Bundled snapshot not found at {SNAPSHOT_FILE}. "
            "Run `python scripts/build_snapshot.py` to regenerate it."
        )
    frame = pd.read_csv(SNAPSHOT_FILE, parse_dates=["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(
        "Loaded snapshot: %d rows, %d symbols, %s to %s",
        len(frame),
        frame["symbol"].nunique(),
        frame["date"].min().date(),
        frame["date"].max().date(),
    )
    return frame


def snapshot_metadata() -> dict:
    """Return provenance metadata describing the bundled snapshot."""
    if not SNAPSHOT_META_FILE.exists():
        return {}
    with SNAPSHOT_META_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class SnapshotProvider:
    """Serves OHLCV data from the repository's bundled extract."""

    def available_symbols(self) -> list[str]:
        """Return every symbol present in the snapshot, alphabetically."""
        return sorted(_read_snapshot()["symbol"].unique().tolist())

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Return the OHLCV frame for ``symbol``.

        Raises:
            KeyError: If the symbol is not part of the snapshot.
        """
        frame = _read_snapshot()
        subset = frame.loc[frame["symbol"] == symbol]
        if subset.empty:
            raise KeyError(
                f"'{symbol}' is not in the bundled snapshot. "
                f"Available: {', '.join(self.available_symbols()[:10])}..."
            )
        subset = subset.drop(columns=["symbol"]).set_index("date").sort_index()
        subset.index.name = "date"
        return subset


# ---------------------------------------------------------------------------
# Live provider
# ---------------------------------------------------------------------------
class YFinanceProvider:
    """Serves OHLCV data downloaded live from Yahoo Finance."""

    def __init__(self, years: int = DEFAULT_LIVE_PERIOD_YEARS) -> None:
        """Store the look-back window used for downloads."""
        self.years = years

    def available_symbols(self) -> list[str]:
        """Live mode accepts any resolvable ticker, so this list is advisory."""
        from config.settings import LIVE_UNIVERSE

        return list(LIVE_UNIVERSE)

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Download and normalise daily OHLCV for ``symbol``.

        Raises:
            RuntimeError: If ``yfinance`` is unavailable or returns no rows.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("yfinance is not installed; use snapshot mode.") from exc

        end = date.today()
        start = end - timedelta(days=int(365.25 * self.years))
        logger.info("Downloading %s from Yahoo Finance (%s to %s)", symbol, start, end)

        try:
            raw = yf.download(
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001 - network failures are varied
            raise RuntimeError(f"Yahoo Finance download failed for {symbol}: {exc}") from exc

        if raw is None or raw.empty:
            raise RuntimeError(f"Yahoo Finance returned no rows for '{symbol}'.")

        return _normalise_yfinance_frame(raw)


def _normalise_yfinance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output into the project's lower-case OHLCV schema."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    frame = frame[[c for c in OHLCV_COLUMNS if c in frame.columns]]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    return frame.sort_index().dropna(how="all")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_provider(mode: DataMode, years: int = DEFAULT_LIVE_PERIOD_YEARS) -> PriceProvider:
    """Return the provider matching ``mode``."""
    return YFinanceProvider(years=years) if mode is DataMode.LIVE else SnapshotProvider()


def market_symbol_for(mode: DataMode) -> str:
    """Return the broad-market symbol appropriate for ``mode``."""
    return LIVE_MARKET_SYMBOL if mode is DataMode.LIVE else MARKET_SYMBOL


def load_prices(
    symbol: str,
    mode: DataMode = DataMode.SNAPSHOT,
    years: int = DEFAULT_LIVE_PERIOD_YEARS,
    validate: bool = True,
) -> tuple[pd.DataFrame, DataMode]:
    """Load OHLCV data for ``symbol``, degrading gracefully to the snapshot.

    Args:
        symbol: Ticker to load.
        mode: Preferred data mode.
        years: Look-back window for live downloads.
        validate: Run :func:`validate_price_frame` before returning.

    Returns:
        A tuple of ``(frame, effective_mode)``. ``effective_mode`` reveals
        whether a live request silently fell back to the snapshot.

    Raises:
        DataValidationError: If the resulting frame fails validation.
        KeyError: If the symbol cannot be served by any provider.
    """
    effective_mode = mode
    frame: pd.DataFrame | None = None

    if mode is DataMode.LIVE:
        try:
            frame = YFinanceProvider(years=years).fetch(symbol)
        except Exception as exc:  # noqa: BLE001 - fall back on any live failure
            logger.warning("Live fetch failed for %s (%s); falling back to snapshot.", symbol, exc)
            effective_mode = DataMode.SNAPSHOT

    if frame is None:
        frame = SnapshotProvider().fetch(symbol)
        effective_mode = DataMode.SNAPSHOT

    if validate:
        validate_price_frame(frame, symbol=symbol, strict=True)
    return frame, effective_mode


def load_market_context(
    mode: DataMode = DataMode.SNAPSHOT,
    years: int = DEFAULT_LIVE_PERIOD_YEARS,
) -> pd.Series | None:
    """Return daily returns of the broad-market series, or None if unavailable.

    The market return is used as an optional context feature. Failing to obtain
    it is not fatal: the feature builder simply omits the column.
    """
    symbol = market_symbol_for(mode)
    try:
        frame, _ = load_prices(symbol, mode=mode, years=years, validate=False)
    except (KeyError, DataValidationError, FileNotFoundError, RuntimeError) as exc:
        logger.warning("Market context unavailable (%s): %s", symbol, exc)
        return None
    returns = frame["close"].astype(float).pct_change()
    returns.name = "market_return"
    return returns
