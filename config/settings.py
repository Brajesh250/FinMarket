"""Central configuration for FinMarket ML.

All tunable constants live here so that notebooks, scripts, the test-suite and
the Streamlit application share a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
SNAPSHOT_DIR: Final[Path] = DATA_DIR / "snapshot"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORT_DIR: Final[Path] = PROJECT_ROOT / "reports"

SNAPSHOT_FILE: Final[Path] = SNAPSHOT_DIR / "market_snapshot.csv.gz"
SNAPSHOT_META_FILE: Final[Path] = SNAPSHOT_DIR / "snapshot_metadata.json"

for _directory in (RAW_DATA_DIR, SNAPSHOT_DIR, MODEL_DIR, REPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
# Symbol used for the broad-market context feature.  In the bundled snapshot
# this is an equal-weighted index computed from every constituent in the
# source dataset; in live mode it maps to a real index ticker (see below).
MARKET_SYMBOL: Final[str] = "MKT_EW"
LIVE_MARKET_SYMBOL: Final[str] = "^GSPC"

# Tickers shipped inside the offline snapshot.  Live mode accepts any symbol
# that yfinance can resolve, so this list is a convenience default only.
SNAPSHOT_UNIVERSE: Final[tuple[str, ...]] = (
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "AMD", "INTC", "CSCO",
    "ORCL", "IBM", "ADBE", "CRM", "QCOM", "NFLX", "JPM", "BAC", "WFC", "C",
    "GS", "V", "MA", "JNJ", "PFE", "MRK", "UNH", "XOM", "CVX", "WMT", "PG",
    "KO", "MCD", "NKE", "HD", "DIS", "BA", "CAT", "T", "VZ",
)

# Suggested tickers offered in the app's "live" mode.
LIVE_UNIVERSE: Final[tuple[str, ...]] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM",
    "^GSPC", "^IXIC", "^DJI",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "^NSEI",
)

MIN_ROWS_REQUIRED: Final[int] = 260  # ~1 trading year; below this we refuse to model
DEFAULT_LIVE_PERIOD_YEARS: Final[int] = 8

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
RSI_WINDOW: Final[int] = 14
MACD_FAST: Final[int] = 12
MACD_SLOW: Final[int] = 26
MACD_SIGNAL: Final[int] = 9
BOLLINGER_WINDOW: Final[int] = 20
BOLLINGER_STD: Final[float] = 2.0
SMA_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20, 50)
EMA_WINDOWS: Final[tuple[int, ...]] = (10, 20)
MOMENTUM_WINDOWS: Final[tuple[int, ...]] = (3, 5, 10, 20)
VOLATILITY_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)
VOLUME_SMA_WINDOW: Final[int] = 20

# Longest look-back used by any feature.  The first `WARMUP_ROWS` rows of every
# ticker are dropped because their indicators are not yet well defined.
WARMUP_ROWS: Final[int] = 60

TRADING_DAYS_PER_YEAR: Final[int] = 252

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
TRAIN_FRACTION: Final[float] = 0.70
VALIDATION_FRACTION: Final[float] = 0.15
# Test fraction is the remainder (0.15).

RANDOM_STATE: Final[int] = 42
CV_SPLITS: Final[int] = 5

# Research signal thresholds applied to P(up).
BULLISH_THRESHOLD: Final[float] = 0.55
BEARISH_THRESHOLD: Final[float] = 0.45
# Round-trip cost assumption used by the illustrative signal study, in basis
# points of notional per position change.
TRANSACTION_COST_BPS: Final[float] = 5.0

DISCLAIMER: Final[str] = (
    "For educational and research purposes only. This application does not "
    "constitute investment advice."
)
