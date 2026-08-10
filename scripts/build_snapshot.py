"""Rebuild the reproducible OHLCV snapshot that ships with this repository.

Why a bundled snapshot exists
-----------------------------
The application's primary data source is Yahoo Finance via ``yfinance``. That
requires outbound network access, which is not always available (locked-down
CI, offline review, sandboxed graders, rate-limited hosts). Every number quoted
in the README and in ``resume_bullets.md`` is therefore computed from this fixed
extract, so the results are reproducible by anyone, on any machine, with no
network and no API key.

Source
------
Daily OHLCV for S&P 500 constituents, February 2013 - February 2018, published
by Cam Nugent as "S&P 500 stock data" and distributed from
``github.com/CNuge/kaggle-code`` (``stock_data/individual_stocks_5yr.zip``).
The archive contains one CSV per ticker with ``date, open, high, low, close,
volume, Name``.

Usage
-----
    python scripts/build_snapshot.py --archive path/to/individual_stocks_5yr.zip
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import MARKET_SYMBOL, SNAPSHOT_FILE, SNAPSHOT_META_FILE, SNAPSHOT_UNIVERSE
from src.utils.logging_utils import get_logger

logger = get_logger("build_snapshot")

SOURCE_URL = (
    "https://raw.githubusercontent.com/CNuge/kaggle-code/master/"
    "stock_data/individual_stocks_5yr.zip"
)
SOURCE_NAME = "S&P 500 daily OHLCV (Cam Nugent, CNuge/kaggle-code)"
_TICKER_PATTERN = re.compile(r"([A-Za-z.\-]+)_data\.csv$")


def read_archive(archive_path: Path) -> dict[str, pd.DataFrame]:
    """Read every per-ticker CSV out of the source archive.

    Args:
        archive_path: Path to ``individual_stocks_5yr.zip``.

    Returns:
        Mapping of ticker symbol to a date-indexed OHLCV frame.

    Raises:
        FileNotFoundError: If the archive does not exist.
    """
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Source archive not found at {archive_path}.\nDownload it first:\n"
            f"  curl -L -o {archive_path} {SOURCE_URL}"
        )

    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith("_data.csv") and not name.startswith("__MACOSX")
        ]
        for member in members:
            match = _TICKER_PATTERN.search(member)
            if not match:
                continue
            symbol = match.group(1).upper()
            raw = pd.read_csv(io.BytesIO(archive.read(member)), parse_dates=["date"])
            raw = raw.dropna(subset=["date", "close"])
            if raw.empty:
                continue
            frame = (
                raw.rename(columns=str.lower)
                .set_index("date")
                .sort_index()[["open", "high", "low", "close", "volume"]]
                .astype(float)
            )
            frame = frame[~frame.index.duplicated(keep="last")]
            frames[symbol] = frame

    logger.info("Read %d tickers from %s", len(frames), archive_path.name)
    return frames


def build_equal_weight_index(
    frames: dict[str, pd.DataFrame], base_level: float = 100.0
) -> tuple[pd.DataFrame, int]:
    """Construct an equal-weighted price index from full-history constituents.

    Each constituent's OHLC is divided by its own first close, so every stock
    enters the index with equal weight on day one; the index is then the
    cross-sectional mean of those normalised paths, rescaled to ``base_level``.
    Only tickers whose history spans the full sample are used, which keeps the
    index free of discontinuities caused by listings and delistings.

    Args:
        frames: Per-ticker OHLCV frames.
        base_level: Index level on the first session.

    Returns:
        A tuple of ``(index_frame, n_constituents)``.

    Raises:
        ValueError: If no constituent covers the full sample.
    """
    calendar = sorted({date for frame in frames.values() for date in frame.index})
    first_session, last_session = calendar[0], calendar[-1]
    expected_length = len(calendar)

    complete = {
        symbol: frame
        for symbol, frame in frames.items()
        if frame.index[0] == first_session
        and frame.index[-1] == last_session
        and len(frame) >= expected_length * 0.99
    }
    if not complete:
        raise ValueError("No constituent spans the full sample; cannot build an index.")

    logger.info("Equal-weight index built from %d full-history constituents", len(complete))

    normalised: list[pd.DataFrame] = []
    volumes: list[pd.Series] = []
    for frame in complete.values():
        base_price = float(frame["close"].iloc[0])
        if base_price <= 0:
            continue
        normalised.append(frame[["open", "high", "low", "close"]] / base_price)
        volumes.append(frame["volume"])

    index_prices = (
        pd.concat(normalised).groupby(level=0).mean().sort_index() * base_level
    )
    index_prices["volume"] = pd.concat(volumes).groupby(level=0).sum().sort_index()
    return index_prices.dropna(), len(complete)


def build_snapshot(
    archive_path: Path,
    universe: tuple[str, ...] = SNAPSHOT_UNIVERSE,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the long-format snapshot frame and its provenance metadata."""
    frames = read_archive(archive_path)

    missing = [symbol for symbol in universe if symbol not in frames]
    if missing:
        logger.warning("Requested tickers absent from source: %s", missing)

    index_frame, n_constituents = build_equal_weight_index(frames)

    records: list[pd.DataFrame] = []
    included: list[str] = []
    for symbol in universe:
        if symbol not in frames:
            continue
        frame = frames[symbol].copy()
        # The source series are split-adjusted but carry no separate dividend
        # adjustment, so adj_close mirrors close. This is stated in the README.
        frame["adj_close"] = frame["close"]
        frame["symbol"] = symbol
        records.append(frame.reset_index())
        included.append(symbol)

    index_out = index_frame.copy()
    index_out["adj_close"] = index_out["close"]
    index_out["symbol"] = MARKET_SYMBOL
    records.append(index_out.reset_index())

    snapshot = pd.concat(records, ignore_index=True)
    snapshot = snapshot[
        ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    ]
    snapshot = snapshot.sort_values(["symbol", "date"]).reset_index(drop=True)
    for column in ("open", "high", "low", "close", "adj_close"):
        snapshot[column] = snapshot[column].round(4)
    snapshot["volume"] = snapshot["volume"].astype("int64")

    metadata = {
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(len(snapshot)),
        "symbols": included + [MARKET_SYMBOL],
        "n_equities": len(included),
        "market_index_symbol": MARKET_SYMBOL,
        "market_index_constituents": int(n_constituents),
        "source_tickers_available": int(len(frames)),
        "start_date": str(snapshot["date"].min().date()),
        "end_date": str(snapshot["date"].max().date()),
        "sessions_per_equity": int(
            snapshot.loc[snapshot["symbol"] == included[0], "date"].nunique()
        ),
        "adjustment_note": (
            "Source prices are split-adjusted; no separate dividend-adjusted "
            "series is published, so adj_close mirrors close."
        ),
    }
    return snapshot, metadata


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/individual_stocks_5yr.zip"),
        help="Path to the downloaded source archive.",
    )
    args = parser.parse_args()

    snapshot, metadata = build_snapshot(args.archive)

    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(SNAPSHOT_FILE, index=False, compression="gzip")
    with SNAPSHOT_META_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    size_mb = SNAPSHOT_FILE.stat().st_size / 1024 / 1024
    logger.info(
        "Wrote %s (%.2f MB): %d rows, %d symbols, %s to %s",
        SNAPSHOT_FILE,
        size_mb,
        metadata["rows"],
        len(metadata["symbols"]),
        metadata["start_date"],
        metadata["end_date"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
