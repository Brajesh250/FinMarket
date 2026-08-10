"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """A deterministic 800-session OHLCV frame with realistic structure.

    Built from a seeded geometric random walk so tests never depend on the
    bundled snapshot, on network access, or on today's date.
    """
    rng = np.random.default_rng(20240101)
    sessions = 800
    dates = pd.bdate_range("2018-01-02", periods=sessions, name="date")

    returns = rng.normal(loc=0.0004, scale=0.013, size=sessions)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0, 0.003, sessions))
    spread = np.abs(rng.normal(0.008, 0.004, sessions))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.integers(1_000_000, 40_000_000, sessions)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume.astype(float),
        },
        index=dates,
    )


@pytest.fixture(scope="session")
def synthetic_market(synthetic_ohlcv: pd.DataFrame) -> pd.Series:
    """A market return series aligned to the synthetic ticker's calendar."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.006, len(synthetic_ohlcv))
    stock_returns = synthetic_ohlcv["close"].pct_change().to_numpy()
    market = pd.Series(stock_returns * 0.6 + noise, index=synthetic_ohlcv.index)
    market.name = "market_return"
    return market


@pytest.fixture(scope="session")
def snapshot_available() -> bool:
    """Whether the bundled snapshot file is present in this checkout."""
    from config.settings import SNAPSHOT_FILE

    return SNAPSHOT_FILE.exists()
