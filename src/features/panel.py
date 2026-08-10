"""Cross-sectional (panel) dataset construction.

A single ticker yields roughly 1,200 daily observations, which is not much for
a problem with as little signal as next-day direction. Stacking many tickers
into one panel gives the learner tens of thousands of rows and forces it to
find relationships that generalise across companies rather than memorising one
name's idiosyncrasies.

The split must then be made on the *date* axis rather than the row axis, so
that every ticker's training window ends before every ticker's test window
begins. ``build_panel`` returns the frame with its dates preserved so
``chronological_split_by_date`` can do exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.loaders import DataMode, SnapshotProvider, load_market_context
from src.data.validation import DataValidationError, validate_price_frame
from src.features.builder import FeatureMatrix, build_feature_matrix
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PanelDataset:
    """A stacked, multi-ticker supervised dataset."""

    features: pd.DataFrame
    target: pd.Series
    next_day_return: pd.Series
    symbols: pd.Series
    dates: pd.DatetimeIndex
    feature_names: list[str]
    included_symbols: list[str]
    skipped: dict[str, str]

    def __len__(self) -> int:
        """Number of stacked observations."""
        return len(self.features)

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Earliest and latest session in the panel."""
        return self.dates.min(), self.dates.max()


def build_panel(
    symbols: list[str],
    mode: DataMode = DataMode.SNAPSHOT,
    include_market_context: bool = True,
) -> PanelDataset:
    """Build one stacked dataset from many tickers.

    Args:
        symbols: Tickers to include.
        mode: Data mode passed through to the loader.
        include_market_context: Add broad-market features when available.

    Returns:
        A :class:`PanelDataset`.

    Raises:
        ValueError: If no ticker produced usable rows.
    """
    provider = SnapshotProvider() if mode is DataMode.SNAPSHOT else None
    market_returns = load_market_context(mode) if include_market_context else None

    blocks: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    included: list[str] = []
    feature_names: list[str] = []

    for symbol in symbols:
        try:
            if provider is not None:
                frame = provider.fetch(symbol)
            else:
                from src.data.loaders import load_prices

                frame, _ = load_prices(symbol, mode=mode, validate=False)
            validate_price_frame(frame, symbol=symbol, strict=True)
            matrix: FeatureMatrix = build_feature_matrix(frame, market_returns=market_returns)
        except (KeyError, DataValidationError, ValueError, RuntimeError) as exc:
            skipped[symbol] = str(exc)
            logger.warning("Skipping %s: %s", symbol, exc)
            continue

        block = matrix.features.copy()
        block["__target"] = matrix.target
        block["__next_day_return"] = matrix.next_day_return
        block["__symbol"] = symbol
        blocks.append(block.reset_index())
        included.append(symbol)
        if not feature_names:
            feature_names = list(matrix.feature_names)

    if not blocks:
        raise ValueError("No ticker produced a usable feature matrix.")

    stacked = pd.concat(blocks, ignore_index=True)
    # Sorting by date first is what makes a date-based chronological split valid.
    stacked = stacked.sort_values(["date", "__symbol"]).reset_index(drop=True)

    # Keep only features shared by every ticker, so the matrix is rectangular.
    common = [name for name in feature_names if name in stacked.columns]
    dataset = PanelDataset(
        features=stacked[common].astype(float),
        target=stacked["__target"].astype(int),
        next_day_return=stacked["__next_day_return"].astype(float),
        symbols=stacked["__symbol"],
        dates=pd.DatetimeIndex(stacked["date"]),
        feature_names=common,
        included_symbols=included,
        skipped=skipped,
    )
    logger.info(
        "Panel built: %d rows x %d features across %d tickers (%s to %s)",
        len(dataset),
        len(common),
        len(included),
        dataset.date_range[0].date(),
        dataset.date_range[1].date(),
    )
    return dataset
