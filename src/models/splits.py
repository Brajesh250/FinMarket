"""Chronological data splitting.

Random shuffling is invalid for market data: it lets the model learn from
sessions that happen after the ones it is scored on. Every split produced here
is contiguous in time, and ``assert_chronological`` is used by the test-suite
to prove it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import CV_SPLITS, TRAIN_FRACTION, VALIDATION_FRACTION


@dataclass(frozen=True)
class ChronologicalSplit:
    """Train / validation / test partitions ordered in time."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        """Row count of each partition."""
        return {
            "train": len(self.X_train),
            "validation": len(self.X_valid),
            "test": len(self.X_test),
        }

    @property
    def periods(self) -> dict[str, tuple[str, str]]:
        """First and last date of each partition, as ISO strings."""

        def bounds(frame: pd.DataFrame) -> tuple[str, str]:
            if frame.empty:
                return ("n/a", "n/a")
            return (str(frame.index[0].date()), str(frame.index[-1].date()))

        return {
            "train": bounds(self.X_train),
            "validation": bounds(self.X_valid),
            "test": bounds(self.X_test),
        }

    @property
    def train_plus_validation(self) -> tuple[pd.DataFrame, pd.Series]:
        """Concatenated train+validation set, used for the final refit."""
        return (
            pd.concat([self.X_train, self.X_valid]),
            pd.concat([self.y_train, self.y_valid]),
        )


def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> ChronologicalSplit:
    """Cut a time-ordered dataset into train / validation / test blocks.

    Args:
        features: Feature matrix indexed by date, ascending.
        target: Aligned label series.
        train_fraction: Share of rows used for training.
        validation_fraction: Share of rows used for validation.

    Returns:
        A :class:`ChronologicalSplit`.

    Raises:
        ValueError: On misaligned inputs, bad fractions, or too few rows.
    """
    if not features.index.equals(target.index):
        raise ValueError("features and target must share an identical index.")
    if not features.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by date ascending before splitting.")
    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("fractions must be positive.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave room for a test set.")

    n_rows = len(features)
    if n_rows < 100:
        raise ValueError(f"Need at least 100 rows to split, got {n_rows}.")

    train_end = int(n_rows * train_fraction)
    valid_end = int(n_rows * (train_fraction + validation_fraction))

    return ChronologicalSplit(
        X_train=features.iloc[:train_end],
        y_train=target.iloc[:train_end],
        X_valid=features.iloc[train_end:valid_end],
        y_valid=target.iloc[train_end:valid_end],
        X_test=features.iloc[valid_end:],
        y_test=target.iloc[valid_end:],
    )


def chronological_split_by_date(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.DatetimeIndex,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[ChronologicalSplit, dict[str, tuple[str, str]]]:
    """Split a stacked panel on calendar boundaries rather than row position.

    With many tickers per session, cutting on row position would place the same
    trading day in two different partitions. Cutting on the sorted list of
    unique dates guarantees that every session belongs to exactly one block and
    that the blocks are strictly ordered in time.

    Args:
        features: Panel feature matrix with a positional index.
        target: Aligned labels.
        dates: Session date for every row.
        train_fraction: Share of *sessions* used for training.
        validation_fraction: Share of *sessions* used for validation.

    Returns:
        A tuple of the split (date-indexed) and the date bounds of each block.

    Raises:
        ValueError: On misaligned inputs or too few distinct sessions.
    """
    if len(features) != len(target) or len(features) != len(dates):
        raise ValueError("features, target and dates must all have the same length.")

    unique_dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    if len(unique_dates) < 100:
        raise ValueError(f"Need at least 100 distinct sessions, got {len(unique_dates)}.")

    train_cut = unique_dates[int(len(unique_dates) * train_fraction) - 1]
    valid_cut = unique_dates[int(len(unique_dates) * (train_fraction + validation_fraction)) - 1]

    train_mask = dates <= train_cut
    valid_mask = (dates > train_cut) & (dates <= valid_cut)
    test_mask = dates > valid_cut

    def block(mask) -> tuple[pd.DataFrame, pd.Series]:
        sub_features = features.loc[mask].copy()
        sub_target = target.loc[mask].copy()
        block_dates = pd.DatetimeIndex(dates[mask])
        sub_features.index = block_dates
        sub_target.index = block_dates
        order = sub_features.index.argsort(kind="stable")
        return sub_features.iloc[order], sub_target.iloc[order]

    X_train, y_train = block(train_mask)
    X_valid, y_valid = block(valid_mask)
    X_test, y_test = block(test_mask)

    split = ChronologicalSplit(
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test, y_test=y_test,
    )
    bounds = {
        "train": (str(unique_dates[0].date()), str(train_cut.date())),
        "validation": (str(unique_dates[unique_dates.get_loc(train_cut) + 1].date()), str(valid_cut.date())),
        "test": (str(unique_dates[unique_dates.get_loc(valid_cut) + 1].date()), str(unique_dates[-1].date())),
    }
    return split, bounds


def time_series_cv(n_splits: int = CV_SPLITS) -> TimeSeriesSplit:
    """Return an expanding-window cross-validator for model selection."""
    return TimeSeriesSplit(n_splits=n_splits)


def assert_chronological(split: ChronologicalSplit) -> None:
    """Raise if any partition overlaps or precedes an earlier one.

    Raises:
        AssertionError: If the temporal ordering is violated.
    """
    blocks = [
        ("train", split.X_train),
        ("validation", split.X_valid),
        ("test", split.X_test),
    ]
    populated = [(name, frame) for name, frame in blocks if not frame.empty]

    for name, frame in populated:
        if not frame.index.is_monotonic_increasing:
            raise AssertionError(f"{name} partition is not sorted ascending.")

    for (earlier_name, earlier), (later_name, later) in zip(populated, populated[1:]):
        if earlier.index[-1] >= later.index[0]:
            raise AssertionError(
                f"{earlier_name} ends at {earlier.index[-1].date()} which is not "
                f"strictly before {later_name} starting {later.index[0].date()}."
            )
