"""Validation rules applied to every OHLCV frame before it reaches the models."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import MIN_ROWS_REQUIRED
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataValidationError(ValueError):
    """Raised when a price frame is unusable for downstream modelling."""


@dataclass
class ValidationReport:
    """Outcome of validating a single OHLCV frame."""

    symbol: str
    rows: int
    passed: bool
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "OK" if self.passed else "FAILED"
        return f"[{status}] {self.symbol}: {self.rows} rows, {len(self.warnings)} warning(s)"


def validate_price_frame(
    frame: pd.DataFrame,
    symbol: str,
    min_rows: int = MIN_ROWS_REQUIRED,
    strict: bool = True,
) -> ValidationReport:
    """Check structural and numerical sanity of an OHLCV frame.

    The checks are intentionally conservative: they catch the failure modes that
    silently corrupt a time-series model (unsorted dates, duplicated sessions,
    non-positive prices, high < low) rather than trying to clean the data.

    Args:
        frame: Frame indexed by ``DatetimeIndex`` with OHLCV columns.
        symbol: Ticker the frame belongs to, used for messages.
        min_rows: Minimum number of usable rows.
        strict: When True, raise on failure instead of only reporting it.

    Returns:
        A :class:`ValidationReport`.

    Raises:
        DataValidationError: If ``strict`` is True and a hard check fails.
    """
    report = ValidationReport(symbol=symbol, rows=int(len(frame)), passed=True)

    def fail(message: str) -> None:
        report.passed = False
        report.warnings.append(message)

    if frame.empty:
        fail("frame is empty")
    else:
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            fail(f"missing required columns: {missing}")

        if not isinstance(frame.index, pd.DatetimeIndex):
            fail("index is not a DatetimeIndex")
        else:
            if not frame.index.is_monotonic_increasing:
                fail("index is not sorted ascending")
            duplicates = int(frame.index.duplicated().sum())
            if duplicates:
                fail(f"{duplicates} duplicated timestamp(s)")

        if not missing:
            if len(frame) < min_rows:
                fail(f"only {len(frame)} rows, need at least {min_rows}")

            prices = frame[["open", "high", "low", "close"]]
            if (prices <= 0).to_numpy().any():
                fail("non-positive price(s) present")
            bad_range = int((frame["high"] < frame["low"]).sum())
            if bad_range:
                fail(f"{bad_range} row(s) with high < low")
            if (frame["volume"] < 0).any():
                fail("negative volume present")

            null_share = float(frame[list(REQUIRED_COLUMNS)].isna().to_numpy().mean())
            if null_share > 0.02:
                fail(f"{null_share:.1%} of OHLCV cells are null")
            elif null_share > 0:
                report.warnings.append(f"{null_share:.2%} null cells (tolerated)")

    if not report.passed:
        logger.warning("Validation failed for %s: %s", symbol, "; ".join(report.warnings))
        if strict:
            raise DataValidationError(f"{symbol}: {'; '.join(report.warnings)}")
    else:
        logger.debug("Validation passed for %s (%d rows)", symbol, len(frame))
    return report
