"""Illustrative research signal derived from predicted probabilities.

This module exists to make the classifier's output interpretable in market
language, not to propose a trading system. The historical study below is a
teaching device: it ignores slippage, liquidity, borrow costs, taxes, position
sizing and the fact that a single realised path is a sample of size one.

For educational and research purposes only. Not investment advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    BEARISH_THRESHOLD,
    BULLISH_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
    TRANSACTION_COST_BPS,
)


class SignalLabel(str, Enum):
    """Three-state research signal."""

    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"


@dataclass(frozen=True)
class SignalStudy:
    """Outcome of applying the signal rule over a historical window."""

    observations: int
    days_in_market_pct: float
    strategy_total_return_pct: float
    buy_hold_total_return_pct: float
    strategy_annualised_return_pct: float
    buy_hold_annualised_return_pct: float
    strategy_annualised_volatility_pct: float
    buy_hold_annualised_volatility_pct: float
    strategy_max_drawdown_pct: float
    buy_hold_max_drawdown_pct: float
    n_position_changes: int
    total_cost_pct: float
    transaction_cost_bps: float

    def as_dict(self) -> dict[str, Any]:
        """Return the study as a plain dictionary."""
        return asdict(self)


def classify_probability(
    probability: float,
    bullish_threshold: float = BULLISH_THRESHOLD,
    bearish_threshold: float = BEARISH_THRESHOLD,
) -> SignalLabel:
    """Map P(up) onto a Bullish / Neutral / Bearish label.

    Args:
        probability: Model probability that the next session closes higher.
        bullish_threshold: Above this, the signal reads Bullish.
        bearish_threshold: Below this, the signal reads Bearish.

    Returns:
        The corresponding :class:`SignalLabel`.

    Raises:
        ValueError: If the thresholds are inconsistent.
    """
    if bearish_threshold > bullish_threshold:
        raise ValueError("bearish_threshold must not exceed bullish_threshold.")
    if probability > bullish_threshold:
        return SignalLabel.BULLISH
    if probability < bearish_threshold:
        return SignalLabel.BEARISH
    return SignalLabel.NEUTRAL


def signal_series(
    probabilities: pd.Series,
    bullish_threshold: float = BULLISH_THRESHOLD,
    bearish_threshold: float = BEARISH_THRESHOLD,
) -> pd.Series:
    """Vectorised version of :func:`classify_probability`."""
    labels = pd.Series(SignalLabel.NEUTRAL.value, index=probabilities.index, name="signal")
    labels[probabilities > bullish_threshold] = SignalLabel.BULLISH.value
    labels[probabilities < bearish_threshold] = SignalLabel.BEARISH.value
    return labels


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline of a cumulative return path."""
    if returns.empty:
        return 0.0
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def _annualise(returns: pd.Series) -> tuple[float, float]:
    """Return (annualised return, annualised volatility) for a return series."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0, 0.0
    growth = float((1.0 + clean).prod())
    annual_return = growth ** (TRADING_DAYS_PER_YEAR / len(clean)) - 1.0 if growth > 0 else -1.0
    annual_vol = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(clean) > 1 else 0.0
    return annual_return, annual_vol


def run_signal_study(
    probabilities: pd.Series,
    next_day_returns: pd.Series,
    bullish_threshold: float = BULLISH_THRESHOLD,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> SignalStudy:
    """Compare a long-when-bullish rule against buy-and-hold, net of costs.

    The alignment is deliberate and explicit: the probability produced *at the
    close of session t* is multiplied by the return *from t to t+1*. No return
    is ever earned on information published after it.

    Args:
        probabilities: P(up) indexed by session date.
        next_day_returns: Return from each session's close to the next close,
            aligned to the same index.
        bullish_threshold: Probability above which the rule holds a long position.
        transaction_cost_bps: Cost in basis points charged on every change of
            position (entering or exiting).

    Returns:
        A populated :class:`SignalStudy`.

    Raises:
        ValueError: If the two series cannot be aligned.
    """
    aligned = pd.concat(
        [probabilities.rename("p_up"), next_day_returns.rename("fwd_return")], axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError("Probabilities and forward returns do not overlap.")

    position = (aligned["p_up"] > bullish_threshold).astype(float)
    gross = position * aligned["fwd_return"]

    # A cost is charged whenever the position differs from the previous session.
    turnover = position.diff().abs().fillna(position.iloc[0])
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net = gross - cost

    strat_annual, strat_vol = _annualise(net)
    hold_annual, hold_vol = _annualise(aligned["fwd_return"])

    return SignalStudy(
        observations=int(len(aligned)),
        days_in_market_pct=round(float(position.mean()) * 100, 2),
        strategy_total_return_pct=round(float((1 + net).prod() - 1) * 100, 2),
        buy_hold_total_return_pct=round(float((1 + aligned["fwd_return"]).prod() - 1) * 100, 2),
        strategy_annualised_return_pct=round(strat_annual * 100, 2),
        buy_hold_annualised_return_pct=round(hold_annual * 100, 2),
        strategy_annualised_volatility_pct=round(strat_vol * 100, 2),
        buy_hold_annualised_volatility_pct=round(hold_vol * 100, 2),
        strategy_max_drawdown_pct=round(_max_drawdown_from_returns(net) * 100, 2),
        buy_hold_max_drawdown_pct=round(_max_drawdown_from_returns(aligned["fwd_return"]) * 100, 2),
        n_position_changes=int(turnover.sum()),
        total_cost_pct=round(float(cost.sum()) * 100, 3),
        transaction_cost_bps=transaction_cost_bps,
    )


def returns_by_signal(
    probabilities: pd.Series,
    next_day_returns: pd.Series,
    bullish_threshold: float = BULLISH_THRESHOLD,
    bearish_threshold: float = BEARISH_THRESHOLD,
) -> pd.DataFrame:
    """Average realised next-day return within each signal bucket."""
    labels = signal_series(probabilities, bullish_threshold, bearish_threshold)
    frame = pd.concat([labels, next_day_returns.rename("fwd_return")], axis=1).dropna()
    grouped = frame.groupby("signal")["fwd_return"].agg(["count", "mean", "std"])
    grouped["mean"] = grouped["mean"] * 100
    grouped["std"] = grouped["std"] * 100
    grouped.columns = ["observations", "mean_next_day_return_pct", "std_next_day_return_pct"]
    order = [SignalLabel.BEARISH.value, SignalLabel.NEUTRAL.value, SignalLabel.BULLISH.value]
    return grouped.reindex([o for o in order if o in grouped.index]).reset_index()
