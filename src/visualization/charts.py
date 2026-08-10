"""Plotly chart builders used by the Streamlit application.

Each function returns a themed :class:`plotly.graph_objects.Figure`. Charts
follow a small set of rules: one y-axis per chart, a legend whenever more than
one series is present, recessive grid and axis ink, and hover enabled by
default so the reader can interrogate values rather than guess them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.stats import daily_returns, monthly_return_table, returns_by_weekday
from src.visualization.theme import (
    BASELINE,
    DIVERGING,
    DOWN_COLOR,
    GRIDLINE,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
    SERIES,
    UP_COLOR,
    apply_theme,
    base_layout,
    series_color,
)


def price_chart(
    frame: pd.DataFrame, symbol: str, moving_averages: dict[str, pd.Series] | None = None
) -> go.Figure:
    """Closing price with optional moving-average overlays."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            name=f"{symbol} close",
            mode="lines",
            line={"color": series_color(0), "width": 2},
            hovertemplate="%{y:,.2f}<extra>close</extra>",
        )
    )
    for i, (label, series) in enumerate((moving_averages or {}).items(), start=1):
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=label,
                mode="lines",
                line={"color": series_color(i), "width": 1.5},
                hovertemplate="%{y:,.2f}<extra>" + label + "</extra>",
            )
        )
    return apply_theme(
        figure,
        title=f"{symbol} closing price",
        y_title="Price",
        height=420,
        showlegend=True,
    )


def candlestick_chart(frame: pd.DataFrame, symbol: str) -> go.Figure:
    """OHLC candlesticks. Direction is labelled in the hover, not colour alone."""
    figure = go.Figure(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name=symbol,
            increasing={"line": {"color": UP_COLOR}, "fillcolor": UP_COLOR},
            decreasing={"line": {"color": DOWN_COLOR}, "fillcolor": DOWN_COLOR},
        )
    )
    figure = apply_theme(figure, title=f"{symbol} price action", y_title="Price", height=420,
                         showlegend=False)
    figure.update_layout(xaxis_rangeslider_visible=False, hovermode="x")
    return figure


def volume_chart(frame: pd.DataFrame, symbol: str) -> go.Figure:
    """Daily traded volume against its own 20-session average."""
    volume_sma = frame["volume"].rolling(20, min_periods=20).mean()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["volume"],
            name="Volume",
            marker={"color": series_color(0), "line": {"width": 0}},
            hovertemplate="%{y:,.0f}<extra>volume</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=volume_sma,
            name="20-day average",
            mode="lines",
            line={"color": series_color(1), "width": 2},
            hovertemplate="%{y:,.0f}<extra>20d avg</extra>",
        )
    )
    return apply_theme(figure, title=f"{symbol} traded volume", y_title="Shares", height=300)


def returns_distribution(frame: pd.DataFrame, symbol: str) -> go.Figure:
    """Histogram of daily returns with the mean marked."""
    returns = (daily_returns(frame["close"]).dropna() * 100).astype(float)
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=70,
            name="Daily return",
            marker={"color": series_color(0), "line": {"width": 0}},
            hovertemplate="%{x:.2f}%: %{y} sessions<extra></extra>",
        )
    )
    figure.add_vline(
        x=float(returns.mean()),
        line={"color": INK_SECONDARY, "width": 1.5, "dash": "dash"},
        annotation_text=f"mean {returns.mean():.3f}%",
        annotation_font={"color": INK_SECONDARY, "size": 11},
    )
    figure = apply_theme(
        figure,
        title=f"{symbol} daily return distribution",
        x_title="Daily return (%)",
        y_title="Sessions",
        height=320,
        showlegend=False,
    )
    figure.update_layout(hovermode="closest")
    return figure


def volatility_chart(frame: pd.DataFrame, windows: tuple[int, ...] = (5, 20)) -> go.Figure:
    """Annualised rolling volatility over one or more windows."""
    returns = daily_returns(frame["close"])
    figure = go.Figure()
    for i, window in enumerate(windows):
        annualised = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252) * 100
        figure.add_trace(
            go.Scatter(
                x=annualised.index,
                y=annualised,
                name=f"{window}-day",
                mode="lines",
                line={"color": series_color(i), "width": 2},
                hovertemplate="%{y:.1f}%<extra>" + f"{window}-day" + "</extra>",
            )
        )
    return apply_theme(
        figure,
        title="Rolling volatility (annualised)",
        y_title="Volatility (%)",
        height=320,
    )


def rsi_chart(rsi: pd.Series) -> go.Figure:
    """RSI with the conventional 30 / 70 reference lines."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rsi.index,
            y=rsi,
            name="RSI(14)",
            mode="lines",
            line={"color": series_color(0), "width": 2},
            hovertemplate="%{y:.1f}<extra>RSI</extra>",
        )
    )
    for level, label in ((70, "Overbought (70)"), (30, "Oversold (30)")):
        figure.add_hline(
            y=level,
            line={"color": INK_MUTED, "width": 1, "dash": "dot"},
            annotation_text=label,
            annotation_font={"color": INK_MUTED, "size": 10},
        )
    figure = apply_theme(figure, title="Relative Strength Index", y_title="RSI", height=280,
                         showlegend=False)
    figure.update_yaxes(range=[0, 100])
    return figure


def macd_chart(macd_frame: pd.DataFrame) -> go.Figure:
    """MACD line, signal line and histogram on a single scale."""
    figure = go.Figure()
    colors = [UP_COLOR if value >= 0 else DOWN_COLOR for value in macd_frame["macd_hist"].fillna(0)]
    figure.add_trace(
        go.Bar(
            x=macd_frame.index,
            y=macd_frame["macd_hist"],
            name="Histogram",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="%{y:.3f}<extra>histogram</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=macd_frame.index, y=macd_frame["macd"], name="MACD",
            mode="lines", line={"color": series_color(0), "width": 2},
            hovertemplate="%{y:.3f}<extra>MACD</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=macd_frame.index, y=macd_frame["macd_signal"], name="Signal",
            mode="lines", line={"color": series_color(1), "width": 2},
            hovertemplate="%{y:.3f}<extra>signal</extra>",
        )
    )
    return apply_theme(figure, title="MACD (12, 26, 9)", y_title="MACD", height=300)


def bollinger_chart(close: pd.Series, bands: pd.DataFrame) -> go.Figure:
    """Price with its Bollinger envelope drawn as a filled band."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=bands.index, y=bands["bb_upper"], name="Upper band", mode="lines",
                   line={"color": series_color(1), "width": 1}, hoverinfo="skip")
    )
    figure.add_trace(
        go.Scatter(x=bands.index, y=bands["bb_lower"], name="Lower band", mode="lines",
                   line={"color": series_color(1), "width": 1},
                   fill="tonexty", fillcolor="rgba(217,89,38,0.10)", hoverinfo="skip")
    )
    figure.add_trace(
        go.Scatter(x=bands.index, y=bands["bb_middle"], name="20-day mean", mode="lines",
                   line={"color": series_color(2), "width": 1.5, "dash": "dash"},
                   hovertemplate="%{y:,.2f}<extra>20d mean</extra>")
    )
    figure.add_trace(
        go.Scatter(x=close.index, y=close, name="Close", mode="lines",
                   line={"color": series_color(0), "width": 2},
                   hovertemplate="%{y:,.2f}<extra>close</extra>")
    )
    return apply_theme(figure, title="Bollinger Bands (20, 2σ)", y_title="Price", height=380)


def up_down_chart(frame: pd.DataFrame) -> go.Figure:
    """Count of up versus down sessions, labelled directly on the bars."""
    returns = daily_returns(frame["close"]).dropna()
    counts = {"Up days": int((returns > 0).sum()), "Down days": int((returns <= 0).sum())}
    figure = go.Figure(
        go.Bar(
            x=list(counts),
            y=list(counts.values()),
            marker={"color": [UP_COLOR, DOWN_COLOR], "line": {"width": 0}},
            text=[f"{value:,}" for value in counts.values()],
            textposition="outside",
            textfont={"color": INK_SECONDARY, "size": 12},
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    figure = apply_theme(figure, title="Up versus down sessions", y_title="Sessions",
                         height=300, showlegend=False)
    figure.update_layout(hovermode="closest")
    return figure


def monthly_heatmap(frame: pd.DataFrame) -> go.Figure:
    """Year-by-month return heatmap on a diverging scale centred at zero."""
    table = monthly_return_table(frame)
    limit = float(np.nanmax(np.abs(table.to_numpy()))) if table.size else 1.0
    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=list(table.columns),
            y=[str(year) for year in table.index],
            colorscale=[[0.0, "#0d366b"], [0.25, "#3987e5"], [0.5, "#383835"],
                        [0.75, "#d03b3b"], [1.0, "#7a1f1f"]][::-1],
            zmid=0,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": {"text": "%", "font": {"color": INK_MUTED, "size": 11}},
                      "tickfont": {"color": INK_MUTED, "size": 10}},
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    figure = apply_theme(figure, title="Monthly returns (%)", height=300, showlegend=False)
    figure.update_layout(hovermode="closest")
    return figure


def weekday_chart(frame: pd.DataFrame) -> go.Figure:
    """Average daily return by day of week, with counts in the hover."""
    table = returns_by_weekday(frame)
    figure = go.Figure(
        go.Bar(
            x=table["weekday"],
            y=table["mean_return_pct"],
            marker={
                "color": [UP_COLOR if v >= 0 else DOWN_COLOR for v in table["mean_return_pct"]],
                "line": {"width": 0},
            },
            customdata=table["observations"],
            hovertemplate="%{x}: %{y:.3f}%<br>%{customdata:,} sessions<extra></extra>",
        )
    )
    figure = apply_theme(figure, title="Average return by weekday", y_title="Mean return (%)",
                         height=300, showlegend=False)
    figure.update_layout(hovermode="closest")
    return figure


def correlation_heatmap(matrix: pd.DataFrame, title: str = "Feature correlation") -> go.Figure:
    """Correlation matrix on a diverging scale."""
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=DIVERGING,
            zmid=0, zmin=-1, zmax=1,
            colorbar={"tickfont": {"color": INK_MUTED, "size": 10}},
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            xgap=1, ygap=1,
        )
    )
    figure = apply_theme(figure, title=title, height=560, showlegend=False)
    figure.update_layout(hovermode="closest", margin={"l": 130, "r": 24, "t": 48, "b": 130})
    figure.update_xaxes(tickangle=-45, tickfont={"size": 9, "color": INK_MUTED})
    figure.update_yaxes(tickfont={"size": 9, "color": INK_MUTED})
    return figure


def roc_curve_chart(curves: dict[str, tuple[np.ndarray, np.ndarray, float | None]]) -> go.Figure:
    """Overlaid ROC curves with the no-skill diagonal for reference."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], name="No skill (0.500)", mode="lines",
            line={"color": INK_MUTED, "width": 1, "dash": "dash"}, hoverinfo="skip",
        )
    )
    for i, (label, (fpr, tpr, auc)) in enumerate(curves.items()):
        suffix = f" ({auc:.3f})" if auc is not None else ""
        figure.add_trace(
            go.Scatter(
                x=fpr, y=tpr, name=f"{label}{suffix}", mode="lines",
                line={"color": series_color(i), "width": 2},
                hovertemplate="FPR %{x:.3f}, TPR %{y:.3f}<extra>" + label + "</extra>",
            )
        )
    figure = apply_theme(
        figure, title="ROC curves (test window)",
        x_title="False positive rate", y_title="True positive rate", height=420,
    )
    figure.update_layout(hovermode="closest")
    return figure


def confusion_chart(matrix: np.ndarray, labels: tuple[str, str] = ("Down", "Up")) -> go.Figure:
    """Confusion matrix with counts printed in every cell."""
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"Predicted {labels[0]}", f"Predicted {labels[1]}"],
            y=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
            colorscale=SEQUENTIAL_BLUE,
            showscale=False,
            text=matrix,
            texttemplate="%{text:,}",
            textfont={"size": 18, "color": "#ffffff"},
            hovertemplate="%{y}, %{x}: %{z:,}<extra></extra>",
            xgap=3, ygap=3,
        )
    )
    figure = apply_theme(figure, title="Confusion matrix", height=320, showlegend=False)
    figure.update_layout(hovermode="closest")
    return figure


def probability_chart(probabilities: pd.Series, actual: pd.Series | None = None) -> go.Figure:
    """Predicted P(up) through the test window, with the 0.5 decision line."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=probabilities.index, y=probabilities, name="P(up) next session",
            mode="lines", line={"color": series_color(0), "width": 1.6},
            hovertemplate="%{y:.3f}<extra>P(up)</extra>",
        )
    )
    if actual is not None:
        aligned = actual.reindex(probabilities.index)
        for value, label, color in ((1, "Actual up", UP_COLOR), (0, "Actual down", DOWN_COLOR)):
            mask = aligned == value
            figure.add_trace(
                go.Scatter(
                    x=probabilities.index[mask], y=probabilities[mask], name=label,
                    mode="markers",
                    marker={"color": color, "size": 5,
                            "line": {"color": "rgba(26,26,25,0.8)", "width": 1}},
                    hovertemplate="%{y:.3f}<extra>" + label + "</extra>",
                )
            )
    figure.add_hline(
        y=0.5, line={"color": INK_MUTED, "width": 1, "dash": "dash"},
        annotation_text="Decision threshold 0.50",
        annotation_font={"color": INK_MUTED, "size": 10},
    )
    return apply_theme(figure, title="Predicted probability of an up day",
                       y_title="P(up)", height=360)


def feature_importance_chart(
    frame: pd.DataFrame, title: str = "Top predictive features", signed: bool = False
) -> go.Figure:
    """Horizontal bar chart of feature importances, most influential on top."""
    ordered = frame.sort_values("importance")
    if signed and "signed_value" in ordered.columns:
        colors = [UP_COLOR if v > 0 else DOWN_COLOR for v in ordered["signed_value"]]
        hover = "%{y}: %{customdata:.4f}<extra></extra>"
        custom = ordered["signed_value"]
    else:
        colors = series_color(0)
        hover = "%{y}: %{x:.5f}<extra></extra>"
        custom = ordered["importance"]

    figure = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            customdata=custom,
            hovertemplate=hover,
        )
    )
    figure = apply_theme(figure, title=title, x_title="Importance",
                         height=max(320, 24 * len(ordered) + 90), showlegend=False)
    figure.update_layout(hovermode="closest", margin={"l": 170, "r": 24, "t": 48, "b": 44})
    figure.update_yaxes(tickfont={"size": 11, "color": INK_SECONDARY})
    return figure


def model_comparison_chart(table: pd.DataFrame, metric: str = "test_roc_auc") -> go.Figure:
    """Compare models and baselines on one metric, sorted best to worst."""
    clean = table.dropna(subset=[metric]).sort_values(metric)
    is_baseline = clean["key"].astype(str).str.startswith("baseline_")
    colors = [INK_MUTED if flag else series_color(0) for flag in is_baseline]

    figure = go.Figure(
        go.Bar(
            x=clean[metric],
            y=clean["model"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{value:.4f}" for value in clean[metric]],
            textposition="outside",
            textfont={"color": INK_SECONDARY, "size": 11},
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    if metric.endswith("roc_auc"):
        figure.add_vline(x=0.5, line={"color": INK_MUTED, "width": 1, "dash": "dash"},
                         annotation_text="No skill (0.50)",
                         annotation_position="bottom right",
                         annotation_font={"color": INK_MUTED, "size": 10})
    figure = apply_theme(
        figure, title=f"Model comparison - {metric.replace('_', ' ')}",
        x_title=metric.replace("_", " "), height=max(300, 34 * len(clean) + 100),
        showlegend=False,
    )
    figure.update_layout(hovermode="closest", margin={"l": 210, "r": 60, "t": 48, "b": 44})
    return figure


def signal_return_chart(table: pd.DataFrame) -> go.Figure:
    """Average realised next-day return within each signal bucket."""
    figure = go.Figure(
        go.Bar(
            x=table["signal"],
            y=table["mean_next_day_return_pct"],
            marker={
                "color": [UP_COLOR if v >= 0 else DOWN_COLOR
                          for v in table["mean_next_day_return_pct"]],
                "line": {"width": 0},
            },
            text=[f"{v:.3f}%" for v in table["mean_next_day_return_pct"]],
            textposition="outside",
            textfont={"color": INK_SECONDARY, "size": 11},
            customdata=table["observations"],
            hovertemplate="%{x}: %{y:.3f}%<br>%{customdata:,} sessions<extra></extra>",
        )
    )
    figure = apply_theme(
        figure, title="Realised next-day return by research signal",
        y_title="Mean next-day return (%)", height=320, showlegend=False,
    )
    figure.update_layout(hovermode="closest")
    return figure


def equity_curve_chart(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> go.Figure:
    """Cumulative growth of the signal rule against buy-and-hold."""
    figure = go.Figure()
    for i, (label, returns) in enumerate(
        (("Signal rule (net of costs)", strategy_returns), ("Buy and hold", benchmark_returns))
    ):
        curve = (1 + returns.fillna(0)).cumprod()
        figure.add_trace(
            go.Scatter(
                x=curve.index, y=curve, name=label, mode="lines",
                line={"color": series_color(i), "width": 2},
                hovertemplate="%{y:.3f}x<extra>" + label + "</extra>",
            )
        )
    return apply_theme(
        figure, title="Illustrative growth of 1 unit (test window only)",
        y_title="Cumulative growth", height=380,
    )
