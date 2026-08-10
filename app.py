"""FinMarket ML - Streamlit application.

Five pages: market overview, technical analysis, machine-learning prediction,
model explainability, and methodology. Data is loaded once per (symbol, mode)
pair and cached; models are trained on demand and cached by the same key.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    BEARISH_THRESHOLD,
    BULLISH_THRESHOLD,
    DISCLAIMER,
    LIVE_UNIVERSE,
    TRANSACTION_COST_BPS,
)
from src.data.loaders import (
    DataMode,
    SnapshotProvider,
    load_market_context,
    load_prices,
    snapshot_metadata,
)
from src.data.validation import DataValidationError
from src.evaluation.explainability import feature_correlation, top_features
from src.evaluation.metrics import roc_points
from src.evaluation.signal import classify_probability, returns_by_signal, run_signal_study
from src.features.builder import build_feature_matrix
from src.features.technical import bollinger_bands, macd, relative_strength_index, simple_moving_average
from src.models.registry import available_models
from src.models.train import run_experiment
from src.utils.stats import summarise_prices
from src.visualization import charts
from src.visualization.theme import INK_MUTED, STATUS_CRITICAL, STATUS_GOOD, STATUS_WARNING

st.set_page_config(
    page_title="FinMarket ML",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  div[data-testid="stMetricValue"] {font-size: 1.45rem;}
  div[data-testid="stMetricLabel"] {color: #898781;}
  .fm-hero {
    border: 1px solid rgba(255,255,255,0.10); border-radius: 12px;
    padding: 1.1rem 1.3rem; margin-bottom: 1.2rem; background: #1a1a19;
  }
  .fm-hero h1 {margin: 0 0 .25rem 0; font-size: 1.55rem; color: #ffffff;}
  .fm-hero p {margin: 0; color: #c3c2b7; font-size: .92rem;}
  .fm-badge {
    display: inline-block; padding: .18rem .6rem; border-radius: 999px;
    font-size: .74rem; font-weight: 600; letter-spacing: .02em; margin-right: .4rem;
  }
  .fm-note {
    border-left: 3px solid #3987e5; padding: .55rem .9rem; margin: .6rem 0 1rem 0;
    background: rgba(57,135,229,0.07); border-radius: 0 8px 8px 0;
    color: #c3c2b7; font-size: .87rem;
  }
  .fm-warn {
    border-left: 3px solid #fab219; padding: .55rem .9rem; margin: .6rem 0 1rem 0;
    background: rgba(250,178,25,0.07); border-radius: 0 8px 8px 0;
    color: #c3c2b7; font-size: .87rem;
  }
  .fm-signal {
    border-radius: 12px; padding: 1.1rem 1.3rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.12); background: #1a1a19;
  }
  .fm-signal .label {font-size: 1.75rem; font-weight: 700; margin: .2rem 0;}
  .fm-signal .sub {color: #898781; font-size: .8rem;}
  footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data and model access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_snapshot_symbols() -> list[str]:
    """Symbols available in the bundled snapshot."""
    return SnapshotProvider().available_symbols()


@st.cache_data(show_spinner=False)
def cached_metadata() -> dict:
    """Provenance metadata for the bundled snapshot."""
    return snapshot_metadata()


@st.cache_data(show_spinner="Loading market data...")
def cached_prices(symbol: str, mode_value: str, years: int) -> tuple[pd.DataFrame, str]:
    """Load an OHLCV frame, returning the mode actually used."""
    frame, effective = load_prices(symbol, mode=DataMode(mode_value), years=years)
    return frame, effective.value


@st.cache_data(show_spinner=False)
def cached_market_returns(mode_value: str, years: int) -> pd.Series | None:
    """Broad-market daily returns for the context features."""
    return load_market_context(DataMode(mode_value), years=years)


@st.cache_resource(show_spinner="Training models...")
def cached_experiment(symbol: str, mode_value: str, years: int, model_keys: tuple[str, ...]):
    """Build features and train the selected models for one ticker."""
    frame, _ = cached_prices(symbol, mode_value, years)
    market = cached_market_returns(mode_value, years)
    matrix = build_feature_matrix(frame, market_returns=market)
    result = run_experiment(matrix, symbol=symbol, model_keys=list(model_keys), run_cv=False)
    return matrix, result


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of Streamlit metrics from (label, value, help) tuples."""
    columns = st.columns(len(items))
    for column, (label, value, helptext) in zip(columns, items):
        column.metric(label, value, help=helptext)


def signal_badge(label: str) -> str:
    """Return an HTML badge coloured by signal state."""
    colors = {
        "Bullish": STATUS_GOOD,
        "Bearish": STATUS_CRITICAL,
        "Neutral": STATUS_WARNING,
    }
    return (
        f'<span class="fm-badge" style="background:{colors.get(label, INK_MUTED)};'
        f'color:#0d0d0d">{label.upper()}</span>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> dict:
    """Render the sidebar and return the user's selections."""
    st.sidebar.markdown("## FinMarket ML")
    st.sidebar.caption("Stock movement prediction & market signal analytics")

    page = st.sidebar.radio(
        "Page",
        ["Market Overview", "Technical Analysis", "ML Prediction",
         "Model Explainability", "About & Methodology"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    mode_label = st.sidebar.radio(
        "Data source",
        ["Bundled snapshot", "Live (Yahoo Finance)"],
        help=(
            "The bundled snapshot is a fixed, reproducible extract of real daily "
            "OHLCV that ships with the repository, so the demo works without "
            "network access. Live mode downloads fresh data via yfinance."
        ),
    )
    mode = DataMode.SNAPSHOT if mode_label.startswith("Bundled") else DataMode.LIVE

    if mode is DataMode.SNAPSHOT:
        symbols = cached_snapshot_symbols()
        default = symbols.index("AAPL") if "AAPL" in symbols else 0
        symbol = st.sidebar.selectbox("Ticker", symbols, index=default)
        years = 0
    else:
        suggestion = st.sidebar.selectbox("Suggested tickers", list(LIVE_UNIVERSE), index=0)
        symbol = st.sidebar.text_input("Ticker symbol", value=suggestion).strip().upper()
        years = st.sidebar.slider("Years of history", 2, 15, 8)

    st.sidebar.divider()
    specs = available_models()
    selected = st.sidebar.multiselect(
        "Models to train",
        options=list(specs),
        default=[k for k in ("logistic_regression", "random_forest", "gradient_boosting")
                 if k in specs],
        format_func=lambda key: specs[key].display_name,
    )
    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER)

    return {"page": page, "symbol": symbol, "mode": mode, "years": years,
            "model_keys": tuple(selected)}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_market_overview(frame: pd.DataFrame, symbol: str, effective_mode: str) -> None:
    """Price, volume, return distribution and headline statistics."""
    st.markdown(
        f'<div class="fm-hero"><h1>Market overview - {symbol}</h1>'
        f"<p>Daily OHLCV, return behaviour and risk statistics for the loaded "
        f"sample.</p></div>",
        unsafe_allow_html=True,
    )

    summary = summarise_prices(frame)
    metric_row([
        ("Sessions", f"{summary.observations:,}", "Number of trading days loaded"),
        ("Period", f"{summary.start_date} to {summary.end_date}", None),
        ("Total return", f"{summary.total_return_pct:+.1f}%",
         "Price change from the first to the last session in the sample"),
        ("Annualised return", f"{summary.annualised_return_pct:+.1f}%",
         "Compound growth rate scaled to a 252-session year"),
    ])
    metric_row([
        ("Annualised volatility", f"{summary.annualised_volatility_pct:.1f}%",
         "Standard deviation of daily returns, scaled by the square root of 252"),
        ("Max drawdown", f"{summary.max_drawdown_pct:.1f}%",
         "Largest peak-to-trough decline in the sample"),
        ("Best day", f"{summary.largest_daily_gain_pct:+.2f}%", None),
        ("Worst day", f"{summary.largest_daily_loss_pct:+.2f}%", None),
    ])

    st.markdown(
        f'<div class="fm-note"><b>Up days:</b> {summary.up_day_share_pct:.1f}% of sessions '
        f"closed higher than the day before. This is the number any direction model "
        f"has to beat &mdash; always predicting &ldquo;up&rdquo; already scores about "
        f"this well.</div>",
        unsafe_allow_html=True,
    )

    view = st.radio("Price view", ["Line", "Candlestick"], horizontal=True,
                    label_visibility="collapsed")
    if view == "Line":
        averages = {
            f"SMA {window}": simple_moving_average(frame["close"], window)
            for window in (20, 50)
        }
        st.plotly_chart(charts.price_chart(frame, symbol, averages), use_container_width=True)
    else:
        window = frame.iloc[-250:] if len(frame) > 250 else frame
        st.plotly_chart(charts.candlestick_chart(window, symbol), use_container_width=True)

    st.plotly_chart(charts.volume_chart(frame, symbol), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.returns_distribution(frame, symbol), use_container_width=True)
        st.plotly_chart(charts.up_down_chart(frame), use_container_width=True)
    with right:
        st.plotly_chart(charts.volatility_chart(frame), use_container_width=True)
        st.plotly_chart(charts.weekday_chart(frame), use_container_width=True)

    st.plotly_chart(charts.monthly_heatmap(frame), use_container_width=True)

    with st.expander("Raw data (last 250 sessions)"):
        st.dataframe(frame.tail(250).sort_index(ascending=False), use_container_width=True)
    st.caption(f"Data source in use: {effective_mode}.")


def page_technical(frame: pd.DataFrame, symbol: str) -> None:
    """Moving averages, RSI, MACD, Bollinger Bands and volatility."""
    st.markdown(
        f'<div class="fm-hero"><h1>Technical analysis - {symbol}</h1>'
        f"<p>The indicators that feed the model, drawn as a human would read "
        f"them.</p></div>",
        unsafe_allow_html=True,
    )

    close = frame["close"].astype(float)
    rsi = relative_strength_index(close)
    macd_frame = macd(close)
    bands = bollinger_bands(close)

    latest_rsi = float(rsi.dropna().iloc[-1]) if rsi.notna().any() else float("nan")
    latest_bb = float(bands["bb_position"].dropna().iloc[-1]) if bands["bb_position"].notna().any() else float("nan")
    latest_macd_hist = float(macd_frame["macd_hist"].dropna().iloc[-1])
    sma20 = simple_moving_average(close, 20)
    close_to_sma20 = float(close.iloc[-1] / sma20.dropna().iloc[-1] - 1) * 100

    metric_row([
        ("RSI (14)", f"{latest_rsi:.1f}",
         "Above 70 is conventionally 'overbought', below 30 'oversold'"),
        ("MACD histogram", f"{latest_macd_hist:+.3f}",
         "Positive means short-term momentum is running ahead of the longer trend"),
        ("Bollinger position", f"{latest_bb:.2f}",
         "0 sits on the lower band, 1 on the upper band"),
        ("Close vs SMA20", f"{close_to_sma20:+.2f}%",
         "How far the last close sits above or below its 20-day average"),
    ])

    tabs = st.tabs(["Moving averages", "RSI", "MACD", "Bollinger Bands", "Volatility"])

    with tabs[0]:
        st.markdown(
            '<div class="fm-note"><b>Moving averages</b> smooth out day-to-day noise. '
            "The model never sees the raw average: it sees ratios such as "
            "<code>close / SMA20 - 1</code>, which are comparable across different "
            "price levels and different companies.</div>",
            unsafe_allow_html=True,
        )
        averages = {f"SMA {w}": simple_moving_average(close, w) for w in (5, 20, 50)}
        st.plotly_chart(charts.price_chart(frame, symbol, averages), use_container_width=True)

    with tabs[1]:
        st.markdown(
            '<div class="fm-note"><b>RSI (Relative Strength Index)</b> compares the size '
            "of recent gains with recent losses on a 0-100 scale. It is used here purely "
            "as a momentum feature, not as a buy or sell rule.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.rsi_chart(rsi), use_container_width=True)

    with tabs[2]:
        st.markdown(
            '<div class="fm-note"><b>MACD</b> is the gap between a fast (12-day) and a '
            "slow (26-day) exponential average, plus a 9-day signal line. The histogram "
            "is the difference between the two.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.macd_chart(macd_frame), use_container_width=True)

    with tabs[3]:
        st.markdown(
            '<div class="fm-note"><b>Bollinger Bands</b> sit two standard deviations '
            "either side of the 20-day average, so they widen when the market gets "
            "volatile and narrow when it calms down.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.bollinger_chart(close, bands), use_container_width=True)

    with tabs[4]:
        st.markdown(
            '<div class="fm-note"><b>Realised volatility</b> is the standard deviation of '
            "recent daily returns, annualised. Volatility clusters: calm periods follow "
            "calm periods, and turbulent ones follow turbulence.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.volatility_chart(frame, windows=(5, 10, 20)),
                        use_container_width=True)


def page_prediction(symbol: str, mode: DataMode, years: int, model_keys: tuple[str, ...]) -> None:
    """Train models, show metrics, and render the latest research signal."""
    st.markdown(
        f'<div class="fm-hero"><h1>ML prediction - {symbol}</h1>'
        f"<p>Will the next session close higher than this one? A binary "
        f"classification problem, validated strictly forward in time.</p></div>",
        unsafe_allow_html=True,
    )

    if not model_keys:
        st.warning("Select at least one model in the sidebar.")
        return

    matrix, result = cached_experiment(symbol, mode.value, years, model_keys)
    split = result.split
    periods = split.periods

    st.markdown(
        f'<div class="fm-note"><b>Target:</b> <code>1</code> when '
        f"<code>close(t+1) &gt; close(t)</code>, otherwise <code>0</code>. Every feature "
        f"for session <i>t</i> uses only information available at that session&rsquo;s "
        f"close, and the data is split strictly by date &mdash; never shuffled.</div>",
        unsafe_allow_html=True,
    )

    metric_row([
        ("Observations", f"{result.n_observations:,}", None),
        ("Features", f"{len(result.feature_names)}", None),
        ("Train window", f"{periods['train'][0]} to {periods['train'][1]}",
         f"{split.sizes['train']:,} sessions"),
        ("Test window", f"{periods['test'][0]} to {periods['test'][1]}",
         f"{split.sizes['test']:,} sessions"),
    ])

    st.subheader("Model comparison")
    table = result.comparison_table()
    display = table.drop(columns=["key"]).copy()
    st.dataframe(
        display.style.format(
            {c: "{:.4f}" for c in display.columns if display[c].dtype.kind == "f"},
            na_rep="-",
        ),
        use_container_width=True,
    )

    best = result.best_model
    baseline_accuracy = result.best_baseline_accuracy
    beats = best.test_metrics.accuracy > baseline_accuracy
    auc_text = f"{best.test_metrics.roc_auc:.4f}" if best.test_metrics.roc_auc else "n/a"
    verdict_class = "fm-note" if beats else "fm-warn"
    verdict = (
        f"beats the strongest naive baseline ({baseline_accuracy:.4f})"
        if beats
        else f"does <b>not</b> beat the strongest naive baseline ({baseline_accuracy:.4f})"
    )
    st.markdown(
        f'<div class="{verdict_class}"><b>{best.display_name}</b> is the strongest model on '
        f"the test window: ROC-AUC {auc_text}, accuracy {best.test_metrics.accuracy:.4f}. "
        f"On accuracy it {verdict}. Daily direction is close to unpredictable, so results "
        f"near 0.50 are the expected outcome rather than a bug.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(charts.model_comparison_chart(table), use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        curves = {}
        for model in result.models.values():
            fpr, tpr = roc_points(split.y_test, model.test_probabilities)
            curves[model.display_name] = (fpr, tpr, model.test_metrics.roc_auc)
        st.plotly_chart(charts.roc_curve_chart(curves), use_container_width=True)
    with right:
        st.plotly_chart(charts.confusion_chart(best.confusion_matrix), use_container_width=True)
        st.caption(f"Confusion matrix for {best.display_name} on the test window.")

    st.plotly_chart(
        charts.probability_chart(best.test_probabilities, split.y_test),
        use_container_width=True,
    )

    st.subheader("Latest model output")
    latest_probability = float(best.test_probabilities.iloc[-1])
    latest_date = best.test_probabilities.index[-1]
    label = classify_probability(latest_probability).value

    columns = st.columns([2, 3])
    with columns[0]:
        st.markdown(
            f'<div class="fm-signal">{signal_badge(label)}'
            f'<div class="label">{latest_probability * 100:.1f}%</div>'
            f'<div class="sub">probability the next session closes higher<br>'
            f"as of {latest_date.date()}</div></div>",
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"**P(up) = {latest_probability:.3f}** &nbsp;&nbsp; "
            f"**P(down) = {1 - latest_probability:.3f}**\n\n"
            f"The research signal reads *Bullish* above {BULLISH_THRESHOLD:.2f}, "
            f"*Bearish* below {BEARISH_THRESHOLD:.2f}, and *Neutral* in between."
        )
        st.markdown(
            f'<div class="fm-warn">This is a model output on historical data, not a '
            f"trading recommendation. {DISCLAIMER}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Research signal study (illustrative only)"):
        st.markdown(
            "What would have happened if a long position were held only on sessions "
            f"where the model's P(up) exceeded {BULLISH_THRESHOLD:.2f}? The comparison "
            f"below charges {TRANSACTION_COST_BPS:.0f} basis points on every change of "
            "position and covers the test window only. It ignores slippage, liquidity, "
            "taxes and the fact that one realised path is a sample of size one."
        )
        forward = matrix.next_day_return.reindex(best.test_probabilities.index)
        try:
            study = run_signal_study(best.test_probabilities, forward)
            metric_row([
                ("Signal rule total return", f"{study.strategy_total_return_pct:+.2f}%", None),
                ("Buy and hold total return", f"{study.buy_hold_total_return_pct:+.2f}%", None),
                ("Days in market", f"{study.days_in_market_pct:.1f}%", None),
                ("Position changes", f"{study.n_position_changes:,}", None),
            ])
            bucket = returns_by_signal(best.test_probabilities, forward)
            st.plotly_chart(charts.signal_return_chart(bucket), use_container_width=True)

            position = (best.test_probabilities > BULLISH_THRESHOLD).astype(float)
            turnover = position.diff().abs().fillna(position.iloc[0])
            strategy_returns = position * forward - turnover * (TRANSACTION_COST_BPS / 10_000)
            st.plotly_chart(
                charts.equity_curve_chart(strategy_returns.dropna(), forward.dropna()),
                use_container_width=True,
            )
        except ValueError as exc:
            st.info(f"Signal study unavailable: {exc}")

    if result.warnings:
        st.caption("Warnings: " + "; ".join(result.warnings))


def page_explainability(symbol: str, mode: DataMode, years: int, model_keys: tuple[str, ...]) -> None:
    """Feature importance, correlation structure and model comparison."""
    st.markdown(
        f'<div class="fm-hero"><h1>Model explainability - {symbol}</h1>'
        f"<p>Which inputs move the predictions, and how much do they overlap?</p></div>",
        unsafe_allow_html=True,
    )

    if not model_keys:
        st.warning("Select at least one model in the sidebar.")
        return

    matrix, result = cached_experiment(symbol, mode.value, years, model_keys)
    specs = available_models()

    chosen_key = st.selectbox(
        "Model to explain",
        list(result.models),
        format_func=lambda key: result.models[key].display_name,
    )
    model = result.models[chosen_key]
    st.caption(specs[chosen_key].description)

    importance, method = top_features(
        model.pipeline,
        result.feature_names,
        features=result.split.X_test,
        target=result.split.y_test,
        top_n=15,
    )
    signed = specs[chosen_key].supports_coefficients

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            charts.feature_importance_chart(
                importance,
                title=f"Top 15 features - {model.display_name} ({method})",
                signed=signed,
            ),
            use_container_width=True,
        )
    with right:
        st.markdown(f"**Method:** {method} importance")
        if signed:
            st.markdown(
                "For a linear model the sign matters: a positive coefficient pushes the "
                "prediction towards an up day, a negative one towards a down day. Bars "
                "are coloured by sign and the exact value is in the hover."
            )
        else:
            st.markdown(
                "Tree ensembles report how much each feature reduced impurity across "
                "the forest. This says which features the model *used*, not the "
                "direction of the effect."
            )
        st.dataframe(
            importance.head(10)[["feature", "importance"]].style.format({"importance": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Permutation importance on the test window")
    st.markdown(
        '<div class="fm-note">Permutation importance shuffles one feature at a time and '
        "measures how much test ROC-AUC falls. It is the most honest of the three views "
        "because it is computed on data the model never trained on: values near zero mean "
        "the feature carried no usable information out of sample.</div>",
        unsafe_allow_html=True,
    )
    if st.button("Compute permutation importance", type="primary"):
        with st.spinner("Permuting features..."):
            from src.evaluation.explainability import permutation_feature_importance

            permuted = permutation_feature_importance(
                model.pipeline, result.split.X_test, result.split.y_test, n_repeats=8
            )
        st.plotly_chart(
            charts.feature_importance_chart(
                permuted.head(15), title="Permutation importance (drop in test ROC-AUC)"
            ),
            use_container_width=True,
        )

    st.subheader("Feature correlation")
    st.markdown(
        '<div class="fm-note">Strongly correlated features carry overlapping information, '
        "which splits importance between them and makes single-feature attributions less "
        "reliable. This is why momentum, volatility and moving-average blocks are kept "
        "deliberately small.</div>",
        unsafe_allow_html=True,
    )
    correlation = feature_correlation(matrix.features, top_n=20)
    st.plotly_chart(charts.correlation_heatmap(correlation), use_container_width=True)

    with st.expander("SHAP summary (optional)"):
        st.caption(
            "SHAP is used when the package is installed and the selected model is "
            "tree-based. It is never required to run this application."
        )
        if st.button("Compute SHAP values"):
            from src.evaluation.explainability import shap_summary

            with st.spinner("Computing SHAP values..."):
                shap_frame = shap_summary(model.pipeline, result.split.X_test)
            if shap_frame is None:
                st.info("SHAP is unavailable for this model or not installed.")
            else:
                renamed = shap_frame.rename(columns={"mean_abs_shap": "importance"})
                st.plotly_chart(
                    charts.feature_importance_chart(
                        renamed.head(15), title="Mean absolute SHAP value"
                    ),
                    use_container_width=True,
                )


def page_about(metadata: dict) -> None:
    """Methodology, data provenance, limitations and disclaimer."""
    st.markdown(
        '<div class="fm-hero"><h1>About &amp; methodology</h1>'
        "<p>How the dataset is built, how the models are validated, and what this "
        "project does not claim.</p></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Problem framing")
    st.markdown(
        """
Predicting an exact future price is a regression problem with an extremely poor
signal-to-noise ratio, and a model can score well on it while being useless
directionally — simply predicting "tomorrow's price is roughly today's price"
produces a low error and no information.

This project instead asks a binary question: **will the next session close
higher than this one?** The label is defined as `1` when
`close(t+1) > close(t)` and `0` otherwise. Framing it this way makes the result
directly measurable against naive benchmarks and keeps the evaluation honest.
        """
    )

    st.subheader("Data")
    if metadata:
        st.markdown(
            f"""
- **Source:** {metadata.get('source_name', 'n/a')}
- **Coverage:** {metadata.get('start_date')} to {metadata.get('end_date')},
  {metadata.get('sessions_per_equity', 0):,} sessions per equity
- **Universe:** {metadata.get('n_equities', 0)} equities plus a
  `{metadata.get('market_index_symbol', 'MKT_EW')}` equal-weighted index built from
  {metadata.get('market_index_constituents', 0)} full-history constituents
- **Rows:** {metadata.get('rows', 0):,}
- **Adjustment:** {metadata.get('adjustment_note', 'n/a')}
            """
        )
        st.caption(f"Source URL: {metadata.get('source_url', 'n/a')}")
    st.markdown(
        "Live mode downloads daily OHLCV from Yahoo Finance through `yfinance`. "
        "If a live download fails for any reason the application silently falls back "
        "to the bundled snapshot, so no page ever dead-ends."
    )

    st.subheader("Leakage control")
    st.markdown(
        """
Three rules are enforced in code and covered by the test-suite:

1. **Causal features only.** Every indicator at session *t* is computed from a
   trailing window ending at *t*. No centred rolling windows, no forward fills
   from the future, no target-derived columns.
2. **Chronological splitting.** Train, validation and test blocks are contiguous
   and strictly ordered; `assert_chronological` fails the run if a later block
   ever starts before an earlier one ends. For the pooled multi-ticker dataset
   the cut is made on *calendar dates*, so one session can never straddle two
   partitions.
3. **Scaling inside the pipeline.** `StandardScaler` sits inside the
   scikit-learn `Pipeline`, so statistics are fitted on training folds only.
        """
    )

    st.subheader("Evaluation")
    st.markdown(
        """
Models are scored with accuracy, precision, recall, F1 and ROC-AUC, and compared
against two naive benchmarks:

- **Majority class** — always predict whichever direction dominated the training
  window. Equities rise on slightly more than half of sessions, so this scores
  above 50% before any modelling happens.
- **Momentum persistence** — predict that tomorrow repeats today's direction.

A model is only interesting if it beats both. Results near 0.50 ROC-AUC are the
normal outcome for daily direction prediction and are reported as such rather
than being tuned away.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
- Daily direction on liquid large-cap equities is close to a coin flip. Any edge
  found here is small, unstable across periods, and well within the range that
  transaction costs would erase.
- The bundled snapshot covers a specific five-year window that was, on the whole,
  a rising market. Results should not be extrapolated to other regimes.
- Prices are split-adjusted but not dividend-adjusted, so total-return figures
  understate what a holder would actually have earned.
- The signal study is a teaching device. It ignores slippage, liquidity limits,
  borrow costs, taxes, position sizing and survivorship, and a single realised
  path is a sample of size one.
- No fundamental, macroeconomic or news data is used in this project. The
  companion project, MarketSense AI, covers the news dimension.
        """
    )

    st.subheader("Tech stack")
    st.markdown(
        "Python · pandas · NumPy · scikit-learn · XGBoost · Plotly · Streamlit · "
        "yfinance · joblib · SHAP (optional) · pytest"
    )

    st.divider()
    st.markdown(f"**Disclaimer.** {DISCLAIMER}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the selected page."""
    selections = sidebar()
    page = selections["page"]
    symbol = selections["symbol"]
    mode = selections["mode"]
    years = selections["years"]

    if page == "About & Methodology":
        page_about(cached_metadata())
        return

    if not symbol:
        st.info("Enter a ticker symbol in the sidebar to begin.")
        return

    try:
        frame, effective_mode = cached_prices(symbol, mode.value, years)
    except (KeyError, DataValidationError, FileNotFoundError) as exc:
        st.error(f"Could not load data for '{symbol}'.")
        st.caption(str(exc))
        st.info(
            "Try one of the tickers in the bundled snapshot, or check the symbol "
            "spelling for live mode (for example `AAPL`, `^GSPC`, `RELIANCE.NS`)."
        )
        return

    if mode is DataMode.LIVE and effective_mode == DataMode.SNAPSHOT.value:
        st.warning(
            "Live download was unavailable, so the bundled snapshot is being used "
            "instead. All figures below come from the snapshot.",
            icon="⚠️",
        )

    if page == "Market Overview":
        page_market_overview(frame, symbol, effective_mode)
    elif page == "Technical Analysis":
        page_technical(frame, symbol)
    elif page == "ML Prediction":
        page_prediction(symbol, DataMode(effective_mode), years, selections["model_keys"])
    elif page == "Model Explainability":
        page_explainability(symbol, DataMode(effective_mode), years, selections["model_keys"])


if __name__ == "__main__":
    main()
