"""Smoke tests for the Streamlit application.

``AppTest`` executes the real script in-process, so these tests catch import
errors, bad widget wiring and page-level exceptions without a browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import SNAPSHOT_FILE

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PAGES = [
    "Market Overview",
    "Technical Analysis",
    "ML Prediction",
    "Model Explainability",
    "About & Methodology",
]

pytestmark = pytest.mark.skipif(
    not SNAPSHOT_FILE.exists(), reason="bundled snapshot not present in this checkout"
)


def _run_page(page: str, timeout: int = 300):
    """Run the app, select ``page`` in the sidebar, and return the app state."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
    app.run()
    assert not app.exception, f"app failed on first render: {app.exception}"

    app.sidebar.radio[0].set_value(page).run()
    return app


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str) -> None:
    """Every page must render cleanly against the bundled snapshot."""
    app = _run_page(page)
    assert not app.exception, f"{page} raised: {app.exception}"


def test_default_page_shows_headline_metrics() -> None:
    """The landing page must show the market-overview metric row."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=300)
    app.run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Sessions" in labels
    assert "Annualised volatility" in labels


def test_prediction_page_reports_model_comparison() -> None:
    """The ML page must produce a comparison table and a probability readout."""
    app = _run_page("ML Prediction")
    assert not app.exception
    assert len(app.dataframe) >= 1, "expected a model-comparison table"
    labels = [metric.label for metric in app.metric]
    assert "Observations" in labels
    assert "Test window" in labels


def test_about_page_lists_data_provenance() -> None:
    """The methodology page must state where the bundled data came from."""
    app = _run_page("About & Methodology")
    assert not app.exception
    body = " ".join(
        element.value
        for element in list(app.markdown) + list(app.subheader) + list(app.header)
    ).lower()
    assert "leakage" in body
    assert "chronological" in body
    assert "investment advice" in body


def test_unknown_ticker_is_handled_gracefully() -> None:
    """A bad symbol in live mode must produce an error message, not a crash."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=300)
    app.run()
    app.sidebar.radio[1].set_value("Live (Yahoo Finance)").run()
    app.sidebar.text_input[0].set_value("NOT_A_REAL_TICKER_XYZ").run()
    assert not app.exception, f"bad ticker raised: {app.exception}"
    # Either a live-fallback warning or a load error is acceptable; a crash is not.
    assert app.error or app.warning or app.info
