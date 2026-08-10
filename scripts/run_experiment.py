"""End-to-end experiment: build features, train every model, write reports.

Two experiments are run:

``panel``
    One model trained across every ticker in the snapshot. This is the headline
    result quoted in the README, because pooling gives tens of thousands of
    observations instead of ~1,200 and therefore a far more stable estimate of
    whether the features carry any signal at all.

``per-ticker``
    The same pipeline applied to each ticker on its own, which is what the
    Streamlit app shows. Reported as a distribution rather than a single number.

Outputs land in ``reports/`` as JSON and Markdown, and the fitted panel model is
persisted to ``models/``.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --tickers AAPL MSFT --skip-per-ticker
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import MODEL_DIR, REPORT_DIR, SNAPSHOT_UNIVERSE
from src.data.loaders import DataMode, SnapshotProvider, load_market_context, snapshot_metadata
from src.evaluation.explainability import top_features
from src.evaluation.signal import run_signal_study
from src.features.builder import build_feature_matrix
from src.features.panel import build_panel
from src.models.train import run_experiment, run_panel_experiment
from src.utils.logging_utils import get_logger

logger = get_logger("run_experiment")


def _json_safe(value):
    """Convert numpy/pandas scalars into JSON-serialisable Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return value


# A kernel SVM is O(n^2) in the number of samples, so it is trained per ticker
# (~1,200 rows) but omitted from the pooled panel (~47,000 rows).
PANEL_MODELS: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
)


def run_panel(tickers: list[str], model_keys: list[str] | None = None) -> dict:
    """Run the pooled cross-sectional experiment and return a report dict."""
    logger.info("=== PANEL EXPERIMENT: %d tickers ===", len(tickers))
    panel = build_panel(tickers, mode=DataMode.SNAPSHOT)
    result, bounds = run_panel_experiment(
        panel, model_keys=list(model_keys or PANEL_MODELS), run_cv=False
    )

    best = result.best_model
    importance, method = top_features(
        best.pipeline,
        result.feature_names,
        features=result.split.X_test,
        target=result.split.y_test,
        top_n=15,
    )

    study = run_signal_study(
        best.test_probabilities.groupby(level=0).mean(),
        panel.next_day_return.groupby(panel.dates).mean().reindex(
            best.test_probabilities.index.unique()
        ),
    )

    model_path = MODEL_DIR / "panel_best_model.joblib"
    joblib.dump(
        {
            "pipeline": best.pipeline,
            "model_key": best.key,
            "feature_names": result.feature_names,
            "trained_on": bounds,
            "tickers": panel.included_symbols,
        },
        model_path,
    )
    logger.info("Saved best panel model to %s", model_path)

    return {
        "experiment": "panel",
        "n_observations": int(len(panel)),
        "n_tickers": len(panel.included_symbols),
        "tickers": panel.included_symbols,
        "skipped": panel.skipped,
        "n_features": len(result.feature_names),
        "feature_names": result.feature_names,
        "date_range": [str(panel.date_range[0].date()), str(panel.date_range[1].date())],
        "split_sizes": result.split.sizes,
        "split_periods": bounds,
        "class_balance_train": float(result.split.y_train.mean()),
        "class_balance_test": float(result.split.y_test.mean()),
        "models": {
            key: {
                "display_name": model.display_name,
                "validation": model.validation_metrics.as_dict(),
                "test": model.test_metrics.as_dict(),
                "confusion_matrix": model.confusion_matrix.tolist(),
                "fit_seconds": round(model.fit_seconds, 2),
            }
            for key, model in result.models.items()
        },
        "baselines": {name: metrics.as_dict() for name, metrics in result.baselines.items()},
        "best_model": {
            "key": best.key,
            "display_name": best.display_name,
            "test": best.test_metrics.as_dict(),
        },
        "top_features": importance.to_dict(orient="records"),
        "importance_method": method,
        "signal_study": study.as_dict(),
        "warnings": result.warnings,
    }


def run_per_ticker(tickers: list[str]) -> dict:
    """Run the single-ticker experiment for each symbol and summarise."""
    logger.info("=== PER-TICKER EXPERIMENTS: %d tickers ===", len(tickers))
    provider = SnapshotProvider()
    market_returns = load_market_context(DataMode.SNAPSHOT)

    rows: list[dict] = []
    failures: dict[str, str] = {}

    for symbol in tickers:
        try:
            frame = provider.fetch(symbol)
            matrix = build_feature_matrix(frame, market_returns=market_returns)
            result = run_experiment(matrix, symbol=symbol, run_cv=False)
            baselines = {
                "majority_baseline_accuracy": result.baselines["majority_class"].accuracy,
                "momentum_baseline_accuracy": result.baselines["momentum_persistence"].accuracy,
            }
            # Record *every* model, not just the winner. Reporting only the best
            # model per ticker would be selection on the test set and would
            # overstate performance; the honest figure is a fixed model's median.
            for key, model in result.models.items():
                rows.append(
                    {
                        "symbol": symbol,
                        "observations": len(matrix),
                        "model_key": key,
                        "model": model.display_name,
                        "test_accuracy": model.test_metrics.accuracy,
                        "test_f1": model.test_metrics.f1,
                        "test_roc_auc": model.test_metrics.roc_auc,
                        **baselines,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.warning("Per-ticker run failed for %s: %s", symbol, exc)
            failures[symbol] = str(exc)

    table = pd.DataFrame(rows)
    summary: dict = {
        "experiment": "per_ticker",
        "n_tickers": int(table["symbol"].nunique()) if not table.empty else 0,
        "failures": failures,
        "selection_note": (
            "Per-model medians are reported across tickers. The 'best model per "
            "ticker' figure is also shown but is optimistically biased, because "
            "choosing the winner on the test window is selection on the test set."
        ),
    }
    if table.empty:
        return summary

    by_model: dict[str, dict] = {}
    for key, group in table.groupby("model_key"):
        auc = group["test_roc_auc"].dropna()
        by_model[str(key)] = {
            "display_name": group["model"].iloc[0],
            "n_tickers": int(len(group)),
            "median_test_roc_auc": float(auc.median()),
            "mean_test_roc_auc": float(auc.mean()),
            "share_above_0.5_auc": float((auc > 0.5).mean()),
            "median_test_accuracy": float(group["test_accuracy"].median()),
            "median_test_f1": float(group["test_f1"].median()),
            "share_beating_majority_baseline": float(
                (group["test_accuracy"] > group["majority_baseline_accuracy"]).mean()
            ),
        }

    strongest = max(by_model.items(), key=lambda item: item[1]["median_test_roc_auc"])
    best_per_ticker = table.loc[table.groupby("symbol")["test_roc_auc"].idxmax()]

    summary.update(
        {
            "by_model": by_model,
            "strongest_model_key": strongest[0],
            "strongest_model": strongest[1]["display_name"],
            "strongest_median_test_roc_auc": strongest[1]["median_test_roc_auc"],
            "median_majority_baseline": float(table["majority_baseline_accuracy"].median()),
            "median_momentum_baseline": float(table["momentum_baseline_accuracy"].median()),
            "biased_best_per_ticker_median_auc": float(
                best_per_ticker["test_roc_auc"].median()
            ),
            "rows": table.to_dict(orient="records"),
        }
    )
    return summary


def write_markdown(panel_report: dict, ticker_report: dict, path: Path) -> None:
    """Render a human-readable summary of both experiments."""
    best = panel_report["best_model"]
    baselines = panel_report["baselines"]
    lines = [
        "# FinMarket ML - experiment report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Dataset",
        "",
        f"- Observations (panel): **{panel_report['n_observations']:,}**",
        f"- Tickers: **{panel_report['n_tickers']}**",
        f"- Engineered features: **{panel_report['n_features']}**",
        f"- Date range: **{panel_report['date_range'][0]} to {panel_report['date_range'][1]}**",
        f"- Train window: {panel_report['split_periods']['train'][0]} to {panel_report['split_periods']['train'][1]} ({panel_report['split_sizes']['train']:,} rows)",
        f"- Validation window: {panel_report['split_periods']['validation'][0]} to {panel_report['split_periods']['validation'][1]} ({panel_report['split_sizes']['validation']:,} rows)",
        f"- Test window: {panel_report['split_periods']['test'][0]} to {panel_report['split_periods']['test'][1]} ({panel_report['split_sizes']['test']:,} rows)",
        f"- Share of up days in test window: {panel_report['class_balance_test']:.4f}",
        "",
        "## Pooled cross-sectional results (test window)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for model in panel_report["models"].values():
        test = model["test"]
        auc = f"{test['roc_auc']:.4f}" if test["roc_auc"] is not None else "n/a"
        lines.append(
            f"| {model['display_name']} | {test['accuracy']:.4f} | {test['precision']:.4f} | "
            f"{test['recall']:.4f} | {test['f1']:.4f} | {auc} |"
        )
    for name, metrics in baselines.items():
        lines.append(
            f"| Baseline: {name.replace('_', ' ')} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | n/a |"
        )

    lines += [
        "",
        f"Best model: **{best['display_name']}** "
        f"(ROC-AUC {best['test']['roc_auc']:.4f}, accuracy {best['test']['accuracy']:.4f}, "
        f"F1 {best['test']['f1']:.4f})",
        "",
        f"Top features ({panel_report['importance_method']} importance):",
        "",
    ]
    for i, row in enumerate(panel_report["top_features"][:10], start=1):
        lines.append(f"{i}. `{row['feature']}` - {row['importance']:.5f}")

    lines += [
        "",
        "## Per-ticker results",
        "",
        f"- Tickers modelled independently: **{ticker_report.get('n_tickers', 0)}**",
        "",
    ]
    if ticker_report.get("by_model"):
        lines += [
            "| Model | Median test ROC-AUC | Share of tickers above 0.50 | Median accuracy | Beats majority baseline |",
            "|---|---|---|---|---|",
        ]
        for stats in sorted(
            ticker_report["by_model"].values(),
            key=lambda s: s["median_test_roc_auc"],
            reverse=True,
        ):
            lines.append(
                f"| {stats['display_name']} | {stats['median_test_roc_auc']:.4f} | "
                f"{stats['share_above_0.5_auc'] * 100:.1f}% | "
                f"{stats['median_test_accuracy']:.4f} | "
                f"{stats['share_beating_majority_baseline'] * 100:.1f}% |"
            )
        lines += [
            "",
            f"Median majority-class baseline accuracy: "
            f"**{ticker_report.get('median_majority_baseline', float('nan')):.4f}**",
            "",
            f"For reference, picking the best of the five models *per ticker on the test "
            f"window* would give a median ROC-AUC of "
            f"{ticker_report.get('biased_best_per_ticker_median_auc', float('nan')):.4f}. "
            f"That number is selection on the test set and is not quoted as a result.",
            "",
        ]
    lines += [
        "## Signal study (illustrative, not a trading recommendation)",
        "",
    ]
    for key, value in panel_report["signal_study"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    lines += [
        "",
        "---",
        "",
        "For educational and research purposes only. This is not investment advice.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=None, help="Subset of tickers to use.")
    parser.add_argument("--skip-per-ticker", action="store_true")
    parser.add_argument(
        "--panel-models",
        nargs="*",
        default=None,
        help=f"Models for the pooled run (default: {' '.join(PANEL_MODELS)}).",
    )
    args = parser.parse_args()

    tickers = args.tickers or [t for t in SNAPSHOT_UNIVERSE]

    panel_report = run_panel(tickers, model_keys=args.panel_models)
    ticker_report = (
        {"experiment": "per_ticker", "n_tickers": 0}
        if args.skip_per_ticker
        else run_per_ticker(tickers)
    )

    combined = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot": snapshot_metadata(),
        "panel": panel_report,
        "per_ticker": ticker_report,
    }
    json_path = REPORT_DIR / "experiment_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, default=_json_safe)
    logger.info("Wrote %s", json_path)

    write_markdown(panel_report, ticker_report, REPORT_DIR / "experiment_report.md")

    best = panel_report["best_model"]
    print("\n" + "=" * 70)
    print(f"Panel observations : {panel_report['n_observations']:,}")
    print(f"Features           : {panel_report['n_features']}")
    print(f"Best model         : {best['display_name']}")
    print(f"Test ROC-AUC       : {best['test']['roc_auc']:.4f}")
    print(f"Test accuracy      : {best['test']['accuracy']:.4f}")
    print(f"Test F1            : {best['test']['f1']:.4f}")
    print(f"Majority baseline  : {panel_report['baselines']['majority_class']['accuracy']:.4f}")
    print(f"Momentum baseline  : {panel_report['baselines']['momentum_persistence']['accuracy']:.4f}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
