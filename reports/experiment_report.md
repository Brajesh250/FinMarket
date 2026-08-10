# FinMarket ML - experiment report

Generated: 2026-08-10 22:06 UTC

## Dataset

- Observations (panel): **46,436**
- Tickers: **39**
- Engineered features: **38**
- Date range: **2013-05-07 to 2018-02-06**
- Train window: 2013-05-07 to 2016-08-31 (32,396 rows)
- Validation window: 2016-09-01 to 2017-05-19 (7,020 rows)
- Test window: 2017-05-22 to 2018-02-06 (7,020 rows)
- Share of up days in test window: 0.5412

## Pooled cross-sectional results (test window)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.4774 | 0.5580 | 0.1645 | 0.2541 | 0.5022 |
| Random Forest | 0.4781 | 0.5535 | 0.1837 | 0.2759 | 0.5155 |
| Gradient Boosting | 0.4990 | 0.5706 | 0.2998 | 0.3931 | 0.5230 |
| XGBoost | 0.4969 | 0.5627 | 0.3153 | 0.4042 | 0.5222 |
| Baseline: majority class | 0.5412 | 0.5412 | 1.0000 | 0.7023 | n/a |
| Baseline: momentum persistence | 0.4981 | 0.5362 | 0.5375 | 0.5369 | n/a |

Best model: **Gradient Boosting** (ROC-AUC 0.5230, accuracy 0.4990, F1 0.3931)

Top features (native importance):

1. `market_volatility_20d` - 0.30428
2. `market_return` - 0.24828
3. `market_return_5d` - 0.22474
4. `month` - 0.05652
5. `day_of_week` - 0.02280
6. `intraday_return` - 0.01716
7. `overnight_return` - 0.01664
8. `volume_trend_5d` - 0.01310
9. `relative_volume` - 0.00915
10. `return_10d` - 0.00850

## Per-ticker results

- Tickers modelled independently: **39**

| Model | Median test ROC-AUC | Share of tickers above 0.50 | Median accuracy | Beats majority baseline |
|---|---|---|---|---|
| Random Forest | 0.5228 | 61.5% | 0.4889 | 30.8% |
| Gradient Boosting | 0.5206 | 61.5% | 0.5056 | 33.3% |
| XGBoost | 0.5125 | 61.5% | 0.5056 | 33.3% |
| Support Vector Machine (RBF) | 0.5000 | 48.7% | 0.5290 | 35.9% |
| Logistic Regression | 0.4967 | 46.2% | 0.4778 | 17.9% |

Median majority-class baseline accuracy: **0.5333**

For reference, picking the best of the five models *per ticker on the test window* would give a median ROC-AUC of 0.5486. That number is selection on the test set and is not quoted as a result.

## Signal study (illustrative, not a trading recommendation)

- observations: 180
- days in market pct: 9.44
- strategy total return pct: 4.15
- buy hold total return pct: 21.09
- strategy annualised return pct: 5.86
- buy hold annualised return pct: 30.72
- strategy annualised volatility pct: 3.02
- buy hold annualised volatility pct: 10.4
- strategy max drawdown pct: -1.01
- buy hold max drawdown pct: -8.24
- n position changes: 19
- total cost pct: 0.95
- transaction cost bps: 5.0

---

For educational and research purposes only. This is not investment advice.
