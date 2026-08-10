# Resume bullets — FinMarket ML

Every number below is produced by `scripts/run_experiment.py` and appears in
`reports/experiment_results.json`. Nothing is estimated.

## Project heading

```
FinMarket ML — Stock Movement & Signal Analytics
Python, Scikit-learn, XGBoost, yfinance, Streamlit | GitHub | Live Demo
```

---

## Option 1 — the data and feature pipeline

> Built an end-to-end ML pipeline over **46,436 ticker-sessions** of daily OHLCV
> spanning **five years and 39 S&P 500 equities**, engineering **38** momentum,
> volatility, moving-average and technical features to predict next-day market
> direction.

## Option 2 — the modelling and benchmarking

> Benchmarked Logistic Regression, Random Forest, Gradient Boosting, XGBoost and
> SVM under leakage-free chronological validation, reaching **0.523 test ROC-AUC**
> against a 0.50 no-skill line while reporting honestly that no model beat the
> **0.541** majority-class accuracy baseline.

## Option 3 — the application

> Developed a Streamlit market analytics platform across **39 selectable
> equities** integrating RSI, MACD, Bollinger Bands, rolling volatility and model
> explainability, with permutation importance showing broad-market context
> features carry **78%** of total predictive importance.

---

## The two strongest, ATS-optimised

**1.**
> Engineered a leakage-free ML pipeline over **46,436 ticker-sessions** of
> five-year OHLCV data across **39 S&P 500 equities**, building **38** momentum,
> volatility and technical-indicator features and benchmarking **5 classifiers**
> under chronological validation to **0.523 test ROC-AUC** versus a 0.50 no-skill
> baseline.

**2.**
> Developed a Streamlit market analytics platform with RSI, MACD, Bollinger Bands
> and SHAP/permutation explainability across **39 equities**, quantifying that
> broad-market volatility and return features drive **78%** of model importance
> while single-stock technical indicators rank near the bottom.

---

## Talking points for interviews

**"Your model barely beats random. Why is that on your resume?"**
Because next-day equity direction is close to unpredictable, and knowing that is
the finding. The pipeline is correct — 79 tests, including ones that recompute
every indicator on a truncated history to prove no future value leaks in — so the
0.523 ROC-AUC is a real measurement of how little signal there is, not a broken
model. I also report that accuracy does *not* beat the majority-class baseline of
0.541, which most projects in this space quietly omit. Anyone claiming 85%
accuracy on daily direction has a leak; I can show exactly where mine would be if
it had one.

**"What does the 78% figure mean?"**
The three broad-market context features — 20-day market volatility, market return,
and 5-day market return — absorb 78% of total feature importance in the best
pooled model. Whatever weak signal exists is a market-state effect: it is about
what the whole market is doing, not about a particular company's RSI. That is a
substantive finding, and it is the opposite of what most retail technical
analysis assumes.

**"How did you prevent leakage?"**
Three enforced rules plus a tripwire. Features are causal by construction and
tested by recomputing on truncated data. Splits are chronological, cut on
calendar dates for the pooled panel so one session cannot straddle partitions,
and passing a shuffled frame raises. Scaling lives inside the sklearn Pipeline so
fold statistics never see held-out rows. And a test fails the build if any single
feature correlates above 0.30 with the target, since that is the signature of a
leak.

**"Why did you report per-model medians instead of the best model per ticker?"**
Because picking the winner on the test window is selection on the test set. Doing
that would have given a median ROC-AUC of 0.5486 instead of 0.5228 — a
half-a-point of pure selection bias. The inflated number is in the JSON report
labelled as biased, and it is not what I quote.
