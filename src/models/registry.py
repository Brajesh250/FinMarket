"""Model zoo.

Each entry is a scikit-learn ``Pipeline`` so that scaling is fitted inside the
training fold only — another small but important leakage guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.settings import RANDOM_STATE
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """A named, constructible model together with a short description."""

    key: str
    display_name: str
    description: str
    factory: Callable[[], Pipeline]
    supports_coefficients: bool = False
    supports_feature_importance: bool = False


def _logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _random_forest() -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_leaf=25,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def _gradient_boosting() -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=3,
                    subsample=0.8,
                    min_samples_leaf=25,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def _support_vector_machine() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _xgboost() -> Pipeline:
    from xgboost import XGBClassifier  # imported lazily; optional dependency

    return Pipeline(
        [
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    min_child_weight=10,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        ]
    )


_BASE_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        description=(
            "Linear baseline. Coefficients show the direction and strength of each "
            "feature's contribution to the probability of an up day."
        ),
        factory=_logistic_regression,
        supports_coefficients=True,
    ),
    ModelSpec(
        key="random_forest",
        display_name="Random Forest",
        description=(
            "Bagged decision trees. Depth and leaf size are constrained because "
            "daily direction has a very low signal-to-noise ratio."
        ),
        factory=_random_forest,
        supports_feature_importance=True,
    ),
    ModelSpec(
        key="gradient_boosting",
        display_name="Gradient Boosting",
        description=(
            "Sequentially boosted shallow trees with a low learning rate, which "
            "tends to be the strongest classical learner on tabular market data."
        ),
        factory=_gradient_boosting,
        supports_feature_importance=True,
    ),
    ModelSpec(
        key="svm_rbf",
        display_name="Support Vector Machine (RBF)",
        description="Kernel margin classifier on standardised features.",
        factory=_support_vector_machine,
    ),
)

_XGBOOST_SPEC = ModelSpec(
    key="xgboost",
    display_name="XGBoost",
    description="Regularised gradient boosting; included when the package is installed.",
    factory=_xgboost,
    supports_feature_importance=True,
)


def xgboost_available() -> bool:
    """Return True when the optional ``xgboost`` package can be imported."""
    try:
        import xgboost  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem disables the model
        return False
    return True


def available_models() -> dict[str, ModelSpec]:
    """Return every model that can actually be constructed in this environment."""
    specs = {spec.key: spec for spec in _BASE_SPECS}
    if xgboost_available():
        specs[_XGBOOST_SPEC.key] = _XGBOOST_SPEC
    else:
        logger.info("xgboost not importable; continuing without it.")
    return specs


def get_model_spec(key: str) -> ModelSpec:
    """Look up a single model spec by key.

    Raises:
        KeyError: If the key is unknown in this environment.
    """
    specs = available_models()
    if key not in specs:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(specs)}")
    return specs[key]


def build_model(key: str) -> Pipeline:
    """Construct a fresh, unfitted pipeline for ``key``."""
    return get_model_spec(key).factory()
