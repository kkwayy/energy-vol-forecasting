"""Gradient boosting (LightGBM) and SHAP analysis."""

import pandas as pd


class LGBMVol:
    """LightGBM regressor for future volatility.

    TODO:
    - choose the target transform (log vol is common) and loss
    - tune a small grid on the validation period only (depth, learning rate, n_estimators)
    - set a fixed seed per run
    """

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params = params or {}
        self.seed = seed

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "LGBMVol":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


def shap_summary(model: LGBMVol, X: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Mean |SHAP| per feature, overall and per period (e.g. calm, 2020, 2022)."""
    raise NotImplementedError
