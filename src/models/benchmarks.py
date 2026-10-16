"""Classical benchmarks: HAR and GARCH(1,1)."""

import pandas as pd


class HAR:
    """Linear regression of future vol on HAR terms (statsmodels OLS).

    TODO: consider fitting on log vol to keep forecasts positive; if so, correct for the
    bias when converting back.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HAR":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class GARCH:
    """GARCH(1,1) on daily returns (arch package), forecasting average variance over the horizon.

    Note: works on returns, not on the feature matrix — needs its own interface into walk-forward.
    TODO: scale returns (e.g. x100) before fitting for numerical stability.
    """

    def __init__(self, horizon: int):
        self.horizon = horizon

    def fit(self, returns: pd.Series) -> "GARCH":
        raise NotImplementedError

    def forecast(self, returns: pd.Series) -> pd.Series:
        raise NotImplementedError
