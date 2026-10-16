"""Forecast evaluation: losses and significance tests."""

import pandas as pd


def qlike(realised_var: pd.Series, forecast_var: pd.Series) -> pd.Series:
    """Per-observation QLIKE loss (defined on variance). Robust to noise in the volatility proxy."""
    raise NotImplementedError


def mse(realised: pd.Series, forecast: pd.Series) -> pd.Series:
    raise NotImplementedError


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> tuple[float, float]:
    """DM test on the loss differential. Returns (statistic, p-value).

    TODO: use a HAC / Newey–West variance with lag ~ horizon - 1, since overlapping
    multi-day targets make the loss differential autocorrelated.
    """
    raise NotImplementedError


def results_table(forecasts: dict[str, pd.Series], realised: pd.Series, horizon: int) -> pd.DataFrame:
    """One row per model: mean QLIKE, mean MSE, DM p-value vs HAR."""
    raise NotImplementedError
