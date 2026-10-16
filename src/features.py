"""Targets and features. Every feature on date t must use only data up to and including t."""

import pandas as pd


def make_target(vol: pd.Series, horizon: int) -> pd.Series:
    """Average volatility over days t+1 .. t+horizon (the thing we forecast at t)."""
    raise NotImplementedError


def har_features(vol: pd.Series) -> pd.DataFrame:
    """HAR terms: yesterday's vol, past 5-day mean, past 22-day mean."""
    raise NotImplementedError


def extra_features(panel: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Features for the ML models.

    TODO:
    - recent returns, split into negative and positive parts (leverage effect)
    - OVX and VIX levels / changes
    - volatility of the other energy assets
    - day of week, EIA-report-day flag (weekly petroleum report, usually Wednesday)
    """
    raise NotImplementedError


def build_dataset(panel: pd.DataFrame, asset: str, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Combine features and target; drop rows with NaNs. Returns (X, y)."""
    raise NotImplementedError


def check_no_lookahead(X: pd.DataFrame, y: pd.Series) -> None:
    """Sanity check: e.g. shift a feature forward and confirm results get suspiciously good,
    or assert feature timestamps never exceed the forecast date."""
    raise NotImplementedError
