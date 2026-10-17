"""Targets and features. Every feature on date t must use only data up to and including t."""

import pandas as pd
from src import data

def make_target(vol: pd.Series, horizon: int) -> pd.Series:
    """Average variance over days t+1 .. t+horizon (the thing we forecast at t)."""

    future = vol.rolling(horizon).mean()
    future= future.shift(-horizon)

    return future.rename(f"target_{horizon}")

def har_features(vol: pd.Series) -> pd.DataFrame:
    """HAR terms: today's variance, past 5-day mean, past 22-day mean."""

    return pd.DataFrame({
        "har_d": vol,
        "har_w": vol.rolling(5).mean(),
        "har_m": vol.rolling(22).mean(),
    })



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

    var = panel[f"{asset}_gk"]
    X = har_features(var)
    y = make_target(var, horizon)
    data = X.join(y).dropna()
    X = data[X.columns]
    y = data[y.name]

    return X,y

    raise NotImplementedError


def check_no_lookahead(X: pd.DataFrame, y: pd.Series) -> None:
    """Sanity check: e.g. shift a feature forward and confirm results get suspiciously good,
    or assert feature timestamps never exceed the forecast date."""
    raise NotImplementedError


if __name__ == "__main__":
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    print(make_target(s, 2))      # row 0 → 2.5, row 1 → 3.5, last two NaN
    print(har_features(s))        # har_w first valid at row 4 → 3.0
    panel = pd.read_csv(data.PROCESSED_DIR/"panel.csv", index_col = 0, parse_dates = True)
    X, y = build_dataset(panel, "WTI", 5)
    print(X.shape, y.shape)  # same number of rows
    print(X.index.min(), X.index.max())