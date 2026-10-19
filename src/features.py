"""Targets and features. Every feature on date t must use only data up to and including t."""

import pandas as pd
from src import data, config
import numpy as np

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

    ret = panel[f"{asset}_ret"]
    ret_neg = ret.clip(upper=0)
    ret_neg_5= ret_neg.rolling(5).sum()

    ovx_var = (panel["OVX"] / 100 ) **2 / 252
    ovx_chg_5 = ovx_var.diff(5)
    vix_var = (panel["VIX"] / 100) ** 2 / 252


    cols = {
        "ret": ret,
        "ret_neg": ret_neg,
        "ret_neg_5": ret_neg_5,
        "ovx_var": ovx_var,
        "ovx_chg_5": ovx_chg_5,
        "vix_var": vix_var,
    }

    for other in config.ENERGY:
        if other != asset:
            cols[f"{other}_gk_w"] = panel[f"{other}_gk"].rolling(5).mean()

    cols["spx_ret"] = np.log(panel["SPX"]).diff()

    cols["dow"] = pd.Series(panel.index.dayofweek, index=panel.index)

    return pd.DataFrame(cols)





def build_dataset(panel: pd.DataFrame, asset: str, horizon: int,extra:bool= False) -> tuple[pd.DataFrame, pd.Series]:
    """Combine features and target; drop rows with NaNs. Returns (X, y)."""

    var = panel[f"{asset}_gk"]
    X = har_features(var)
    if extra:
        X= X.join(extra_features(panel, asset))
    y = make_target(var, horizon)

    data = X.join(y).dropna()
    X = data[X.columns]
    y = data[y.name]

    return X,y




def check_no_lookahead(X: pd.DataFrame, y: pd.Series) -> None:
    """Sanity check: e.g. shift a feature forward and confirm results get suspiciously good,
    or assert feature timestamps never exceed the forecast date."""
    raise NotImplementedError


if __name__ == "__main__":
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    print(make_target(s, 2))      # row 0 → 2.5, row 1 → 3.5, last two NaN
    print(har_features(s))        # har_w first valid at row 4 → 3.0
    panel = pd.read_csv(data.PROCESSED_DIR/"panel.csv", index_col = 0, parse_dates = True)
    X, y = build_dataset(panel, "WTI", 5, extra=True)
    print(X.shape)
    print(X.describe().T[["mean", "min", "max"]])