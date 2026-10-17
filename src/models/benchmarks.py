"""Classical benchmarks: HAR and GARCH(1,1)."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
class HAR:
    """Linear regression of future vol on HAR terms (statsmodels OLS).

    TODO: consider fitting on log vol to keep forecasts positive; if so, correct for the
    bias when converting back.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HAR":

        log_X = np.log(X)
        log_X = sm.add_constant(log_X, has_constant="add")

        log_y = np.log(y)


        result = sm.OLS(log_y,log_X).fit()

        self.result = result
        self.resid_var = result.resid.var()
        return self




    def predict(self, X: pd.DataFrame) -> pd.Series:
        log_X = np.log(X)
        log_X = sm.add_constant(log_X, has_constant="add")

        log_pred = self.result.predict(log_X)

        return np.exp(log_pred + self.resid_var / 2)

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
if __name__ == "__main__":
    from src import data, features
    from src.walkforward import yearly_splits

    panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
    X, y = features.build_dataset(panel, "WTI", 5)

    train, val, test = next(yearly_splits(X.index, 2012, 1, 5))   # first split only
    fit_idx = train.append(val)                                   # HAR has nothing to tune
    model = HAR().fit(X.loc[fit_idx], y.loc[fit_idx])
    print(model.result.params)

    pred = model.predict(X.loc[test])
    print(pd.DataFrame({"pred": pred, "actual": y.loc[test]}).head(10))