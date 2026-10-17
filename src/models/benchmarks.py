"""Classical benchmarks: HAR and GARCH(1,1)."""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from arch import arch_model
from src.walkforward import yearly_splits
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

def garch_walkforward(returns: pd.Series, gk_var: pd.Series, target_index: pd.DatetimeIndex,
                          first_test_year: int, val_years: int, horizon: int,
                          calibrate: bool = True) -> pd.Series:
    """GARCH(1,1) forecasts of average daily return variance over the next `horizon` days,
    one test year at a time. Parameters are fitted only on data before each test year."""
    r = returns.dropna() * 100

    am = arch_model(r, mean="Constant",vol="GARCH",p=1,q=1)
    predictions =[]
    for _, _, test in yearly_splits(target_index,first_test_year,val_years,horizon):
        res = am.fit(last_obs=test[0],disp="off")
        fc = res.forecast(horizon=horizon, start=test[0], reindex=False)
        var = fc.variance.mean(axis=1) / 10_000          # 1. forecast in decimal units

        if calibrate:                                     # 2. then rescale it
            past_gk = gk_var.loc[:test[0]].iloc[:-1]
            past_r2 = returns.loc[:test[0]].iloc[:-1] ** 2
            scale = past_gk.mean() / past_r2.mean()
            var = var * scale


        predictions.append(var.loc[test])


    return pd.concat(predictions).rename("garch")



if __name__ == "__main__":
    from src import data, features, config
    from src.walkforward import run_walkforward


    panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
    X, y = features.build_dataset(panel, "WTI", 5)

    train, val, test = next(yearly_splits(X.index, 2012, 1, 5))   # first split only
    fit_idx = train.append(val)                                   # HAR has nothing to tune
    model = HAR().fit(X.loc[fit_idx], y.loc[fit_idx])
    print(model.result.params)
    pred = model.predict(X.loc[test])
    print(pd.DataFrame({"pred": pred, "actual": y.loc[test]}).head(10))
    garch = garch_walkforward(panel["WTI_ret"], panel["WTI_gk"], y.index,
                          config.FIRST_TEST_YEAR, config.VALIDATION_YEARS, 5)
    print(garch.index.min(), garch.index.max(), len(garch), garch.index.is_unique)
    print("mean GARCH forecast:", garch.mean(), "| mean target:", y.loc[garch.index].mean())
    print("median GARCH / target:", (garch / y.loc[garch.index]).median())
    har_pred = run_walkforward(HAR, X, y, config.FIRST_TEST_YEAR, config.VALIDATION_YEARS, 5)
    print("median HAR / target:  ", (har_pred / y.loc[har_pred.index]).median())
