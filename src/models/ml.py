

"""Gradient boosting (LightGBM) and SHAP analysis."""
import numpy as np
import pandas as pd
import lightgbm as lgb

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_estimators": 2000,
    "verbose": -1,
}

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

    def fit(self, X_train, y_train, X_val, y_val) -> "LGBMVol":
        # target = log of future variance relative to the past month's level
        y_tr = np.log(y_train) - np.log(X_train["har_m"])
        y_va = np.log(y_val) - np.log(X_val["har_m"])

        self.model = lgb.LGBMRegressor(**PARAMS, random_state=self.seed)
        self.model.fit(
            X_train, y_tr,
            eval_X=X_val, eval_y=y_va,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        self.resid_var = (y_va - self.model.predict(X_val)).var()
        return self

    def predict(self, X) -> pd.Series:
        log_pred = self.model.predict(X) + np.log(X["har_m"])
        return pd.Series(np.exp(log_pred + self.resid_var / 2), index=X.index)

def shap_summary(model: LGBMVol, X: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Mean |SHAP| per feature, overall and per period (e.g. calm, 2020, 2022)."""
    raise NotImplementedError


if __name__ == "__main__":
    from src import data, features
    from src.walkforward import yearly_splits

    panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
    X, y = features.build_dataset(panel, "WTI", 5, extra=True)

    train, val, test = next(yearly_splits(X.index, 2012, 1, 5))
    model = LGBMVol(seed=0).fit(X.loc[train], y.loc[train], X.loc[val], y.loc[val])

    print("trees used:", model.model.best_iteration_)
    print("resid var:", model.resid_var)
    pred = model.predict(X.loc[test])
    print(pd.DataFrame({"pred": pred, "actual": y.loc[test]}).head(10))