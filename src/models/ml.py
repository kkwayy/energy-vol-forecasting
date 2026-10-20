

"""Gradient boosting (LightGBM) and SHAP analysis."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.models.benchmarks import HAR
from src.walkforward import run_walkforward

import shap
from src.walkforward import yearly_splits

HAR_COLS = ["har_d", "har_w", "har_m"]


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

    def __init__(self, params: dict | None = None, seed: int = 0,anchor: str="har_m"):
        self.params = params or {}
        self.seed = seed
        self.anchor = anchor

    def fit(self, X_train, y_train, X_val, y_val) -> "LGBMVol":
        # target = log of future variance relative to the past month's level

        if self.anchor == "har":
            self.har = HAR().fit(X_train[HAR_COLS], y_train)  # train rows only

        y_tr = np.log(y_train) - self._log_base(X_train)
        y_va = np.log(y_val) - self._log_base(X_val)

        self.model = lgb.LGBMRegressor(**{**PARAMS, **self.params}, random_state=self.seed)
        self.model.fit(
            X_train, y_tr,
            eval_X=X_val, eval_y=y_va,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        self.resid_var = (y_va - self.model.predict(X_val)).var()
        return self

    def predict(self, X) -> pd.Series:
        log_pred = self.model.predict(X) + self._log_base(X)
        return pd.Series(np.exp(log_pred + self.resid_var / 2), index=X.index)


    def _log_base(self, X):
        if self.anchor == "har":
            return np.log(self.har.predict(X[HAR_COLS]))
        return np.log(X["har_m"])

def shap_summary(model: LGBMVol, X: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Mean |SHAP| per feature, overall and per period (e.g. calm, 2020, 2022)."""
    raise NotImplementedError

def lgbm_seed_average(X, y, first_test_year, val_years, horizon, n_seeds=5, anchor="har_m"):
    preds = [run_walkforward(lambda s=s: LGBMVol(seed=s, anchor=anchor), X, y,
                             first_test_year, val_years, horizon, use_val=True)
             for s in range(n_seeds)]
    return pd.concat(preds, axis=1).mean(axis=1).rename(f"lgbm_{anchor}")

def walkforward_shap(X, y, first_test_year, val_years, horizon, seed=0, anchor="har"):
    """SHAP values for every test day: refit one model per test year (as in the walk-forward)
    and explain its forecasts on that year's test rows."""
    frames = []
    for train, val, test in yearly_splits(X.index, first_test_year, val_years, horizon):
        m = LGBMVol(seed=seed, anchor=anchor).fit(X.loc[train], y.loc[train], X.loc[val], y.loc[val])
        sv = shap.TreeExplainer(m.model).shap_values(X.loc[test])
        frames.append(pd.DataFrame(sv, index=test, columns=X.columns))
    return pd.concat(frames)

if __name__ == "__main__":
    from src import data, features
    from src.evaluate import qlike, diebold_mariano

    panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
    X, y = features.build_dataset(panel, "WTI", 5, extra=True)

    har = run_walkforward(HAR, X[HAR_COLS], y, 2012, 1, 5)
    actual = y.loc[har.index]
    loss_har = qlike(actual, har)

    results = {"HAR": loss_har}
    for anchor in ["har_m", "har"]:
        avg = lgbm_seed_average(X, y, 2012, 1, 5, anchor=anchor)
        loss = qlike(actual, avg.loc[har.index])
        stat, p = diebold_mariano(loss, loss_har, 5)
        print(f"{anchor:6s} QLIKE {loss.mean():.4f} | DM vs HAR {stat:.2f}, p {p:.3f}")
        results[f"LGBM_{anchor}"] = loss

    importance = {}
    for h in [1, 22]:
        X_h, y_h = features.build_dataset(panel, "WTI", h, extra=True)
        sv = walkforward_shap(X_h, y_h, 2012, 1, h)
        importance[f"h={h}"] = sv.abs().mean()

    imp = pd.DataFrame(importance).sort_values("h=1", ascending=False)
    print(imp.round(4))

    print(pd.DataFrame(results).groupby(loss_har.index.year).mean().round(3))