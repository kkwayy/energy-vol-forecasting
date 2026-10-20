"""Forecast evaluation: losses and significance tests."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
def qlike(realised_var: pd.Series, forecast_var: pd.Series) -> pd.Series:
    """Per-observation QLIKE loss (defined on variance). Robust to noise in the volatility proxy."""
    ratio = realised_var / forecast_var

    return ratio - np.log(ratio) -1


def mse(realised: pd.Series, forecast: pd.Series) -> pd.Series:
    raise NotImplementedError


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> tuple[float, float]:
    """DM test on the loss differential. Returns (statistic, p-value).

    TODO: use a HAC / Newey–West variance with lag ~ horizon - 1, since overlapping
    multi-day targets make the loss differential autocorrelated.

    """

    d = (loss_a - loss_b).dropna()
    result = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon - 1, 1)})
    return result.tvalues.iloc[0], result.pvalues.iloc[0]



def results_table(forecasts: dict[str, pd.Series], realised: pd.Series, horizon: int) -> pd.DataFrame:
    """One row per model: mean QLIKE, mean MSE, DM p-value vs HAR."""
    raise NotImplementedError



if __name__ == "__main__":
    from src import data, features, config
    from src.models import benchmarks
    from src.models.ml import lgbm_seed_average          # NEW
    from src.walkforward import run_walkforward

    panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
    rows = []
    for asset in config.ENERGY:
        for h in config.HORIZONS:
            X, y = features.build_dataset(panel, asset, h)
            har = run_walkforward(benchmarks.HAR, X, y, config.FIRST_TEST_YEAR, config.VALIDATION_YEARS, h)
            actual = y.loc[har.index]
            naive = X.loc[har.index, "har_m"]

            loss_har = qlike(actual, har)
            loss_naive = qlike(actual, naive)
            stat, p = diebold_mariano(loss_har, loss_naive, h)

            garch = benchmarks.garch_walkforward(panel[f"{asset}_ret"], panel[f"{asset}_gk"], y.index,
                                                 config.FIRST_TEST_YEAR, config.VALIDATION_YEARS, h)
            loss_garch = qlike(actual, garch.loc[har.index])
            stat_g, p_g = diebold_mariano(loss_garch, loss_har, h)

            # NEW: HAR-anchored LightGBM, 5-seed average
            X_full, _ = features.build_dataset(panel, asset, h, extra=True)
            lgbm = lgbm_seed_average(X_full, y, config.FIRST_TEST_YEAR, config.VALIDATION_YEARS, h, anchor="har")
            loss_lgbm = qlike(actual, lgbm.loc[har.index])
            stat_l, p_l = diebold_mariano(loss_lgbm, loss_har, h)

            # HAR-IV: HAR + log OVX implied variance (linear)
            harx = run_walkforward(benchmarks.HAR, X_full[["har_d", "har_w", "har_m", "ovx_var"]], y,
                                   config.FIRST_TEST_YEAR, config.VALIDATION_YEARS, h)
            loss_harx = qlike(actual, harx.loc[har.index])
            stat_x, p_x = diebold_mariano(loss_harx, loss_har, h)  # HAR-IV vs HAR
            stat_lx, p_lx = diebold_mariano(loss_lgbm, loss_harx, h)  # LightGBM vs HAR-IV

            # save all forecasts for later analysis
            fc = pd.DataFrame({"actual": actual, "HAR": har, "naive": naive,
                               "GARCH": garch.loc[har.index], "LGBM": lgbm.loc[har.index],
                               "HARX": harx.loc[har.index]})
            out_dir = data.PROCESSED_DIR.parents[1] / "results" / "forecasts"
            out_dir.mkdir(parents=True, exist_ok=True)
            fc.to_csv(out_dir / f"{asset}_h{h}.csv")
            print(f"done: {asset} h={h}")

            rows.append({"asset": asset, "h":    h,
                         "QLIKE_HAR": loss_har.mean(), "QLIKE_naive": loss_naive.mean(),
                         "DM_HAR_vs_naive": stat, "p_naive": p,
                         "QLIKE_GARCH": loss_garch.mean(),
                         "DM_GARCH_vs_HAR": stat_g, "p_GARCH": p_g,
                         "QLIKE_LGBM": loss_lgbm.mean(),
                         "DM_LGBM_vs_HAR": stat_l, "p_LGBM": p_l,
                         "QLIKE_HARX": loss_harx.mean(),
                         "DM_HARX_vs_HAR": stat_x, "p_HARX": p_x,
                         "DM_LGBM_vs_HARX": stat_lx, "p_LGBM_vs_HARX": p_lx})

    table = pd.DataFrame(rows)

    print(table[["asset", "h", "QLIKE_HAR", "QLIKE_HARX", "QLIKE_LGBM",
                 "DM_HARX_vs_HAR", "p_HARX", "DM_LGBM_vs_HARX", "p_LGBM_vs_HARX"]].round(3).to_string(index=False))
    table.to_csv(data.PROCESSED_DIR.parents[1] / "results" / "benchmarks_and_lgbm.csv", index=False)