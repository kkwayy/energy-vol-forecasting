"""Expanding-window walk-forward splits. Every model is scored through this, so all comparisons are fair."""

from typing import Iterator
import pandas as pd



def yearly_splits(index: pd.DatetimeIndex, first_test_year: int, val_years: int,horizon:int
                  ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield (train_idx, val_idx, test_idx) for each test year.

    train: everything before the validation period
    val:   the `val_years` before the test year (tuning only)
    test:  one calendar year

    TODO: leave a gap of `horizon` days between sets so targets don't overlap the next set.
    """

    last_year = index.year.max()

    for test_year in range(first_test_year, last_year + 1):
        val_start_year = test_year - val_years

        train = index[index.year < val_start_year]
        val = index[(index.year >= val_start_year) & (index.year < test_year)]
        test = index[index.year == test_year]
        train = train[:-horizon]
        val = val[:-horizon]

        yield train, val, test

def run_walkforward(model_factory, X, y, first_test_year, val_years, horizon,
                    use_val: bool = False) -> pd.Series:
    predictions = []
    for train, val, test in yearly_splits(X.index, first_test_year, val_years, horizon):
        if use_val:      # models with settings: fit on train, use val for early stopping
            model = model_factory().fit(X.loc[train], y.loc[train], X.loc[val], y.loc[val])
        else:            # HAR: nothing to tune, so fit on train + val
            fit_idx = train.append(val)
            model = model_factory().fit(X.loc[fit_idx], y.loc[fit_idx])
        predictions.append(model.predict(X.loc[test]))
    return pd.concat(predictions)


if __name__ == "__main__":
    from src import data, features, config
    from src.models import benchmarks
    from src.models.ml import LGBMVol
    from src.evaluate import qlike, diebold_mariano

    panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
    X, y = features.build_dataset(panel, "WTI", 5, extra=True)
    har = run_walkforward(benchmarks.HAR, X[["har_d", "har_w", "har_m"]], y, 2012, 1, 5)
    lgbm = run_walkforward(LGBMVol, X, y, 2012, 1, 5, use_val=True)

    actual = y.loc[har.index]
    loss_har, loss_lgbm = qlike(actual, har), qlike(actual, lgbm.loc[har.index])
    print("HAR :", loss_har.mean())
    print("LGBM:", loss_lgbm.mean())
    print("DM (LGBM vs HAR):", diebold_mariano(loss_lgbm, loss_har, 5))
    by_year = pd.DataFrame({"HAR": loss_har, "LGBM": loss_lgbm}).groupby(loss_har.index.year).mean()
    print(by_year.round(3))

    seed_preds = {}
    for s in range(5):
        seed_preds[s] = run_walkforward(lambda: LGBMVol(seed=s), X, y, 2012, 1, 5, use_val=True)

    # 1. how much does each seed vary?
    for s, pred in seed_preds.items():
        print(f"seed {s}: QLIKE {qlike(actual, pred.loc[har.index]).mean():.4f}")

    # 2. the average of the 5 seeds' forecasts
    lgbm_avg = pd.concat(seed_preds, axis=1).mean(axis=1)
    loss_avg = qlike(actual, lgbm_avg.loc[har.index])
    print("LGBM 5-seed average:", loss_avg.mean())
    print("DM (avg vs HAR):", diebold_mariano(loss_avg, loss_har, 5))
