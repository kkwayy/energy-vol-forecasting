"""Expanding-window walk-forward splits. Every model is scored through this, so all comparisons are fair."""

from typing import Iterator
import pandas as pd
from src import data, features, config
from src.models import benchmarks


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


def run_walkforward(model_factory, X: pd.DataFrame, y: pd.Series,
                    first_test_year: int, val_years: int, horizon: int) -> pd.Series:
    """Fit a fresh model per test year, predict that year, and join all out-of-sample forecasts."""
    predictions = []

    for train, val, test in yearly_splits(X.index, first_test_year, val_years, horizon):
        fit_idx = train.append(val)  # HAR has nothing to tune
        model = model_factory().fit(X.loc[fit_idx], y.loc[fit_idx])
        predictions.append(model.predict(X.loc[test]))

    return pd.concat(predictions)



if __name__ == "__main__":
    from src import data, features, config
    from src.models import benchmarks

    panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
    X, y = features.build_dataset(panel, "WTI", 5)
    pred = run_walkforward(benchmarks.HAR, X, y, config.FIRST_TEST_YEAR, config.VALIDATION_YEARS, 5)
    print(pred.index.min(), pred.index.max(), len(pred), pred.index.is_unique)