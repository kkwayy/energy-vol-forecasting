"""Expanding-window walk-forward splits. Every model is scored through this, so all comparisons are fair."""

from typing import Iterator
import pandas as pd


def yearly_splits(index: pd.DatetimeIndex, first_test_year: int, val_years: int
                  ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield (train_idx, val_idx, test_idx) for each test year.

    train: everything before the validation period
    val:   the `val_years` before the test year (tuning only)
    test:  one calendar year

    TODO: leave a gap of `horizon` days between sets so targets don't overlap the next set.
    """
    raise NotImplementedError


def run_walkforward(model_factory, X: pd.DataFrame, y: pd.Series, **split_kwargs) -> pd.Series:
    """Fit a fresh model per split, predict on the test year, and stitch out-of-sample forecasts together."""
    raise NotImplementedError
