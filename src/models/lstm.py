"""Small LSTM forecaster (PyTorch). Optional — first thing to cut if short on time."""

import pandas as pd


def make_sequences(X: pd.DataFrame, y: pd.Series, lookback: int):
    """Turn the feature matrix into (samples, lookback, features) windows ending at each date t."""
    raise NotImplementedError


class LSTMVol:
    """One or two LSTM layers -> linear head -> vol forecast.

    TODO:
    - standardise features using training-set statistics only
    - early stopping on the validation period
    - run several seeds and report mean and spread
    """

    def __init__(self, lookback: int = 22, hidden: int = 32, seed: int = 0):
        self.lookback, self.hidden, self.seed = lookback, hidden, seed

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LSTMVol":
        raise NotImplementedError

    def predict(self, X) -> pd.Series:
        raise NotImplementedError
