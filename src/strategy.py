"""Economic value: does a better forecast make a better volatility-targeted strategy?"""

import pandas as pd


def vol_target_positions(forecast_vol: pd.Series, target_vol: float, max_leverage: float = 3.0) -> pd.Series:
    """Position on day t+1 = target vol / forecast vol made at t, capped at max_leverage."""
    raise NotImplementedError


def strategy_returns(positions: pd.Series, asset_returns: pd.Series, cost_bps: float) -> pd.Series:
    """Daily returns net of costs proportional to turnover (|change in position|).

    Make sure positions are lagged correctly relative to returns.
    """
    raise NotImplementedError


def performance(returns: pd.Series) -> dict:
    """Annualised return, volatility, Sharpe, max drawdown, turnover."""
    raise NotImplementedError
