"""Volatility-targeted strategy: does a better variance forecast give better risk-adjusted returns?"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def vol_target_positions(forecast_var: pd.Series, target_vol: float = 0.15,
                         max_leverage: float = 2.0, rebalance_every: int = 1) -> pd.Series:
    """Position decided at the close of day t from that day's forecast (daily variance)."""
    ann_vol = np.sqrt(forecast_var * TRADING_DAYS)
    pos = (target_vol / ann_vol).clip(upper=max_leverage)
    if rebalance_every > 1:                       # only change position every n days
        keep = np.arange(len(pos)) % rebalance_every == 0
        pos = pos.where(keep).ffill()
    return pos


def strategy_returns(pos: pd.Series, log_ret: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    """Position set at close t earns day t+1's return; costs charged on position changes."""
    simple_ret = np.exp(log_ret) - 1
    held = pos.shift(1)                           # yesterday's decision earns today's return
    gross = held * simple_ret.reindex(held.index)
    cost = held.diff().abs() * cost_bps / 10_000
    return (gross - cost).dropna()


def performance(r: pd.Series, pos: pd.Series | None = None,target_vol:float = 0.15) -> dict:
    wealth = (1 + r).cumprod()
    out = {
        "ann_return": r.mean() * TRADING_DAYS,
        "ann_vol": r.std() * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": (wealth / wealth.cummax() - 1).min(),
        # how steady was risk? spread of 22-day realised vol (lower = closer to target)
        "vol_of_vol": (r.rolling(22).std() * np.sqrt(TRADING_DAYS)).std(),
        "track_err": ((r.rolling(22).std() * np.sqrt(TRADING_DAYS)) - target_vol).abs().mean(),
    }
    if pos is not None:
        out["turnover"] = pos.diff().abs().mean() * TRADING_DAYS   # position traded per year
        out["cap_binds_%"] = (pos >= 2.0 - 1e-9).mean() * 100
    return out