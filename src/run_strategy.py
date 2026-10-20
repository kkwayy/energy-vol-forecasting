import pandas as pd
from src import data, config
from src.strategy import vol_target_positions, strategy_returns, performance

panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
fc_dir = data.PROCESSED_DIR.parents[1] / "results" / "forecasts"

rows = []
for asset in config.ENERGY:
    for h, every in [(1, 1), (5, 5)]:                       # main and robustness
        fc = pd.read_csv(fc_dir / f"{asset}_h{h}.csv", index_col=0, parse_dates=True)
        log_ret = panel[f"{asset}_ret"]
        for model in ["HAR", "HARX", "LGBM", "GARCH", "constant"]:
            if model == "constant":
                pos = pd.Series(1.0, index=fc.index)
            else:
                pos = vol_target_positions(fc[model], rebalance_every=every)
            for cost in [0, 2, 5, 10]:
                r = strategy_returns(pos, log_ret, cost)
                rows.append({"asset": asset, "version": f"h{h}", "model": model,
                             "cost_bps": cost, **performance(r, pos)})

res = pd.DataFrame(rows)
res.to_csv(data.PROCESSED_DIR.parents[1] / "results" / "strategy.csv", index=False)
main = res[(res.version == "h1") & (res.cost_bps == 2)]
print(main.round(3).to_string(index=False))