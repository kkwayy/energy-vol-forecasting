import pandas as pd
from src import data

res = pd.read_csv(data.PROCESSED_DIR.parents[1] / "results" / "strategy.csv")
models = ["HAR", "HARX", "LGBM", "GARCH"]
vt = res[res.model.isin(models)]

# A. Forecast quality: tracking error, daily (h1) vs every-5-days (h5), at 2 bps
print("Tracking error to 15% target (lower = better):")
print(vt[vt.cost_bps == 2].pivot_table(index=["asset", "version"], columns="model",
                                       values="track_err").round(4))

# B. Cost sensitivity: Sharpe at 0 vs 10 bps, daily version
print("\nSharpe by cost (h1):")
print(vt[vt.version == "h1"].pivot_table(index=["asset", "cost_bps"], columns="model",
                                          values="sharpe").round(3))

# C. Turnover, daily vs every 5 days
print("\nTurnover (x position per year):")
print(vt[vt.cost_bps == 0].pivot_table(index=["asset", "version"], columns="model",
                                       values="turnover").round(1))