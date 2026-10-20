import pandas as pd
from src import data, config
from src.evaluate import qlike, diebold_mariano

panel = pd.read_csv(data.PROCESSED_DIR / "panel.csv", index_col=0, parse_dates=True)
fc_dir = data.PROCESSED_DIR.parents[1] / "results" / "forecasts"

rows = []
for asset in config.ENERGY:
    for h in config.HORIZONS:
        fc = pd.read_csv(fc_dir / f"{asset}_h{h}.csv", index_col=0, parse_dates=True)
        ovx = panel.loc[fc.index, "OVX"]
        high = ovx > ovx.quantile(0.8)                     # the fixed rule

        for regime, mask in [("high", high), ("normal", ~high)]:
            f = fc[mask]
            l_har, l_lgbm = qlike(f["actual"], f["HAR"]), qlike(f["actual"], f["LGBM"])
            stat, p = diebold_mariano(l_lgbm, l_har, h)
            rows.append({"asset": asset, "h": h, "regime": regime, "days": mask.sum(),
                         "QLIKE_HAR": l_har.mean(), "QLIKE_LGBM": l_lgbm.mean(),
                         "DM_LGBM_vs_HAR": stat, "p": p})

print(pd.DataFrame(rows).round(3).to_string(index=False))