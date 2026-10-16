"""Download and clean prices; compute daily volatility."""

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from  src import config
from pathlib import Path
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def download(tickers: dict, start: str, end: str | None) -> dict[str, pd.DataFrame]:
    """Download daily OHLC for each ticker with yfinance.

    Returns {name: DataFrame[Open, High, Low, Close]}.
    TODO: save raw files to data/raw/ so you don't re-download every run.
    """


    result ={}
    for name,code in tickers.items():
        path = RAW_DIR/ f"{name}.csv"

        if path.exists():
            df = pd.read_csv(path,index_col =0,parse_dates=True)

        else:
            df= yf.download(code,start,end,auto_adjust=False, multi_level_index=False)
            df= df[["Open", "High", "Low", "Close", "Volume"]]

            RAW_DIR.mkdir(parents=True, exist_ok=True)

            df = df.dropna(how="all")
            if df.empty:
                print(f"Warning: no data for {name} ({code})")
                continue
            df.to_csv(path)



        result[name]= df

    return result




def clean(raw: dict[str, pd.DataFrame], energy_names: list[str]) -> dict[str, pd.DataFrame]:
    """Clean each series, then align all series onto one calendar.

    Per series: sort, trim to [START_DATE, END_DATE), drop duplicate dates, and drop rows
    with missing, non-positive or inconsistent OHLC prices (High < Low, Open/Close outside
    [Low, High]). This removes WTI's negative-price days in April 2020.

    Roll jumps are not corrected (no contract-level data); large close-to-close moves are
    reported for inspection only.
    """
    price_cols = ["Open", "High", "Low", "Close"]
    cleaned = {}

    for name, df in raw.items():
        # 1. Order, trim, de-duplicate
        df = df.sort_index()
        df = df.loc[config.START_DATE:]
        if config.END_DATE is not None:
            df = df[df.index < config.END_DATE]   # match yfinance's exclusive end
        df = df[~df.index.duplicated(keep="first")]

        # 2. Invalid rows
        missing = df[price_cols].isna().any(axis=1)
        non_pos = (df[price_cols] <= 0).any(axis=1)
        high_low = df["High"] < df["Low"]
        out_of_range = (
            (df["Open"] < df["Low"]) | (df["Open"] > df["High"]) |
            (df["Close"] < df["Low"]) | (df["Close"] > df["High"])
        )
        bad = missing | non_pos | high_low | out_of_range
        print(f"{name}: removed {bad.sum()} rows (missing={missing.sum()}, "
              f"non_pos={non_pos.sum()}, high_low={high_low.sum()}, "
              f"out_of_range={out_of_range.sum()})")
        df = df[~bad]

        # 3. Flags for inspection (energy assets only)
        if name in energy_names:
            zero_range = df["High"] == df["Low"]
            r = np.log(df["Close"]).diff()
            big = r.abs() > 0.2
            print(f"{name}: zero-range days={zero_range.sum()}, moves >20%={big.sum()}")
            if big.any():
                print(r[big].round(3).to_string())

        cleaned[name] = df



    # 4. Align onto one calendar
    calendar = cleaned[energy_names[0]].index
    for name in energy_names[1:]:
        calendar = calendar.intersection(cleaned[name].index)
    print(f"Calendar: {len(calendar)} dates")

    for name, df in cleaned.items():
        if name in energy_names:
            cleaned[name] = df.reindex(calendar)
        else:
            cleaned[name] =df.reindex(calendar).ffill(limit=1)

    for name, df in cleaned.items():
        n_missing = df[price_cols].isna().any(axis=1)
        if n_missing.any():
            print(f"WARNING {name}: {n_missing.sum()} dates still missing")
            print(df.index[n_missing].date)


    return cleaned

def garman_klass(ohlc: pd.DataFrame) -> pd.Series:
    """Daily Garman–Klass variance from Open/High/Low/Close.

    Returns daily (not annualised) variance. Multiply by 252 to annualise;
    take the square root for volatility.
    """
    log_hl = np.log(ohlc["High"] / ohlc["Low"])
    log_co = np.log(ohlc["Close"] / ohlc["Open"])
    gk_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2


    return gk_var.rename("gk_var")




def build_panel() -> pd.DataFrame:
    """Run download -> clean -> volatility for all assets; save to data/processed/."""

    raw = {**download(config.ENERGY, config.START_DATE, config.END_DATE),
           **download(config.CONTEXT, config.START_DATE, config.END_DATE)}
    cleaned = clean(raw, list(config.ENERGY))

    cols = {}

    for name in config.ENERGY:                      # WTI, NatGas, HeatingOil, Gasoline
        df = cleaned[name]
        cols[f"{name}_gk"] = garman_klass(df)
        cols[f"{name}_ret"] = np.log(df["Close"]).diff()

    for name in config.CONTEXT:                     # OVX, VIX, SPX
        cols[name] = cleaned[name]["Close"]

    panel = pd.DataFrame(cols)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED_DIR / "panel.csv")
    return panel



    return panel
if __name__ == "__main__":
    panel = build_panel()
    print(panel.shape)
    print(panel.head())

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in config.ENERGY:
        gk = panel[f"{name}_gk"]
        smoothed = np.sqrt(gk.rolling(22).mean() * 252)
        ax.plot(smoothed.index, smoothed, label=name)
        print(name, "min:", gk.min(), "missing:", gk.isna().sum(),
              "median ann. vol:", round(np.sqrt(gk.median() * 252), 3))

    ax.set_ylabel("Annualised volatility (22-day avg)")
    ax.legend()
    fig.savefig(Path(__file__).resolve().parents[1] / "results" / "figures" / "gk_vol.png", dpi=150)
    plt.show()
