# ML Volatility Forecasting for Energy Futures

**Question:** Do machine learning models forecast energy futures volatility better than classical models, and do better forecasts translate into better risk-adjusted returns?

## Structure

```
data/raw/            downloaded prices (not committed)
data/processed/      cleaned panel + features (not committed)
src/config.py        tickers, dates, horizons, walk-forward settings
src/data.py          download, clean, Garman–Klass volatility
src/features.py      targets + feature set (lookahead-safe)
src/walkforward.py   expanding-window train/val/test splits
src/models/          benchmarks (HAR, GARCH), LightGBM, LSTM
src/evaluate.py      QLIKE, MSE, Diebold–Mariano
src/strategy.py      volatility-targeted strategy + metrics
run.py               runs the full pipeline
notebooks/           exploration and figures
results/             tables and figures for the write-up
```

## Order of work

1. `data.py` → `features.py` → `walkforward.py`
2. `models/benchmarks.py` + `evaluate.py` (benchmark results)
3. `models/ml.py` (LightGBM) — **version 1 checkpoint**
4. `models/lstm.py`, SHAP analysis
5. `strategy.py` (economic value)
6. Write-up below

## Results

_TODO_

## Limitations

- Garman–Klass volatility from daily OHLC is noisier than intraday realised volatility
- Yahoo continuous futures contain roll jumps
- Single data source
