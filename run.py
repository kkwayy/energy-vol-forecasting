"""Run the full pipeline. Uncomment stages as you build them."""

from src import config


def main():
    # 1. Data
    # panel = data.build_panel()

    # 2. For each asset and horizon:
    #    X, y = features.build_dataset(panel, asset, h)
    #    forecasts = {
    #        "HAR":   walkforward.run_walkforward(HAR, ...),
    #        "GARCH": ...,
    #        "LGBM":  ...,
    #        "LSTM":  ...,   # optional
    #    }
    #    table = evaluate.results_table(forecasts, realised, h)

    # 3. SHAP on LightGBM, by period

    # 4. Strategy: vol targeting with each model's forecasts, at each cost level

    # 5. Save tables to results/ and figures to results/figures/
    pass


if __name__ == "__main__":
    main()
