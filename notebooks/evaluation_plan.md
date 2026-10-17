# Evaluation Plan

Written 2026-10-17, before any model was trained. Changes after this date are
logged at the bottom with a reason.

## 1. Question
Do machine learning models forecast energy futures volatility better than
classical models, and do better forecasts improve a volatility-targeted strategy?

## 2. Data
- Assets: WTI (CL=F), Natural Gas (NG=F), Heating Oil (HO=F), Gasoline (RB=F)
- Context features: OVX, VIX, S&P 500 (closes only)
- Period: 2007-07-30 to 2026-10-15; 4,830 trading days on a shared calendar
- Daily volatility measure: Garman–Klass variance from daily OHLC
- See data_notes.md for cleaning decisions

## 3. Target
Average daily Garman–Klass variance over the next h trading days, forecast at the
close of day t, for h = 1, 5 and 22. All forecasts and losses are in variance units.

## 4. Walk-forward scheme
- Expanding window, retrained once per test year
- Test years: 2012 to 2026, one calendar year each (2026 is partial, ending 2026-10-15)
- Validation: the calendar year before each test year, used only for tuning
- Training: all data before the validation year
- Gap: h trading days between train/validation and validation/test, so targets
  never overlap the next set
- Every model uses identical splits

## 5. Models
- Benchmarks: HAR, GARCH(1,1)
- ML: LightGBM; LSTM (optional)
- Models with randomness (LightGBM, LSTM) are run with 5 random seeds; results
  reported as mean and spread

## 6. Metrics
- Primary: QLIKE loss
- Secondary: MSE
- Significance: Diebold–Mariano test vs HAR, with Newey–West standard errors
  (lag = h − 1)
- A model "beats HAR" only if its QLIKE is lower AND the DM test is significant
  at the 5% level
- Results reported per asset and per horizon
- Regime breakdown (calm vs crisis periods): definitions to be fixed and logged
  below before any model is evaluated on test data

## 7. Economic test
Volatility-targeted position on each asset: target vol ÷ forecast vol, with
costs of 0, 2, 5 and 10 bps. Compare Sharpe ratio, maximum drawdown and turnover
across models.
Settings (target vol, leverage cap, rebalancing frequency): to be fixed and logged
below before any model is evaluated on test data.

## 8. Rules
- The test years are used once, for the final comparison. No tuning on them.
- Features only use information available at the close of day t.
- Hyperparameters are chosen on validation data only.
- Every model variant tried is recorded in the research log, including failures.

## Change log
- (none yet)