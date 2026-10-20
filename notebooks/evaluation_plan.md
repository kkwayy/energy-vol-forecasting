# Evaluation Plan

Written 2026-10-17, before any model was trained. Changes after this date are
logged at the bottom with a reason. Sections 1–8 are kept as originally written;
all later decisions are recorded in the change log.

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

### 2026-10-17 — Benchmarks
- HAR fitted on log variance with the log-normal bias correction exp(σ²/2);
  as it has nothing to tune, it is fitted on train + validation.
- Added a naive benchmark (forecast = past 22-day average variance) as a sanity
  floor for HAR. Reason: any model must beat the simplest possible forecast.
- GARCH(1,1) fitted on close-to-close returns (×100) with `last_obs` set to the
  first test date (verified to exclude it). Forecasts rescaled to Garman–Klass
  units by the ratio of mean GK variance to mean squared return over data before
  each test year. Reason: GARCH targets return variance, which includes the
  overnight move that GK omits; without rescaling QLIKE would penalise a units
  difference rather than forecasting skill.

### 2026-10-19 — LightGBM design
- Feature set for ML models (14 features): HAR terms; own return, negative part,
  5-day sum of negative parts; OVX and VIX converted to daily variance
  ((IV/100)²/252) and 5-day change in OVX variance; other energy assets' 5-day
  GK variance; S&P 500 log return; day of week. All use data up to day t only.
- LightGBM fixed settings: learning rate 0.03, 15 leaves, min 50 samples per leaf,
  80% row and feature subsampling, up to 2,000 trees with early stopping
  (100 rounds) on the validation year. Bias-correction variance taken from
  validation residuals (training residuals are too small for trees).
- Target changed from log variance to log(y) − log(har_m). Reason: with the level
  target, trees could not extrapolate to calm levels (2012 forecasts ~1.7× too
  high, early stopping at 36 trees).
- LightGBM forecasts reported as the average of 5 seeds; single-seed spread
  also reported. Reason: seed 0 alone gave a misleadingly favourable result.

### 2026-10-20 — HAR anchor
- LightGBM target changed to log(y) − log(HAR forecast), with HAR fitted on the
  training rows only inside each fit (validation and test anchors are
  out-of-sample HAR forecasts). Reason: the 2022 diagnostic plot showed the
  har_m-anchored model lagging at spike onsets. This change was made after
  inspecting a test year (WTI, h = 5); it was therefore confirmed on the 11
  asset/horizon combinations not examined during development (5 significant
  wins, 4 no difference, 2 significant losses vs HAR).

### 2026-10-20 — Regime rule (fixed before any regime analysis)
- A test day is high-volatility if OVX at that day's close is in the top 20% of
  its 2012–2026 values; otherwise normal. Chosen after yearly results had been
  seen, but before any regime analysis; not to be changed. The rule only labels
  days for reporting; no model uses it. Limitation: OVX measures crude-oil
  stress, so it is an imperfect regime measure for natural gas.
- DM tests within a regime are approximate (non-contiguous days).

### 2026-10-20 — Additional comparison model
- HAR-IV (HAR + log OVX daily variance, linear, fitted on train + validation)
  added as a planned comparison before any of its results were seen. Reason: to
  separate the information in implied volatility from LightGBM's non-linearity.

### 2026-10-20 — Strategy settings (fixed before any strategy results)
- Long-only; position = 15% ÷ forecast annualised volatility, capped at 2×.
- Main: h = 1 forecasts, rebalanced daily. Robustness: h = 5 forecasts,
  rebalanced every 5 days.
- Costs: 0, 2, 5, 10 bps per unit of position traded. Baseline: constant 1× long.
- Applied to HAR, HAR-IV, LightGBM (HAR-anchored, 5-seed average) and GARCH,
  using the saved out-of-sample forecasts (no refitting).
- Added metrics: vol-of-vol and tracking error (mean absolute gap between the
  strategy's 22-day realised volatility and 15%). Reason: Sharpe ratios are
  dominated by oil's price direction; tracking error isolates forecast quality.
- Known limitation: Yahoo continuous-futures returns include roll jumps; model
  comparisons are fair, absolute returns are not realistic.

### Deviations from the original plan
- MSE (secondary metric, section 6) has not yet been computed; all reported
  results use QLIKE.
- Brent was excluded from the asset universe before modelling (thin NYMEX listing
  before ~2015; see data_notes.md).