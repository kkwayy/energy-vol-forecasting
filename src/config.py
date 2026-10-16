"""Project settings. Fix these before running any model, and don't change them after seeing test results."""

ENERGY = {
    "WTI": "CL=F",
    #"Brent": "BZ=F",
    "NatGas": "NG=F",
    "HeatingOil": "HO=F",
    "Gasoline": "RB=F",
}
CONTEXT = {
    "OVX": "^OVX",
    "VIX": "^VIX",
    "SPX": "^GSPC",
}

START_DATE = "2007-07-30" # TODO: check earliest common date across tickers (OVX starts ~2007)
END_DATE = "2026-10-16"        # None = up to today

HORIZONS = [1, 5, 22]       # forecast horizons in trading days

# Walk-forward: expanding window, retrain yearly
FIRST_TEST_YEAR = 2012
VALIDATION_YEARS = 1        # years before each test year used for tuning

TRADING_DAYS = 252
RANDOM_SEEDS = [0, 1, 2, 3, 4]

# Strategy
TARGET_VOL = 0.15           # annualised
COST_BPS = [0, 2, 5, 10]    # transaction cost levels to test
