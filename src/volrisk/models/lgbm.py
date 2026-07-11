"""LightGBM regressor on the full feature set, deliberately un-tuned.

Fixed, modest hyperparameters — no per-ticker tuning, no early stopping — so
the ablation compares model CLASSES rather than tuning effort. random_state
pins determinism.
"""

from __future__ import annotations

from lightgbm import LGBMRegressor

LGBM_FEATURES: tuple[str, ...] = (
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "r2",
    "park_var",
    "gk_var",
    "rvol_5",
    "rvol_21",
    "rvol_63",
    "har_rv_w",
    "har_rv_m",
)

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}


def make_lgbm_model() -> LGBMRegressor:
    return LGBMRegressor(**LGBM_PARAMS)
