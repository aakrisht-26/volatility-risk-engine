"""Log returns on adjusted close, and the telescoping acceptance identity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``log_return = ln(adj_close_t / adj_close_{t-1})`` for one ticker.

    Rows must belong to a single ticker, sorted by trade_date. Returns are
    computed over consecutive AVAILABLE sessions — no calendar fill — so a data
    gap yields one honest multi-day return rather than fabricated bars. The
    first row's return is NULL by construction.

    Adjusted close is used because unadjusted returns embed dividend drops and
    split jumps that are not market risk.
    """
    out = df.copy()
    out["log_return"] = np.log(out["adj_close"]).diff()
    return out


def telescoping_check(df: pd.DataFrame) -> tuple[float, float]:
    """Return (sum of log returns, ln(P_end / P_start)) for one ticker's frame.

    Log returns telescope: the two values are equal by identity, so any
    material difference means the return series is corrupt. This is the
    Step-5 acceptance check.
    """
    total = float(df["log_return"].iloc[1:].sum())
    endpoints = float(np.log(df["adj_close"].iloc[-1] / df["adj_close"].iloc[0]))
    return total, endpoints
