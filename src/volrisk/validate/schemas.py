"""Pandera schema for canonical daily OHLCV bars.

Every batch that enters the pipeline is validated against this schema:
the backfill validates each fetched batch immediately after landing it in
``data/raw/`` (the landing zone keeps exactly what the provider returned, for
replay and forensics; a failed batch still halts the pipeline before anything
downstream consumes it), and the Step-4 loader revalidates before Postgres.

Failures raise :class:`pandera.errors.SchemaErrors` with every violation in
the batch collected and logged, not just the first.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

logger = logging.getLogger(__name__)

_POSITIVE = pa.Check.gt(0)


def _is_python_date(series: pd.Series) -> pd.Series:
    """True where values are plain ``datetime.date`` (no intraday component)."""
    return series.map(lambda v: isinstance(v, dt.date) and not isinstance(v, dt.datetime))


RAW_DAILY_BARS_SCHEMA = pa.DataFrameSchema(
    columns={
        "ticker": pa.Column(str, nullable=False),
        "trade_date": pa.Column(
            object,
            checks=pa.Check(_is_python_date, name="is_python_date"),
            nullable=False,
        ),
        "open": pa.Column("float64", _POSITIVE, nullable=False),
        "high": pa.Column("float64", _POSITIVE, nullable=False),
        "low": pa.Column("float64", _POSITIVE, nullable=False),
        "close": pa.Column("float64", _POSITIVE, nullable=False),
        "adj_close": pa.Column("float64", _POSITIVE, nullable=False),
        # ^VIX legitimately reports zero volume; missing volume stays NA.
        "volume": pa.Column("Int64", pa.Check.ge(0), nullable=True),
    },
    checks=[
        pa.Check(lambda df: df["high"] >= df["low"], name="high_ge_low"),
        pa.Check(
            lambda df: (df["open"] <= df["high"]) & (df["open"] >= df["low"]),
            name="open_within_high_low",
        ),
        pa.Check(
            lambda df: (df["close"] <= df["high"]) & (df["close"] >= df["low"]),
            name="close_within_high_low",
        ),
    ],
    unique=["ticker", "trade_date"],
    strict=True,  # exactly the canonical columns, nothing else
    name="raw_daily_bars",
)


def validate_daily_bars(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Validate one batch of canonical daily bars; return it unchanged on success.

    Raises :class:`pandera.errors.SchemaErrors` carrying every violation in the
    batch; the full failure table is logged first so pipeline logs are
    actionable without a debugger. ``context`` labels the log line (ticker,
    file, ...).
    """
    try:
        return RAW_DAILY_BARS_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        label = f" [{context}]" if context else ""
        logger.error(
            "daily-bars validation failed%s: %d violation(s)\n%s",
            label,
            len(exc.failure_cases),
            exc.failure_cases.to_string(),
        )
        raise
