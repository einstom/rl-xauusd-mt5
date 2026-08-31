"""CSV loading, timezone-safe indexing, resampling and temporal splits.

All bars are indexed by candle CLOSE time in broker/server time
(EET/EEST via Europe/Helsinki), so a timestamp always means "this candle is
complete and known" — the features built on it are causal by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip whitespace and the <ANGLE> brackets of MT5's native Bars export.
    df.columns = [c.strip().strip("<>") for c in df.columns]

    # Handle small variations in capitalization or accidental trailing spaces.
    mapping = {}
    for c in df.columns:
        low = c.lower().strip()
        if low == "open": mapping[c] = "Open"
        if low == "high": mapping[c] = "High"
        if low == "low": mapping[c] = "Low"
        if low == "close": mapping[c] = "Close"
        # Prefer TICKVOL (always populated) over VOL (usually 0 for CFD gold).
        if low in ("volume", "tick_volume", "tickvol"): mapping[c] = "Volume"
    return df.rename(columns=mapping)


def _read_mt_csv(path: Path) -> pd.DataFrame:
    """Read either CSV flavour this project accepts:

    * comma-separated with a combined time column — mt5_data.py download /
      MT4-style exports:  ``Time (EET),Open,High,Low,Close,Volume``
    * MT5's built-in Bars export (Ctrl+U → Bars → Export Bars): tab-separated
      with ``<DATE>\t<TIME>\t<OPEN>…<TICKVOL>`` — the no-code path to get M1
      history out of the macOS/Windows terminal for research.

    For the second flavour DATE and TIME are combined into a single
    'Time (EET)' column so both continue through one code path.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(path, sep=sep)
    df = _standardize_columns(df)

    cols_upper = {c.upper(): c for c in df.columns}
    if "DATE" in cols_upper:
        date = df[cols_upper["DATE"]].astype(str).str.strip()
        if "TIME" in cols_upper:
            time = df[cols_upper["TIME"]].astype(str).str.strip()
        else:
            time = "00:00:00"  # D1 exports carry no TIME column
        df["Time (EET)"] = date + " " + time
    return df


def load_mt_ohlcv_csv(
    path: str | Path,
    time_col: str = "Time (EET)",
    source_tz: str = "Europe/Helsinki",
    timestamp_is_bar_open: bool = True,
    bar_duration: str = "1min",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_days_for_demo: Optional[int] = None,
) -> pd.DataFrame:
    """Load MT4/MT5 OHLCV CSV with a broker-time column like 'Time (EET)'.

    Returns a timezone-aware DataFrame indexed by broker/server bar CLOSE time.
    MT4/MT5 exports commonly timestamp candles by their open time; when
    timestamp_is_bar_open=True, the index is shifted forward by bar_duration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}. Run `python mt5_data.py download` on the "
            f"MT5 machine (or drop an MT4/MT5 export there) or change CFG.csv_path."
        )

    df = _read_mt_csv(path)

    if time_col not in df.columns:
        # Robust fallback: find a column that starts with Time.
        candidates = [c for c in df.columns if c.lower().startswith("time")]
        if not candidates:
            raise ValueError(f"Could not find time column '{time_col}'. Columns: {df.columns.tolist()}")
        time_col = candidates[0]

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}. Columns found: {df.columns.tolist()}")

    # Numeric coercion, robust to accidental spaces.
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # MT-style exports are usually 2020.01.09 01:00:00.
    ts = pd.to_datetime(df[time_col].astype(str).str.strip(), errors="coerce", format="%Y.%m.%d %H:%M:%S")
    if ts.isna().mean() > 0.01:
        # More flexible fallback.
        ts = pd.to_datetime(df[time_col].astype(str).str.strip(), errors="coerce")

    df = df.loc[~ts.isna()].copy()
    ts = ts.loc[~ts.isna()]

    # Localize broker/server time. DST ambiguity can happen; fall back safely.
    try:
        idx = ts.dt.tz_localize(source_tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        idx = ts.dt.tz_localize(source_tz, ambiguous=True, nonexistent="shift_forward")

    idx = pd.DatetimeIndex(idx, name="Time")
    if timestamp_is_bar_open:
        bar_delta = pd.to_timedelta(bar_duration)
        if bar_delta <= pd.Timedelta(0):
            raise ValueError(f"bar_duration must be positive, got {bar_duration!r}")
        idx = idx + bar_delta

    df.index = idx
    df = df[OHLCV].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    # Basic sanity filters.
    df = df[(df["High"] >= df[["Open", "Close"]].max(axis=1)) &
            (df["Low"] <= df[["Open", "Close"]].min(axis=1))]

    if start_date:
        df = df.loc[pd.Timestamp(start_date, tz=source_tz):]
    if end_date:
        df = df.loc[:pd.Timestamp(end_date, tz=source_tz)]

    if max_days_for_demo is not None and len(df):
        last = df.index.max()
        first = last - pd.Timedelta(days=max_days_for_demo)
        df = df.loc[first:last]

    return df


def resample_ohlcv(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    """Resample OHLCV while preserving the timezone-aware close-time index."""
    out = df.resample(rule, label="right", closed="right").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Small table for notebook/CLI sanity checks."""
    if df.empty:
        return pd.DataFrame({"value": []})
    diffs = df.index.to_series().diff().dropna()
    return pd.DataFrame({
        "value": {
            "rows": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "timezone": str(df.index.tz),
            "median_spacing": str(diffs.median()) if len(diffs) else "NA",
            "missing_ohlc_rows": int(df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum()),
            "duplicated_timestamps": int(df.index.duplicated().sum()),
            "min_close": float(df["Close"].min()),
            "max_close": float(df["Close"].max()),
        }
    })


def split_train_val_test(df: pd.DataFrame, train_frac=0.80, val_frac=0.10, embargo_bars=200):
    """Temporal split with embargo gaps.

    embargo_bars removes that many bars from each side of every split boundary
    to prevent EWM-based features from carrying training-period information
    into val/test observations.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:max(train_end - embargo_bars, 0)]
    val = df.iloc[min(train_end + embargo_bars, n):max(val_end - embargo_bars, 0)]
    test = df.iloc[min(val_end + embargo_bars, n):]
    return train, val, test


def make_sliding_folds(
    df: pd.DataFrame,
    train_years: float = 5.0,
    val_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
    embargo_bars: int = 200,
):
    """Rolling train / val / test walk-forward with calendar-sized windows.

    Simulates periodic retraining.  Each fold is::

        train = [t0,            t0 + train_years)
        val   = [train_end,     + val_months)     ← checkpoint selection
        test  = [val_end,       + test_months)    ← TRUE out-of-sample

    then ``t0`` advances by ``step_months`` and the whole window slides forward.
    With ``step_months == test_months`` the test windows are contiguous, so
    stitching them yields one continuous out-of-sample track record — exactly
    the "retrain every step_months, trade the next test_months live" workflow.

    ``embargo_bars`` are purged at the START of the val and test windows so the
    EWM-based features there cannot straddle a boundary into the previous
    segment.

    Returns
    -------
    list[tuple[DataFrame, DataFrame, DataFrame]]
        ``[(train, val, test), …]`` in chronological order.  Only folds whose
        full test window fits within the data are returned.
    """
    if df.empty:
        return []

    idx = df.index
    start, end = idx.min(), idx.max()
    train_off = pd.DateOffset(months=int(round(train_years * 12)))
    val_off = pd.DateOffset(months=val_months)
    test_off = pd.DateOffset(months=test_months)
    step_off = pd.DateOffset(months=step_months)

    folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    t0 = start
    while True:
        val_start = t0 + train_off
        test_start = val_start + val_off
        test_end = test_start + test_off
        if test_end > end:
            break  # require a full test window; stop once one no longer fits

        a = int(idx.searchsorted(t0, side="left"))
        b = int(idx.searchsorted(val_start, side="left"))
        c = int(idx.searchsorted(test_start, side="left"))
        d = int(idx.searchsorted(test_end, side="left"))

        train = df.iloc[a:max(b - embargo_bars, a)]
        val = df.iloc[min(b + embargo_bars, c):c]
        test = df.iloc[min(c + embargo_bars, d):d]

        if len(train) and len(val) and len(test):
            folds.append((train, val, test))

        t0 = t0 + step_off
        if t0 + train_off >= end:
            break  # next train window would run past the data

    return folds
