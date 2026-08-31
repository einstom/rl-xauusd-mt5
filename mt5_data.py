"""MetaTrader 5 data bridge: live + historical M1/H1 bars, timezone-safe.

Broker/server time handling
---------------------------
MT5 returns bar times as epoch seconds that encode the BROKER's wall clock
as if it were UTC (the terminal applies no offset).  Most brokers run
EET/EEST server time, for which ``Europe/Helsinki`` is the practical tz
choice.  The safe round-trip used everywhere here:

    epoch seconds ──(unit="s", utc=True)──► "fake UTC" wall clock
                  ──tz_localize(None)────► naive server wall clock
                  ──tz_localize(source_tz)► true tz-aware broker time

and the reverse when building request windows for ``copy_rates_range``.

Bar-time convention: MT5 stamps bars at OPEN time.  The in-memory frames
returned by ``fetch_*`` are shifted to CLOSE-time indexing (matching the
whole pipeline: a timestamp means "this candle is complete"), while the CSV
written by ``download`` keeps MT4/MT5-style OPEN-time strings so it is
byte-compatible with data_loader.load_mt_ohlcv_csv(timestamp_is_bar_open=True).

The ``MetaTrader5`` package is Windows-only and needs a running MT5 terminal.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CFG

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - non-Windows dev machines
    mt5 = None

_TF_MAP = None  # filled lazily; mt5 constants only exist when the package does


def _require_mt5():
    if mt5 is None:
        raise ImportError(
            "The MetaTrader5 package is not installed (Windows-only). "
            "Run this module on the machine that hosts the MT5 terminal: "
            "pip install MetaTrader5"
        )
    global _TF_MAP
    if _TF_MAP is None:
        _TF_MAP = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
    return mt5


def _mt5_timeframe(tf: str):
    _require_mt5()
    if tf not in _TF_MAP:
        raise ValueError(f"Unsupported timeframe '{tf}'. Supported: {sorted(_TF_MAP)}")
    return _TF_MAP[tf]


_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


class Mt5Session:
    """Context manager around mt5.initialize()/shutdown().

    Credentials resolve in order: explicit args → CFG.mt5_* → environment
    variables MT5_TERMINAL_PATH / MT5_LOGIN / MT5_PASSWORD / MT5_SERVER.
    With everything None it attaches to the already-running, logged-in terminal.
    """

    def __init__(
        self,
        terminal_path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
    ):
        self.terminal_path = terminal_path or CFG.mt5_terminal_path or os.environ.get("MT5_TERMINAL_PATH")
        env_login = os.environ.get("MT5_LOGIN")
        self.login = login or CFG.mt5_login or (int(env_login) if env_login else None)
        self.password = password or CFG.mt5_password or os.environ.get("MT5_PASSWORD")
        self.server = server or CFG.mt5_server or os.environ.get("MT5_SERVER")

    def __enter__(self):
        _require_mt5()
        kwargs = {}
        if self.login:
            kwargs.update(login=self.login, password=self.password, server=self.server)
        args = (self.terminal_path,) if self.terminal_path else ()
        if not mt5.initialize(*args, **kwargs):
            raise ConnectionError(f"mt5.initialize() failed: {mt5.last_error()}")
        info = mt5.terminal_info()
        acct = mt5.account_info()
        print(f"MT5 connected: build {getattr(info, 'build', '?')}, "
              f"account {getattr(acct, 'login', '?')} @ {getattr(acct, 'server', '?')}, "
              f"trade_allowed={getattr(info, 'trade_allowed', '?')}")
        return self

    def __exit__(self, exc_type, exc, tb):
        mt5.shutdown()
        return False


# ── Time conversion helpers ──────────────────────────────────────────────────

def server_epoch_to_tz_index(epoch_seconds, source_tz: str = None) -> pd.DatetimeIndex:
    """Epoch seconds carrying broker wall-clock → tz-aware DatetimeIndex."""
    source_tz = source_tz or CFG.source_tz
    naive = pd.to_datetime(epoch_seconds, unit="s", utc=True).tz_localize(None)
    idx = pd.DatetimeIndex(naive)
    try:
        return idx.tz_localize(source_tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        # DST fall-back hour is genuinely ambiguous in server wall-clock data;
        # resolving it as DST keeps the index monotonic enough for our use.
        return idx.tz_localize(source_tz, ambiguous=True, nonexistent="shift_forward")


def tz_aware_to_server_utc(ts: pd.Timestamp | datetime, source_tz: str = None) -> datetime:
    """tz-aware timestamp → broker wall clock stamped as UTC, the form
    copy_rates_range/copy_ticks_range expect (they apply no offset)."""
    source_tz = source_tz or CFG.source_tz
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("pass a tz-aware timestamp")
    wall = ts.tz_convert(source_tz).tz_localize(None)
    return wall.to_pydatetime().replace(tzinfo=timezone.utc)


def rates_to_df(rates, timeframe: str, source_tz: str = None,
                shift_to_close: bool = True) -> pd.DataFrame:
    """MT5 rates array → OHLCV DataFrame indexed by tz-aware bar CLOSE time."""
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(rates)
    idx = server_epoch_to_tz_index(df["time"].values, source_tz)
    if shift_to_close:
        idx = idx + pd.Timedelta(minutes=_TF_MINUTES[timeframe])
    df.index = pd.DatetimeIndex(idx, name="Time")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "tick_volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    return df[~df.index.duplicated(keep="last")]


# ── Fetching ─────────────────────────────────────────────────────────────────

def fetch_recent_bars(symbol: str, timeframe: str, count: int,
                      include_forming: bool = False) -> pd.DataFrame:
    """Latest ``count`` bars.  By default the forming (incomplete) bar at
    position 0 is EXCLUDED so every returned bar is closed — the only safe
    input for causal features."""
    _require_mt5()
    start_pos = 0 if include_forming else 1
    rates = mt5.copy_rates_from_pos(symbol, _mt5_timeframe(timeframe), start_pos, count)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos({symbol}, {timeframe}) failed: {mt5.last_error()}")
    return rates_to_df(rates, timeframe)


def fetch_bars_range(symbol: str, timeframe: str,
                     start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Bars whose OPEN time falls in [start, end] (tz-aware inputs)."""
    _require_mt5()
    rates = mt5.copy_rates_range(
        symbol, _mt5_timeframe(timeframe),
        tz_aware_to_server_utc(start), tz_aware_to_server_utc(end),
    )
    if rates is None:
        raise RuntimeError(f"copy_rates_range({symbol}, {timeframe}) failed: {mt5.last_error()}")
    return rates_to_df(rates, timeframe)


def latest_closed_bar_time(symbol: str, timeframe: str) -> pd.Timestamp:
    """Close time of the most recently completed bar (tz-aware)."""
    df = fetch_recent_bars(symbol, timeframe, count=1)
    if df.empty:
        raise RuntimeError(f"No {timeframe} bars available for {symbol}")
    return df.index[-1]


def download_history(
    symbol: str = None,
    timeframe: str = "M1",
    years: float = 10.0,
    out_csv: str | Path = None,
    chunk_days: int = 30,
) -> Path:
    """Download historical bars in chunks and write an MT4/MT5-style CSV
    (open-time 'Time (EET)' strings) that data_loader reads unchanged.

    The terminal serves only as much history as the broker keeps; raise
    'Max bars in chart' in MT5 Options → Charts if chunks come back short.
    """
    symbol = symbol or CFG.mt5_symbol
    out_csv = Path(out_csv) if out_csv else CFG.csv_path
    _require_mt5()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol}) failed: {mt5.last_error()}")

    now = pd.Timestamp.now(tz=CFG.source_tz)
    start_all = now - pd.DateOffset(months=int(round(years * 12)))
    print(f"Downloading {symbol} {timeframe}: {start_all.date()} → {now.date()} "
          f"in {chunk_days}-day chunks …")

    parts: list[pd.DataFrame] = []
    t0 = start_all
    while t0 < now:
        t1 = min(t0 + timedelta(days=chunk_days), now)
        # Keep raw open-time stamps here; the CSV must store open time.
        rates = mt5.copy_rates_range(
            symbol, _mt5_timeframe(timeframe),
            tz_aware_to_server_utc(t0), tz_aware_to_server_utc(t1),
        )
        if rates is not None and len(rates):
            parts.append(rates_to_df(rates, timeframe, shift_to_close=False))
            print(f"  {t0.date()} → {t1.date()}: {len(rates):>7,} bars")
        else:
            print(f"  {t0.date()} → {t1.date()}: no data ({mt5.last_error()})")
        t0 = t1

    if not parts:
        raise RuntimeError("No history returned — check symbol, broker history depth, "
                           "and 'Max bars in chart'.")

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    out = df.copy()
    out.insert(0, CFG.time_col,
               out.index.tz_localize(None).strftime("%Y.%m.%d %H:%M:%S"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out):,} bars ({out.index.min()} → {out.index.max()}) → {out_csv}")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 data bridge.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="download M1 history to CSV for training")
    p_dl.add_argument("--symbol", default=None)
    p_dl.add_argument("--timeframe", default="M1")
    p_dl.add_argument("--years", type=float, default=10.0)
    p_dl.add_argument("--out", default=None)

    sub.add_parser("check", help="connect, print terminal/account/symbol status")

    args = parser.parse_args()
    with Mt5Session():
        if args.cmd == "download":
            download_history(symbol=args.symbol, timeframe=args.timeframe,
                             years=args.years, out_csv=args.out)
        elif args.cmd == "check":
            symbol = CFG.mt5_symbol
            mt5.symbol_select(symbol, True)
            si = mt5.symbol_info(symbol)
            print(f"{symbol}: bid={si.bid} ask={si.ask} spread={si.spread}pt "
                  f"digits={si.digits} contract={si.trade_contract_size}")
            last = latest_closed_bar_time(symbol, "H1")
            print(f"Latest closed H1 bar (close time, {CFG.source_tz}): {last}")
