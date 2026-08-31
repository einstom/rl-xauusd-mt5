"""Equity-curve and trade-log metrics shared by training, selection and eval."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def summarize_equity(equity_df: pd.DataFrame, initial_equity: float = 10_000.0,
                     periods_per_year: int = 6_003) -> dict:
    """periods_per_year default matches H1 XAUUSD (23h × 261 days)."""
    if equity_df.empty:
        return {}
    eq = equity_df["equity"].astype(float)
    n = len(eq)
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = eq.iloc[-1] / initial_equity - 1.0
    max_dd = drawdown(eq).min()

    # Annualised return (CAGR-style scaling by bar count).
    ann_return = (1.0 + total_return) ** (periods_per_year / max(n, 1)) - 1.0

    # Sharpe (excess return over 0-rate risk-free assumption, per-bar scale).
    sharpe = np.nan
    ret_std = returns.std()
    if ret_std and ret_std > 0:
        sharpe = returns.mean() / ret_std * np.sqrt(periods_per_year)

    # Sortino: penalises only downside volatility.
    sortino = np.nan
    down = returns[returns < 0]
    if len(down) > 1:
        down_std = down.std()
        if down_std > 0:
            sortino = returns.mean() / down_std * np.sqrt(periods_per_year)

    # Calmar: annualised return divided by absolute max drawdown.
    calmar = np.nan
    if np.isfinite(ann_return) and abs(max_dd) > 1e-10:
        calmar = ann_return / abs(max_dd)

    return {
        "initial_equity": initial_equity,
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "annualized_return_pct": float(ann_return * 100) if np.isfinite(ann_return) else np.nan,
        "max_drawdown_pct": float(max_dd * 100),
        "sharpe_like": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino_ratio": float(sortino) if np.isfinite(sortino) else np.nan,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else np.nan,
        "n_equity_points": int(n),
    }


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "avg_r": np.nan,
            "median_r": np.nan,
            "avg_bars_in_trade": np.nan,
        }
    pnl = trades["pnl"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return {
        "n_trades": int(len(trades)),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": float(pf),
        "avg_r": float(trades["r_mult"].mean()),
        "median_r": float(trades["r_mult"].median()),
        "avg_bars_in_trade": float(trades["bars_in_trade"].mean()),
    }


def full_report(equity_df: pd.DataFrame, trades: pd.DataFrame,
                initial_equity: float = 10_000.0,
                periods_per_year: int = 6_003) -> pd.DataFrame:
    d = {}
    d.update(summarize_equity(equity_df, initial_equity=initial_equity,
                              periods_per_year=periods_per_year))
    d.update(summarize_trades(trades))
    return pd.DataFrame.from_dict(d, orient="index", columns=["value"])
