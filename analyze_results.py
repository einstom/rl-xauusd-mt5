"""Post-run diagnostics for a sliding walk-forward result.

Reads the artifacts train_ppo.py --mode sliding leaves under models/:
  sliding_walk_forward_summary.csv   per-fold TEST metrics
  sliding_oos_equity.csv             stitched out-of-sample equity curve
  sliding_oos_trades.csv             every test-window trade (fold-tagged)

and prints:
  1. per-fold table with the gate's weak spots flagged
  2. drawdown timeline of the stitched curve (worst episodes, time under water)
  3. trade quality: R-multiple stats, exit-reason and bracket-choice breakdown,
     PnL by UTC session and by fold
  4. COST SENSITIVITY — re-prices every trade under a grid of
     spread/slippage/commission assumptions and reports how return and profit
     factor hold up.  Caveat: this re-pricing keeps the original entries/exits
     and position sizes (no compounding feedback, and a retrained agent would
     adapt its behaviour to higher costs), so treat it as a first-order bound;
     the honest test is retraining with `--spread/--slippage/--commission`.

Usage:
  python analyze_results.py                       # reads models/
  python analyze_results.py --models-dir models --spreads 0.20,0.30,0.45 --commissions 0.0,0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import drawdown


def _load(models_dir: Path):
    summary_p = models_dir / "sliding_walk_forward_summary.csv"
    equity_p = models_dir / "sliding_oos_equity.csv"
    trades_p = models_dir / "sliding_oos_trades.csv"
    if not summary_p.exists():
        raise SystemExit(f"{summary_p} not found — run train_ppo.py --mode sliding first.")
    summary = pd.read_csv(summary_p)
    equity = pd.read_csv(equity_p, index_col=0) if equity_p.exists() else None
    if equity is not None:
        # The stitched index mixes +02:00/+03:00 (EET/EEST) offsets; force UTC.
        equity.index = pd.to_datetime(equity.index, utc=True, errors="coerce")
        equity = equity[equity.index.notna()]
    trades = pd.read_csv(trades_p) if trades_p.exists() and trades_p.stat().st_size > 2 else None
    if trades is not None and trades.empty:
        trades = None
    if trades is not None:
        for col in ("entry_time", "exit_time"):
            trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce")
    return summary, equity, trades


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def per_fold(summary: pd.DataFrame) -> None:
    _section("PER-FOLD TEST (true out-of-sample)")
    cols = [c for c in ["fold", "test_start", "test_end", "test_return_pct", "test_sharpe",
                        "test_profit_factor", "test_win_rate_pct", "test_max_dd_pct",
                        "test_avg_r", "test_n_trades"] if c in summary.columns]
    print(summary[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:,.2f}"))
    pf = summary["test_profit_factor"]
    worst = summary.loc[pf.idxmin()]
    print(f"\n  weakest fold : {int(worst['fold'])} "
          f"({worst['test_start']} → {worst['test_end']})  PF={worst['test_profit_factor']:.2f}  "
          f"return={worst['test_return_pct']:+.1f}%")
    latest = summary.iloc[-1]
    print(f"  latest fold  : {int(latest['fold'])}  PF={latest['test_profit_factor']:.2f}  "
          f"return={latest['test_return_pct']:+.1f}%  "
          f"(the deployable model's own regime — weight it heavily)")


def drawdown_timeline(equity: pd.DataFrame) -> None:
    if equity is None or equity.empty:
        return
    _section("STITCHED OOS DRAWDOWN TIMELINE")
    eq = equity["equity"].astype(float)
    dd = drawdown(eq)
    print(f"  final equity : {eq.iloc[-1]:,.0f}  (start {eq.iloc[0]:,.0f})")
    print(f"  max drawdown : {dd.min() * 100:+.1f}%")
    under = (dd < 0).mean()
    print(f"  time under water: {under * 100:.0f}% of bars")

    # Worst distinct drawdown episodes: peak → trough → recovery.
    peak_idx = eq.cummax()
    episodes = []
    in_dd = False
    for t, (e, p) in zip(eq.index, zip(eq, peak_idx)):
        if e < p and not in_dd:
            in_dd, ep_start, ep_low, ep_low_t = True, t, e, t
        elif in_dd:
            if e < ep_low:
                ep_low, ep_low_t = e, t
            if e >= p:
                episodes.append((ep_start, ep_low_t, t, ep_low / p - 1.0))
                in_dd = False
    if in_dd:
        episodes.append((ep_start, ep_low_t, None, ep_low / peak_idx.iloc[-1] - 1.0))
    episodes.sort(key=lambda x: x[3])
    print("  worst episodes (start → trough → recovered, depth):")
    for start, low_t, rec, depth in episodes[:3]:
        rec_s = rec.date() if rec is not None else "NOT RECOVERED"
        print(f"    {start.date()} → {low_t.date()} → {rec_s}   {depth * 100:+.1f}%")


def trade_quality(trades: pd.DataFrame) -> None:
    if trades is None:
        print("\n(no sliding_oos_trades.csv — re-run the walk-forward with the "
              "current code to get per-trade diagnostics)")
        return
    _section(f"TRADE QUALITY — {len(trades):,} test trades")
    r = trades["r_mult"].astype(float)
    print(f"  win rate {(r > 0).mean() * 100:.1f}%   avg R {r.mean():+.3f}   "
          f"median R {r.median():+.3f}   best {r.max():+.2f}   worst {r.min():+.2f}")

    print("\n  by exit reason:")
    g = trades.groupby("exit_reason").agg(n=("pnl", "size"), total_pnl=("pnl", "sum"),
                                          avg_r=("r_mult", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\n  by bracket choice (SL mult × TP R):")
    g = trades.groupby(["sl_atr_mult", "tp_r_bracket"]).agg(
        n=("pnl", "size"), total_pnl=("pnl", "sum"), avg_r=("r_mult", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\n  by UTC entry session:")
    h = trades["entry_time"].dt.hour
    session = pd.cut(h, bins=[-1, 6, 12, 15, 21, 24],
                     labels=["asia(0-6)", "london(7-12)", "overlap(13-15)",
                             "newyork(16-21)", "late(22-23)"])
    g = trades.groupby(session, observed=True).agg(
        n=("pnl", "size"), total_pnl=("pnl", "sum"), avg_r=("r_mult", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:,.2f}"))

    if "fold" in trades.columns:
        print("\n  by fold:")
        g = trades.groupby("fold").agg(n=("pnl", "size"), total_pnl=("pnl", "sum"),
                                       avg_r=("r_mult", "mean"),
                                       win_pct=("r_mult", lambda s: (s > 0).mean() * 100))
        print(g.to_string(float_format=lambda v: f"{v:,.2f}"))


def cost_sensitivity(trades: pd.DataFrame, models_dir: Path,
                     spreads: list[float], slippages: list[float],
                     commissions: list[float]) -> None:
    if trades is None:
        return
    _section("COST SENSITIVITY (re-priced trades; see module docstring caveat)")

    # Original cost assumptions: prefer run_info.json, fall back to config.
    from config import CFG
    info_p = models_dir / "run_info.json"
    ri = json.loads(info_p.read_text()) if info_p.exists() else {}
    spread0 = float(ri.get("spread_price", CFG.spread_price))
    slip0 = float(ri.get("slippage_price", CFG.slippage_price))
    comm0 = float(ri.get("commission_per_trade", CFG.commission_per_trade))
    print(f"  run's costs: spread={spread0}  slippage={slip0}/side  commission={comm0}/trade")

    units = trades["units"].astype(float)
    pnl0 = trades["pnl"].astype(float)
    # Round trip pays the full spread + slippage on both sides (price units
    # per unit of position) plus the cash commission.
    gross = pnl0 + units * (spread0 + 2 * slip0) + comm0

    rows = []
    for sp in spreads:
        for sl in slippages:
            for cm in commissions:
                pnl = gross - units * (sp + 2 * sl) - cm
                wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
                rows.append({
                    "spread": sp, "slip/side": sl, "comm": cm,
                    "total_pnl": pnl.sum(),
                    "profit_factor": wins / losses if losses > 0 else np.inf,
                    "win_rate_pct": (pnl > 0).mean() * 100,
                    "avg_pnl/trade": pnl.mean(),
                })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    be = out[out["total_pnl"] <= 0]
    if be.empty:
        print("\n  edge survives every scenario in the grid.")
    else:
        first = be.iloc[0]
        print(f"\n  edge breaks even near: spread={first['spread']}, "
              f"slippage={first['slip/side']}/side, commission={first['comm']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward result diagnostics.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--spreads", default="0.20,0.30,0.45",
                        help="comma list of full-spread scenarios (price units)")
    parser.add_argument("--slippages", default="0.02,0.05",
                        help="comma list of per-side slippage scenarios")
    parser.add_argument("--commissions", default="0.0,0.7",
                        help="comma list of cash-per-trade commission scenarios")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    summary, equity, trades = _load(models_dir)

    per_fold(summary)
    drawdown_timeline(equity)
    trade_quality(trades)
    cost_sensitivity(
        trades, models_dir,
        spreads=[float(x) for x in args.spreads.split(",")],
        slippages=[float(x) for x in args.slippages.split(",")],
        commissions=[float(x) for x in args.commissions.split(",")],
    )
    print()


if __name__ == "__main__":
    main()
