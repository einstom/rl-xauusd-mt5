"""Live H1 decision loop: MT5 bars → 25 features → PPO policy → bracket order.

Mirrors env_bracket.BracketTradingEnv exactly:
  * observation = 25 stationary market features (latest CLOSED H1 bar) + 6
    position-state features, normalized with the SAME VecNormalize statistics
    the checkpoint was selected under;
  * action = MultiDiscrete [direction, sl_bucket, tp_bucket];
  * SL = chosen ATR multiple, TP = chosen R-multiple of the SL risk, lots =
    fixed-fractional (CFG.risk_fraction of account equity).
TP/SL live server-side inside the bracket order, so intrabar hits are handled
by the broker just as M1 simulation handled them in the backtest.

Run on the Windows machine hosting the logged-in MT5 terminal:
    python live_trading.py --dry-run        # decide + log, never send orders
    python live_trading.py                  # trade (demo account first!)
    python live_trading.py --once           # one decision, then exit
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from features import N_FEATURES, prepare_feature_frame
from model_artifacts import load_run_info, resolve_project_path, resolve_sb3_model_path
import mt5_data
import mt5_execution as ex

STATE_PATH = Path("live_state.json")
N_POS_FEATURES = 6  # must match BracketTradingEnv.n_pos_features


# ── Model loading ────────────────────────────────────────────────────────────

def load_policy(models_dir: str = "models", force: bool = False):
    """Load the deployable checkpoint + its VecNormalize snapshot."""
    from stable_baselines3 import PPO

    _, run_info = load_run_info(models_dir)
    if run_info.get("gate_passed") is False and not force:
        raise SystemExit(
            "run_info.json says gate_passed=false — the walk-forward consistency "
            "gate did NOT approve this model for deployment. Re-train, or pass "
            "--force if you accept the risk on a demo account."
        )
    model_path = resolve_sb3_model_path(
        run_info.get("best_model_path", run_info["model_path"]), ".")
    vecnorm_path = resolve_project_path(
        run_info.get("best_model_vecnorm_path", run_info["vecnorm_path"]), ".")

    model = PPO.load(str(model_path))
    with open(vecnorm_path, "rb") as f:
        vecnorm = pickle.load(f)
    vecnorm.training = False
    vecnorm.norm_reward = False

    n_obs = int(np.prod(model.observation_space.shape))
    expected = N_FEATURES + N_POS_FEATURES
    if n_obs != expected:
        raise SystemExit(f"Model expects {n_obs} obs dims, pipeline builds {expected} — "
                         f"model and code are out of sync.")
    print(f"Policy   : {model_path}")
    print(f"VecNorm  : {vecnorm_path}")
    sl_mults = run_info.get("sl_atr_multipliers", list(CFG.sl_atr_multipliers))
    tp_rs = run_info.get("tp_r_multipliers", list(CFG.tp_r_multipliers))
    print(f"Brackets : SL {sl_mults} × ATR, TP {tp_rs} R")
    return model, vecnorm, sl_mults, tp_rs


# ── Local bracket state (risk_cash / tp_r are not stored by MT5) ─────────────

def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _state_for_position(position, state: dict) -> dict:
    """Bracket metadata for the open position; reconstructed from MT5 fields
    after a restart when the local file is missing."""
    key = str(position.ticket)
    if key in state:
        return state[key]

    si = ex.ensure_symbol(position.symbol)
    entry = position.price_open
    sl_distance = abs(entry - position.sl) if position.sl else 0.0
    tp_r = (abs(position.tp - entry) / sl_distance) if (position.tp and sl_distance) else 1.5
    risk_cash = position.volume * sl_distance * ex.money_per_price_unit_per_lot(si)
    entry_close = mt5_data.server_epoch_to_tz_index([position.time])[0] \
        .ceil("1h").isoformat()
    rec = {"risk_cash": risk_cash, "tp_r": tp_r, "sl_distance": sl_distance,
           "entry_bar_close": entry_close, "reconstructed": True}
    state[key] = rec
    _save_state(state)
    print(f"[state] reconstructed bracket metadata for ticket {position.ticket}: {rec}")
    return rec


# ── Observation ──────────────────────────────────────────────────────────────

def build_market_row(symbol: str):
    """Feature row of the latest CLOSED H1 bar (plus raw atr for sizing)."""
    bars = mt5_data.fetch_recent_bars(symbol, "H1", CFG.live_warmup_h1_bars)
    if len(bars) < CFG.warmup_bars + 250:
        raise RuntimeError(f"Only {len(bars)} H1 bars available — raise MT5 "
                           f"'Max bars in chart' or lower live_warmup_h1_bars.")
    feat, feature_cols = prepare_feature_frame(
        bars, warmup_bars=CFG.warmup_bars,
        atr_period=CFG.atr_period, rsi_period=CFG.rsi_period)
    return feat.iloc[-1], feature_cols


def position_state_features(position, row, state: dict) -> np.ndarray:
    """Replicates BracketTradingEnv._position_state_features from live data."""
    mt5 = mt5_data.mt5

    if position is None:
        return np.zeros(N_POS_FEATURES, dtype=np.float32)

    rec = _state_for_position(position, state)
    direction = 1 if position.type == mt5.POSITION_TYPE_BUY else -1
    close = float(row["Close"])
    atr = max(float(row["atr"]), 1e-12)
    risk_cash = max(float(rec["risk_cash"]), 1e-12)

    si = ex.ensure_symbol(position.symbol)
    units = position.volume * ex.money_per_price_unit_per_lot(si)
    unrealized = (close - position.price_open) * units * direction
    unrealized_r = unrealized / risk_cash
    dist_tp_atr = ((position.tp - close) * direction) / atr if position.tp else 0.0
    dist_sl_atr = ((close - position.sl) * direction) / atr if position.sl else 0.0

    entry_close = pd.Timestamp(rec["entry_bar_close"])
    bars_in_trade = max(int((row.name - entry_close) / pd.Timedelta(hours=1)), 0)

    return np.array([
        direction,
        unrealized_r,
        min(bars_in_trade / 100.0, 10.0),
        dist_tp_atr,
        dist_sl_atr,
        float(rec["tp_r"]),
    ], dtype=np.float32)


def build_observation(row, feature_cols, position, state: dict, vecnorm) -> np.ndarray:
    market = row[feature_cols].astype(float).to_numpy(dtype=np.float32)
    pos = position_state_features(position, row, state)
    obs = np.concatenate([market, pos]).astype(np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return vecnorm.normalize_obs(obs)


# ── One decision ─────────────────────────────────────────────────────────────

def decide_and_act(model, vecnorm, sl_mults, tp_rs, dry_run: bool) -> None:
    mt5 = mt5_data.mt5

    symbol = CFG.mt5_symbol
    state = _load_state()

    # 1. Sync: if the bracket resolved (TP/SL hit) the position is gone —
    #    drop stale local state so a fresh entry starts clean.
    position = ex.get_open_position(symbol)
    open_tickets = {str(position.ticket)} if position else set()
    stale = [k for k in state if k not in open_tickets]
    for k in stale:
        print(f"[state] position {k} closed since last decision (TP/SL or manual)")
        del state[k]
    if stale:
        _save_state(state)

    # 2. Observe the latest closed H1 bar.
    row, feature_cols = build_market_row(symbol)
    obs = build_observation(row, feature_cols, position, state, vecnorm)
    action, _ = model.predict(obs, deterministic=True)
    direction_raw, sl_idx, tp_idx = int(action[0]), int(action[1]), int(action[2])
    desired = {0: 0, 1: 1, 2: -1}[direction_raw]

    atr = float(row["atr"])
    print(f"[{row.name}] close={row['Close']:.2f} atr={atr:.2f} "
          f"action=(dir={desired:+d}, sl={sl_mults[sl_idx]}xATR, tp={tp_rs[tp_idx]}R) "
          f"position={'none' if position is None else position.ticket}")

    current_dir = 0
    if position is not None:
        current_dir = 1 if position.type == mt5.POSITION_TYPE_BUY else -1

    # 3. Act — same semantics as BracketTradingEnv.step().
    if desired == current_dir and desired != 0:
        print("        hold: keeping the open bracket untouched")
        return
    if desired == 0 and position is None:
        return

    if dry_run:
        if position is not None and desired != current_dir:
            print(f"        DRY-RUN: would close position {position.ticket}")
        if desired != 0:
            print(f"        DRY-RUN: would open {'BUY' if desired > 0 else 'SELL'} "
                  f"SL={sl_mults[sl_idx]}xATR={sl_mults[sl_idx]*atr:.2f} "
                  f"TP={tp_rs[tp_idx]}R")
        return

    if position is not None and desired != current_dir:
        ex.close_position(position)
        state.pop(str(position.ticket), None)
        _save_state(state)

    if desired != 0:
        risk_cash = ex.account_equity() * CFG.risk_fraction
        plan = ex.build_bracket_plan(
            symbol, desired, atr=atr,
            sl_atr_mult=float(sl_mults[sl_idx]),
            tp_r=float(tp_rs[tp_idx]),
            risk_cash=risk_cash,
        )
        if plan is None:
            return
        ex.place_bracket_order(plan)
        new_pos = ex.get_open_position(symbol)
        if new_pos is not None:
            state[str(new_pos.ticket)] = {
                "risk_cash": plan.risk_cash,
                "tp_r": plan.tp_r,
                "sl_distance": plan.sl_distance,
                "entry_bar_close": row.name.isoformat(),
            }
            _save_state(state)


# ── Loop ─────────────────────────────────────────────────────────────────────

def _seconds_until_next_h1_close(grace: int = 10) -> float:
    now = pd.Timestamp.now(tz=CFG.source_tz)
    next_close = (now.floor("1h") + pd.Timedelta(hours=1))
    return max((next_close - now).total_seconds() + grace, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live RL bracket trading on MT5.")
    parser.add_argument("--dry-run", action="store_true",
                        help="log decisions without sending any order")
    parser.add_argument("--once", action="store_true",
                        help="one decision on the latest closed bar, then exit")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--force", action="store_true",
                        help="run even if the walk-forward gate failed (demo only!)")
    args = parser.parse_args()

    model, vecnorm, sl_mults, tp_rs = load_policy(args.models_dir, force=args.force)

    with mt5_data.Mt5Session():
        ex.ensure_symbol(CFG.mt5_symbol)
        if args.once:
            decide_and_act(model, vecnorm, sl_mults, tp_rs, args.dry_run)
            return

        print(f"Entering H1 loop for {CFG.mt5_symbol} "
              f"({'DRY-RUN' if args.dry_run else 'LIVE'}). Ctrl-C to stop.")
        last_bar = None
        while True:
            try:
                bar_time = mt5_data.latest_closed_bar_time(CFG.mt5_symbol, "H1")
                if bar_time != last_bar:
                    decide_and_act(model, vecnorm, sl_mults, tp_rs, args.dry_run)
                    last_bar = bar_time
            except KeyboardInterrupt:
                print("Stopped by user.")
                return
            except Exception as e:  # keep the loop alive over transient errors
                print(f"[warn] decision failed: {e!r} — retrying next bar")
            time.sleep(_seconds_until_next_h1_close())


if __name__ == "__main__":
    main()
