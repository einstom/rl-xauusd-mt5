"""PPO training with anti-overfitting checkpoint selection and walk-forward.

Two entry points:
  * train()                       — one train/val split (one fold's worth of work)
  * train_sliding_walk_forward()  — sliding-window walk-forward (train 5y →
        select checkpoint on next 6m val → judge TRUE out-of-sample on the
        following 6m test → slide 6m), stitched into one continuous OOS
        equity track record + a deployment gate.

Checkpoint selection ("best of the worst", _ConsistencyEvalCallback):
    every eval_freq steps the current policy is rolled out on BOTH the tail of
    the training data and the validation window; each leg is scored
    q = reward − dd_penalty · max_drawdown_pct and the checkpoint's score is
    min(q_train, q_val).  The saved "best" model maximises that minimum — a
    model that is good on only one side of the split boundary never wins.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import CFG
from data_loader import (
    load_mt_ohlcv_csv,
    make_sliding_folds,
    resample_ohlcv,
    split_train_val_test,
)
from env_bracket import BracketTradingEnv
from evaluate import drawdown, full_report
from features import prepare_feature_frame
from model_artifacts import load_run_info


# ── Pickle-safe factory for SubprocVecEnv ────────────────────────────────────
# Must be at module level (not a lambda) so subprocesses can import it.
def _spawn_train_env(decision_df, m1_df, feature_cols, episode_steps):
    return build_env(decision_df, m1_df, feature_cols,
                     randomize_start=True, episode_steps=episode_steps)


def _slice_m1_for_decision_window(m1_df, decision_df):
    if decision_df.empty:
        return m1_df.iloc[0:0].copy()
    start = decision_df.index.min()
    end = decision_df.index.max()
    return m1_df.loc[(m1_df.index > start) & (m1_df.index <= end)].copy()


def _load_decision_features():
    """Load M1, resample to the decision timeframe, build the causal feature
    frame.  Shared by every mode so all of them see byte-identical bars."""
    m1 = load_mt_ohlcv_csv(
        CFG.csv_path,
        time_col=CFG.time_col,
        source_tz=CFG.source_tz,
        timestamp_is_bar_open=CFG.timestamp_is_bar_open,
        bar_duration=CFG.pandas_execution_tf,
        start_date=CFG.start_date,
        end_date=CFG.end_date,
        max_days_for_demo=CFG.max_days_for_demo,
    )
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, feature_cols = prepare_feature_frame(
        decision,
        warmup_bars=CFG.warmup_bars,
        atr_period=CFG.atr_period,
        rsi_period=CFG.rsi_period,
    )
    return m1, feat, feature_cols


def load_datasets():
    """Single chronological train/val/test split."""
    m1, feat, feature_cols = _load_decision_features()
    train_feat, val_feat, test_feat = split_train_val_test(
        feat,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        embargo_bars=CFG.split_embargo_bars,
    )
    return m1, feature_cols, train_feat, val_feat, test_feat


def build_env(decision_df, m1_df, feature_cols, randomize_start: bool = False,
              episode_steps: int | None = None):
    env = BracketTradingEnv(
        decision_df,
        m1_df,
        feature_cols,
        sl_atr_multipliers=CFG.sl_atr_multipliers,
        tp_r_multipliers=CFG.tp_r_multipliers,
        initial_equity=CFG.initial_equity,
        risk_fraction=CFG.risk_fraction,
        spread_price=CFG.spread_price,
        slippage_price=CFG.slippage_price,
        commission_per_trade=CFG.commission_per_trade,
        holding_penalty=CFG.holding_penalty,
        reward_mtm_weight=CFG.reward_mtm_weight,
        randomize_start=randomize_start,
        max_episode_steps=episode_steps,
    )
    return Monitor(env)


class _CaptureDoneWrapper(gym.Wrapper):
    """Saves equity_curve and trade_log the instant the episode ends.

    DummyVecEnv calls env.reset() automatically when step() returns done=True,
    which wipes BracketTradingEnv.trades before we can read it.  This wrapper
    intercepts the done signal and stashes the results before the auto-reset.
    """

    def __init__(self, env):
        super().__init__(env)
        self.saved_equity: pd.DataFrame | None = None
        self.saved_trades: pd.DataFrame | None = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            inner = self.env
            while hasattr(inner, "env"):
                inner = inner.env
            self.saved_equity = inner.equity_curve()
            self.saved_trades = inner.trade_log()
        return obs, reward, terminated, truncated, info


class _SyncVecNormCallback(BaseCallback):
    """Copies running obs/ret stats from the train VecNormalize to every eval
    VecNormalize before each checkpoint evaluation."""

    def __init__(self, train_venv: VecNormalize,
                 eval_venvs: list[VecNormalize], eval_freq: int):
        super().__init__()
        self.train_venv = train_venv
        self.eval_venvs = eval_venvs
        self.eval_freq = eval_freq
        self._last_sync = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_sync >= self.eval_freq:
            for venv in self.eval_venvs:
                venv.obs_rms = deepcopy(self.train_venv.obs_rms)
                venv.ret_rms = deepcopy(self.train_venv.ret_rms)
            self._last_sync = self.num_timesteps
        return True


class _ConsistencyEvalCallback(BaseCallback):
    """Saves the checkpoint that maximises a drawdown-penalised consistency score.

    Standard EvalCallback picks the highest val reward in isolation.  That can
    select a model that got lucky on the val period while being terrible on
    training data — a sign it never really learned.

    This callback evaluates on BOTH a 'train eval' slice (the last ~len(val)
    bars of training data) and the val set.  For each leg it computes a
    risk-adjusted quality, then scores the checkpoint by the WEAKER leg:

        q_leg  = reward_leg − dd_penalty · max_drawdown_pct_leg
        score  = min(q_train, q_val)

    → rewards models that are simultaneously profitable AND low-drawdown on
      both sides of the split boundary (the "best of the worst")
    → anchored by the weaker leg, so a large train/val gap is penalised

    Do-nothing guard: an idle policy makes no trades → flat equity → ~0 DD,
    which would otherwise look 'perfectly stable' under the penalty.  A
    checkpoint is only eligible to become 'best' when BOTH legs are profitable
    (reward > 0) and place at least `min_trades` trades.

    dd_penalty is in reward-units per 1% of max drawdown.  Calibrate it against
    eval_logs/consistency_evals.csv: too large and a flat, barely-trading model
    wins; too small and it has no effect.
    """

    def __init__(
        self,
        train_eval_env: VecNormalize,
        val_env: VecNormalize,
        eval_freq: int,
        best_model_save_path: str,
        log_path: str | None = None,
        dd_penalty: float = 1.0,
        min_trades: int = 5,
        verbose: int = 1,
        train_venv: VecNormalize | None = None,
    ):
        super().__init__(verbose=verbose)
        self.train_eval_env = train_eval_env
        self.val_env = val_env
        # The live training VecNormalize — snapshotted next to best_model so the
        # saved checkpoint is later evaluated under the normalisation it was
        # selected with (the obs_rms drifts as training continues).
        self.train_venv = train_venv
        self.eval_freq = eval_freq
        self.best_model_save_path = Path(best_model_save_path)
        self.log_path = Path(log_path) if log_path else None
        self.dd_penalty = float(dd_penalty)
        self.min_trades = int(min_trades)
        self.best_score = -float("inf")
        self._last_eval = 0
        self._rows: list[dict] = []

    def _run_one_episode(self, venv: VecNormalize) -> tuple[float, float, int]:
        """One deterministic episode → (cumulative_reward, max_dd_pct, n_trades)."""
        from stable_baselines3.common.evaluation import evaluate_policy
        rewards, _ = evaluate_policy(
            self.model, venv,
            n_eval_episodes=1,
            deterministic=True,
            return_episode_rewards=True,
        )
        reward = float(rewards[0])

        # venv = VecNormalize → .venv = DummyVecEnv → .envs[0] = _CaptureDoneWrapper.
        capture = venv.venv.envs[0]
        eq = capture.saved_equity
        if eq is not None and not eq.empty and "equity" in eq:
            max_dd_pct = float(abs(drawdown(eq["equity"].astype(float)).min()) * 100.0)
            n_trades = len(capture.saved_trades) if capture.saved_trades is not None else 0
        else:
            max_dd_pct = 0.0
            n_trades = 0
        return reward, max_dd_pct, n_trades

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        train_r, train_dd, train_n = self._run_one_episode(self.train_eval_env)
        val_r, val_dd, val_n = self._run_one_episode(self.val_env)

        # Risk-adjusted quality per leg, then anchor on the weaker leg.
        q_train = train_r - self.dd_penalty * train_dd
        q_val = val_r - self.dd_penalty * val_dd
        score = min(q_train, q_val)
        gap = abs(train_r - val_r)

        # Do-nothing guard: both legs must be genuinely profitable and active.
        eligible = (
            train_r > 0 and val_r > 0
            and train_n >= self.min_trades and val_n >= self.min_trades
        )

        row = dict(timesteps=self.num_timesteps,
                   train_eval_r=round(train_r, 3),
                   val_r=round(val_r, 3),
                   train_dd_pct=round(train_dd, 3),
                   val_dd_pct=round(val_dd, 3),
                   q_train=round(q_train, 3),
                   q_val=round(q_val, 3),
                   score=round(score, 3),
                   gap=round(gap, 3),
                   eligible=eligible)
        self._rows.append(row)

        marker = ""
        if eligible and score > self.best_score:
            self.best_score = score
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_save_path / "best_model"))
            # Snapshot the obs/reward normalisation stats AS OF this checkpoint
            # (project invariant: VecNormalize travels with every saved model).
            if self.train_venv is not None:
                self.train_venv.save(str(self.best_model_save_path / "best_model_vecnorm.pkl"))
            marker = "  ← BEST"

        if self.verbose >= 1:
            flag = "" if eligible else "  (ineligible)"
            print(
                f"[{self.num_timesteps:>9,}]  "
                f"train={train_r:+7.2f}/dd{train_dd:4.1f}%  "
                f"val={val_r:+7.2f}/dd{val_dd:4.1f}%  "
                f"score={score:+8.2f}{flag}{marker}",
                flush=True,
            )

        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._rows).to_csv(
                self.log_path / "consistency_evals.csv", index=False
            )

        return True


def _linear_schedule(initial_value: float):
    """SB3 learning-rate schedule: decays linearly to 0 as training progresses.
    Shrinking late-training updates keeps an over-fitting tail from overrunning
    the best checkpoint."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def train(
    total_timesteps: int = 2_000_000,
    seed: int = 42,
    out_dir: str = "models",
    train_episode_steps: int = 2048,
    eval_freq: int = 50_000,
    dd_penalty: float = 1.0,
    n_envs: int = 1,
    device: str = "auto",
    datasets: tuple | None = None,
):
    """Train one PPO model on one train/val split; the test stays sealed."""
    import torch
    from stable_baselines3 import PPO

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if datasets is None:
        m1, feature_cols, train_feat, val_feat, test_feat = load_datasets()
    else:
        m1, feature_cols, train_feat, val_feat, test_feat = datasets
    train_m1 = _slice_m1_for_decision_window(m1, train_feat)
    val_m1 = _slice_m1_for_decision_window(m1, val_feat)

    # ── Training environments ────────────────────────────────────────────────
    if n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env_fns = [
            partial(_spawn_train_env, train_feat, train_m1, feature_cols, train_episode_steps)
            for _ in range(n_envs)
        ]
        start_method = "spawn" if sys.platform == "win32" else "forkserver"
        print(f"Training envs : {n_envs} parallel (SubprocVecEnv, {start_method})")
        train_env_raw = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        print("Training envs : 1 (DummyVecEnv — set n_envs=4 for ~4x speedup)")
        train_env_raw = DummyVecEnv([
            lambda: build_env(train_feat, train_m1, feature_cols,
                              randomize_start=True, episode_steps=train_episode_steps)
        ])

    train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Eval environments (deterministic, no reward norm) ────────────────────
    val_env_raw = DummyVecEnv([
        lambda: _CaptureDoneWrapper(
            build_env(val_feat, val_m1, feature_cols,
                      randomize_start=False, episode_steps=None))
    ])
    val_env = VecNormalize(val_env_raw, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, training=False)

    # Train-eval slice: the last ~len(val) bars of training data.  Same era as
    # val, other side of the boundary, same size → rewards on the same scale.
    train_eval_feat = train_feat.iloc[-len(val_feat):]
    train_eval_m1 = _slice_m1_for_decision_window(m1, train_eval_feat)
    train_eval_env_raw = DummyVecEnv([
        lambda: _CaptureDoneWrapper(
            build_env(train_eval_feat, train_eval_m1, feature_cols,
                      randomize_start=False, episode_steps=None))
    ])
    train_eval_env = VecNormalize(train_eval_env_raw, norm_obs=True, norm_reward=False,
                                  clip_obs=10.0, training=False)

    print(f"Train-eval slice : {len(train_eval_feat):,} bars  "
          f"{train_eval_feat.index.min().date()} to {train_eval_feat.index.max().date()}")
    print(f"Val              : {len(val_feat):,} bars  "
          f"{val_feat.index.min().date()} to {val_feat.index.max().date()}")

    sync_cb = _SyncVecNormCallback(
        train_env,
        eval_venvs=[train_eval_env, val_env],
        eval_freq=eval_freq,
    )
    eval_cb = _ConsistencyEvalCallback(
        train_eval_env=train_eval_env,
        val_env=val_env,
        eval_freq=eval_freq,
        best_model_save_path=str(Path(out_dir) / "best_model"),
        log_path=str(Path(out_dir) / "eval_logs"),
        dd_penalty=dd_penalty,
        verbose=1,
        train_venv=train_env,
    )

    effective_buffer = train_episode_steps * n_envs
    batch_size = min(1024 if device == "cuda" else 256, effective_buffer)

    if CFG.ppo_lr_schedule == "linear":
        learning_rate = _linear_schedule(CFG.ppo_learning_rate)
    else:
        learning_rate = CFG.ppo_learning_rate

    model = PPO(
        "MlpPolicy",
        train_env,
        device=device,
        verbose=1,
        seed=seed,
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=CFG.ppo_clip_range,
        ent_coef=CFG.ppo_ent_coef,
        vf_coef=0.5,
        n_steps=train_episode_steps,
        batch_size=batch_size,
        n_epochs=CFG.ppo_n_epochs,
        target_kl=CFG.ppo_target_kl,
        policy_kwargs={
            "net_arch": list(CFG.ppo_net_arch),
            "optimizer_kwargs": {"weight_decay": CFG.ppo_weight_decay},
        },
    )
    print(f"PPO batch_size : {batch_size}  (buffer {effective_buffer} transitions per update)")

    model.learn(total_timesteps=total_timesteps, callback=CallbackList([sync_cb, eval_cb]))

    # Human-readable slug: ppo_H1_sl1-1.5_tp1.5-2_2000k_seed42
    def _fmt(v: float) -> str:
        return str(v).rstrip("0").rstrip(".")
    sl_str = "-".join(_fmt(x) for x in CFG.sl_atr_multipliers)
    tp_str = "-".join(_fmt(x) for x in CFG.tp_r_multipliers)
    slug = (f"ppo_{CFG.decision_timeframe}"
            f"_sl{sl_str}_tp{tp_str}"
            f"_{total_timesteps // 1000}k_seed{seed}")

    model_path = Path(out_dir) / f"{slug}.zip"
    vecnorm_path = Path(out_dir) / f"{slug}_vecnorm.pkl"
    model.save(model_path)
    train_env.save(str(vecnorm_path))

    run_info = {
        "slug": slug,
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path),
        "decision_timeframe": CFG.decision_timeframe,
        "sl_atr_multipliers": list(CFG.sl_atr_multipliers),
        "tp_r_multipliers": list(CFG.tp_r_multipliers),
        "feature_count": len(feature_cols),
        "total_timesteps": total_timesteps,
        "train_episode_steps": train_episode_steps,
        "seed": seed,
        "dd_penalty": dd_penalty,
        "risk_fraction": CFG.risk_fraction,
        "spread_price": CFG.spread_price,
    }
    best_model_zip = Path(out_dir) / "best_model" / "best_model.zip"
    best_model_vecnorm = Path(out_dir) / "best_model" / "best_model_vecnorm.pkl"
    if best_model_zip.exists():
        run_info["best_model_path"] = str(best_model_zip)
        if best_model_vecnorm.exists():
            run_info["best_model_vecnorm_path"] = str(best_model_vecnorm)
    info_path = Path(out_dir) / "run_info.json"
    info_path.write_text(json.dumps(run_info, indent=2))

    print(f"Model      → {model_path}")
    print(f"VecNorm    → {vecnorm_path}")
    print(f"Run info   → {info_path}")
    print("Test split remains sealed. Use final_holdout_eval.py when the model is frozen.")
    return model, train_env


def _rollout_on_split(model, vecnorm_path, m1, feature_cols, decision_df):
    """Deterministic rollout of a model over one decision split.

    Returns (equity_df, trades_df, report_dict).  VecNormalize stats are loaded
    from disk and frozen so evaluation never updates them."""
    m1_sl = _slice_m1_for_decision_window(m1, decision_df)

    def _make():
        return _CaptureDoneWrapper(
            build_env(decision_df, m1_sl, feature_cols,
                      randomize_start=False, episode_steps=None)
        )

    raw = DummyVecEnv([_make])
    if vecnorm_path is not None and Path(vecnorm_path).exists():
        venv = VecNormalize.load(str(vecnorm_path), raw)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = raw

    obs = venv.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(action)

    capture = venv.venv.envs[0] if hasattr(venv, "venv") else venv.envs[0]
    eq = capture.saved_equity if capture.saved_equity is not None else pd.DataFrame()
    trades = capture.saved_trades if capture.saved_trades is not None else pd.DataFrame()
    rep = full_report(eq, trades, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)
    return eq, trades, rep["value"].to_dict()


def evaluate_on_split(model, vecnorm_path, m1, feature_cols, decision_df) -> dict:
    _eq, _trades, rep = _rollout_on_split(model, vecnorm_path, m1, feature_cols, decision_df)
    return rep


def _passes_consistency_gate(
    summary: pd.DataFrame,
    ret_col: str = "test_return_pct",
    pf_col: str = "test_profit_factor",
    sharpe_col: str = "test_sharpe",
) -> tuple[bool, list[str]]:
    """Decide whether the walk-forward folds are consistent enough to deploy.

    A strategy that only works on some folds has no robust edge, so the gate
    requires breadth (enough folds genuinely profitable), a floor (no single
    fold catastrophic), and a positive mean risk-adjusted result."""
    n = len(summary)
    ret = summary[ret_col]
    pf = summary[pf_col]
    sharpe = summary[sharpe_col]

    good = int(((ret > 0) & (pf > CFG.gate_min_profit_factor)).sum())
    worst_pf = float(pf.min())          # NaN folds (no trades) skipped by .min()
    mean_sharpe = float(sharpe.mean())

    c_count = good >= CFG.min_consistent_folds
    c_worst = worst_pf >= CFG.gate_worst_fold_min_pf
    c_sharpe = (mean_sharpe > 0) if CFG.gate_require_mean_sharpe_positive else True
    passed = bool(c_count and c_worst and c_sharpe)

    ok = lambda b: "OK  " if b else "FAIL"
    detail = [
        f"[{ok(c_count)}] folds with return>0 & PF>{CFG.gate_min_profit_factor:g}: "
        f"{good}/{n}  (need >= {CFG.min_consistent_folds})",
        f"[{ok(c_worst)}] worst-fold PF: {worst_pf:.2f}  "
        f"(need >= {CFG.gate_worst_fold_min_pf:g})",
    ]
    if CFG.gate_require_mean_sharpe_positive:
        detail.append(f"[{ok(c_sharpe)}] mean Sharpe: {mean_sharpe:+.2f}  (need > 0)")
    return passed, detail


def _load_fold_model(fold_dir: str):
    """Load a fold's deployable checkpoint: best_model if the consistency
    callback found an eligible one, else the final save."""
    from stable_baselines3 import PPO
    _, ri = load_run_info(fold_dir)
    model_path = ri.get("best_model_path", ri["model_path"])
    vecnorm_path = ri.get("best_model_vecnorm_path", ri["vecnorm_path"])
    return PPO.load(str(model_path)), str(vecnorm_path)


def _promote_fold_to_production(out_dir: str, fold_k: int, gate_passed: bool) -> None:
    """Copy the final fold's artifacts up to the production slot (out_dir) and
    write a production run_info.json.  gate_passed is recorded so the model is
    always available even on a failed gate, while downstream (live_trading.py)
    can still tell it was not approved."""
    src = Path(out_dir) / "sliding" / f"fold_{fold_k}"
    dst = Path(out_dir)
    _, fr = load_run_info(src)
    slug = fr["slug"]

    for name in (f"{slug}.zip", f"{slug}_vecnorm.pkl"):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    if (src / "best_model").exists():
        shutil.copytree(src / "best_model", dst / "best_model", dirs_exist_ok=True)
    src_log = src / "eval_logs" / "consistency_evals.csv"
    if src_log.exists():
        (dst / "eval_logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_log, dst / "eval_logs" / "consistency_evals.csv")

    prod = dict(fr)
    prod["model_path"] = str(dst / f"{slug}.zip")
    prod["vecnorm_path"] = str(dst / f"{slug}_vecnorm.pkl")
    bm = dst / "best_model" / "best_model.zip"
    bmv = dst / "best_model" / "best_model_vecnorm.pkl"
    if bm.exists():
        prod["best_model_path"] = str(bm)
        if bmv.exists():
            prod["best_model_vecnorm_path"] = str(bmv)
    prod["promoted_from_fold"] = fold_k
    prod["gate_passed"] = bool(gate_passed)
    (dst / "run_info.json").write_text(json.dumps(prod, indent=2))

    marker = Path(out_dir) / "NO_DEPLOY.txt"
    if gate_passed:
        marker.unlink(missing_ok=True)
        print(f"\n  ✓ GATE PASSED — fold {fold_k} promoted to production at {out_dir}/")
    else:
        marker.write_text(
            "Walk-forward consistency gate FAILED.\n"
            f"The best model of the final fold ({fold_k}) IS still saved and "
            "referenced by run_info.json (gate_passed=false) — but it is NOT "
            "approved for live deployment. Inspect the per-fold summary first.\n"
        )
        print(f"\n  ✗ GATE FAILED — model saved (gate_passed=false), NOT approved for deployment.")


def train_sliding_walk_forward(
    total_timesteps: int = 3_000_000,
    seed: int = 42,
    out_dir: str = "models",
    train_episode_steps: int = 2048,
    target_evals_per_fold: int = 20,
    dd_penalty: float = 0.8,
    n_envs: int = 4,
    device: str = "auto",
):
    """Sliding-window walk-forward that simulates periodic retraining.

    Each fold trains on CFG.sliding_train_years of data, selects its checkpoint
    on the next CFG.sliding_val_months (val), and is judged TRUE out-of-sample
    on the following CFG.sliding_test_months (test) — a window the model never
    saw during training OR checkpoint selection.  The whole window then slides
    CFG.sliding_step_months forward and repeats.

    Every fold's test window is stitched into one continuous out-of-sample
    equity curve (sliding_oos_equity.csv).  The gate is computed on the TEST
    metrics (the honest ones), and the final (most recent) fold is the
    deployable model.  Per-fold timesteps are fixed because every train window
    is the same calendar length.
    """
    m1, feat, feature_cols = _load_decision_features()
    folds = make_sliding_folds(
        feat,
        train_years=CFG.sliding_train_years,
        val_months=CFG.sliding_val_months,
        test_months=CFG.sliding_test_months,
        step_months=CFG.sliding_step_months,
        embargo_bars=CFG.split_embargo_bars,
    )
    n_folds = len(folds)
    if n_folds == 0:
        raise ValueError("No sliding folds produced — not enough data for the "
                         "chosen train/val/test window. Check CFG.sliding_* / dataset.")

    print("=" * 72)
    print(f"  SLIDING WALK-FORWARD — {n_folds} folds  "
          f"(train {CFG.sliding_train_years:g}y → val {CFG.sliding_val_months}m → "
          f"test {CFG.sliding_test_months}m, slide {CFG.sliding_step_months}m)")
    for k, (tr, va, te) in enumerate(folds, start=1):
        print(f"    fold {k:>2}: train {tr.index.min().date()}→{tr.index.max().date()}"
              f" | val {va.index.min().date()}→{va.index.max().date()}"
              f" | TEST {te.index.min().date()}→{te.index.max().date()} ({len(te):,} bars)")
    print("=" * 72)

    summary_rows: list[dict] = []
    test_equities: list[pd.DataFrame] = []
    test_trade_logs: list[pd.DataFrame] = []

    for k, (tr, va, te) in enumerate(folds, start=1):
        fold_dir = str(Path(out_dir) / "sliding" / f"fold_{k}")
        eval_freq_k = max(25_000, total_timesteps // target_evals_per_fold)
        print(f"\n── Fold {k}/{n_folds}"
              f"  | train {len(tr):,}  val {len(va):,}  test {len(te):,} bars"
              f"  | TEST {te.index.min().date()}→{te.index.max().date()}  → {fold_dir}")

        train(
            total_timesteps=total_timesteps,
            seed=seed,
            out_dir=fold_dir,
            train_episode_steps=train_episode_steps,
            eval_freq=eval_freq_k,
            dd_penalty=dd_penalty,
            n_envs=n_envs,
            device=device,
            datasets=(m1, feature_cols, tr, va, te),
        )

        # Evaluate the deployable (best) checkpoint on val (reference) and on
        # the untouched TEST window (the honest out-of-sample result).
        model, vp = _load_fold_model(fold_dir)
        val_rep = evaluate_on_split(model, vp, m1, feature_cols, va)
        test_eq, test_trades, test_rep = _rollout_on_split(model, vp, m1, feature_cols, te)
        test_equities.append(test_eq)
        if test_trades is not None and not test_trades.empty:
            test_trade_logs.append(test_trades)

        summary_rows.append({
            "fold": k,
            "train_start": tr.index.min().date(), "train_end": tr.index.max().date(),
            "test_start": te.index.min().date(), "test_end": te.index.max().date(),
            "val_return_pct": val_rep.get("total_return_pct"),
            "val_profit_factor": val_rep.get("profit_factor"),
            "test_return_pct": test_rep.get("total_return_pct"),
            "test_sharpe": test_rep.get("sharpe_like"),
            "test_profit_factor": test_rep.get("profit_factor"),
            "test_win_rate_pct": test_rep.get("win_rate_pct"),
            "test_max_dd_pct": test_rep.get("max_drawdown_pct"),
            "test_avg_r": test_rep.get("avg_r"),
            "test_n_trades": test_rep.get("n_trades"),
        })
        print(f"   fold {k} TEST: return={test_rep.get('total_return_pct'):+.1f}%  "
              f"PF={test_rep.get('profit_factor'):.2f}  "
              f"Sharpe={test_rep.get('sharpe_like'):+.2f}  "
              f"trades={test_rep.get('n_trades')}")

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(out_dir) / "sliding_walk_forward_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ── Stitch every fold's test window into one continuous OOS equity curve ──
    running = CFG.initial_equity
    parts = []
    for eq in test_equities:
        if eq is None or eq.empty or "equity" not in eq:
            continue
        s = eq["equity"].astype(float)
        scaled = s / CFG.initial_equity * running          # chain (compound) folds
        parts.append(scaled)
        running = float(scaled.iloc[-1])
    stitched = pd.concat(parts) if parts else pd.Series(dtype=float)
    stitched_df = stitched.to_frame("equity")
    stitched_path = Path(out_dir) / "sliding_oos_equity.csv"
    stitched_df.to_csv(stitched_path)

    all_trades = pd.concat(test_trade_logs, ignore_index=True) if test_trade_logs else pd.DataFrame()
    oos = full_report(stitched_df, all_trades, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()

    print("\n" + "=" * 72)
    print("  SLIDING WALK-FORWARD — PER-FOLD TEST (true out-of-sample)")
    show = ["fold", "test_start", "test_end", "test_return_pct", "test_sharpe",
            "test_profit_factor", "test_win_rate_pct", "test_max_dd_pct", "test_n_trades"]
    print(summary[show].to_string(index=False))
    print("\n  STITCHED out-of-sample track record (compounded across all test windows):")
    print(f"    total return : {oos.get('total_return_pct'):+.1f}%")
    print(f"    Sharpe-like  : {oos.get('sharpe_like'):+.2f}")
    print(f"    max drawdown : {oos.get('max_drawdown_pct'):+.1f}%")
    print(f"    profit factor: {oos.get('profit_factor'):.2f}   trades: {oos.get('n_trades')}")

    passed, detail = _passes_consistency_gate(summary)
    print("\n  Consistency gate (on test windows):")
    for line in detail:
        print(f"    {line}")
    _promote_fold_to_production(out_dir, n_folds, passed)

    print(f"\n  Per-fold summary → {summary_path}")
    print("=" * 72)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the XAUUSD RL bracket agent.")
    parser.add_argument("--mode", choices=["sliding", "single"], default="sliding",
                        help="sliding = walk-forward retrain simulation (default); "
                             "single = one chronological train/val split")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                        help="PPO env-steps per fold (sliding) or total (single)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dd-penalty", type=float, default=0.8,
                        help="reward-units subtracted per 1%% max drawdown in checkpoint selection")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    if args.mode == "sliding":
        train_sliding_walk_forward(
            total_timesteps=args.timesteps, seed=args.seed, out_dir=args.out_dir,
            dd_penalty=args.dd_penalty, n_envs=args.n_envs, device=args.device,
        )
    else:
        train(
            total_timesteps=args.timesteps, seed=args.seed, out_dir=args.out_dir,
            dd_penalty=args.dd_penalty, n_envs=args.n_envs, device=args.device,
        )
