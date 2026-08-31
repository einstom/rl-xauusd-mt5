"""Central configuration for the XAUUSD RL bracket-trading pipeline.

Baseline architecture: ZiadFrancis/Reinforcement_Trading_Part_2, extended with
a MetaTrader 5 bridge (mt5_data.py / mt5_execution.py / live_trading.py).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# ── Timeframe helpers ────────────────────────────────────────────────────────

# Map every common notation (MT4/MT5, shorthand, pandas) to the canonical
# pandas resample rule.  Add rows here to support new timeframes.
_TF_TO_PANDAS: dict[str, str] = {
    # MT4/MT5     shorthand    pandas rule
    "M1":  "1min",  "1m":  "1min",  "1min":  "1min",
    "M5":  "5min",  "5m":  "5min",  "5min":  "5min",
    "M15": "15min", "15m": "15min", "15min": "15min",
    "M30": "30min", "30m": "30min", "30min": "30min",
    "H1":  "1h",    "1h":  "1h",   "1H":    "1h",
    "H4":  "4h",    "4h":  "4h",   "4H":    "4h",
    "D1":  "1D",    "1d":  "1D",   "1D":    "1D",
}

# Approximate trading bars per year for XAUUSD.
# Basis: 23 trading hours/day × 261 trading days/year
# (XAUUSD trades ~Sun 22:00 UTC to Fri 22:00 UTC, minus ~1h daily maintenance)
_BARS_PER_YEAR: dict[str, int] = {
    "1min":  360_180,   # 23 × 60 × 261
    "5min":   72_036,   # 23 × 12 × 261
    "15min":  24_012,   # 23 × 4  × 261
    "30min":  12_006,   # 23 × 2  × 261
    "1h":      6_003,   # 23 × 1  × 261
    "4h":      1_501,   # 23/4    × 261  (≈ 6 bars/day)
    "1D":        261,   # 1       × 261
}


@dataclass
class ProjectConfig:
    # ── Data source ──────────────────────────────────────────────────────────
    # M1 CSV produced either by an MT4/MT5 export or by mt5_data.py
    # (`python mt5_data.py download`).  Column layout: Time (EET), OHLCV.
    csv_path: Path = Path("data/XAUUSD_M1.csv")
    time_col: str = "Time (EET)"

    # Many brokers call this EET but actually follow EET/EEST server time.
    # Europe/Helsinki is a practical EET/EEST timezone choice.
    source_tz: str = "Europe/Helsinki"

    # MT4/MT5-style M1 exports timestamp each row at candle OPEN.  Internally
    # the project indexes bars by CLOSE time so a decision timestamp means
    # "this candle is complete and known" — no lookahead by construction.
    timestamp_is_bar_open: bool = True

    # Execution stays at M1 for realistic intrabar TP/SL fill simulation.
    # The RL agent observes and acts on the decision timeframe (H1).
    execution_timeframe: str = "M1"
    decision_timeframe: str = "H1"

    # Optional date narrowing for smoke tests (None = full dataset).
    max_days_for_demo: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # ── Validation scheme ────────────────────────────────────────────────────
    # (a) Single chronological split → data_loader.split_train_val_test()
    # (b) Sliding-window walk-forward → data_loader.make_sliding_folds()
    # Both seal the LAST test_frac of bars as the final holdout that only
    # final_holdout_eval.py may reveal, exactly once, after the model freeze.
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1        # sealed final holdout (== 1 - train - val)

    # Sliding-window walk-forward: train 5y → select on next 6m (val) → judge
    # TRUE out-of-sample on the following 6m (test) → slide 6m and repeat.
    # Stitching every fold's test window yields one continuous OOS track record.
    sliding_train_years: float = 5.0
    sliding_val_months: int = 6
    sliding_test_months: int = 6
    sliding_step_months: int = 6

    # ── Walk-forward deployment gate ─────────────────────────────────────────
    # The final fold is promoted to the production slot (models/) ONLY if the
    # out-of-sample folds are consistently good; otherwise a NO_DEPLOY marker
    # is written.  The gate only ever PREVENTS a bad deploy.
    min_consistent_folds: int = 4              # folds needing return>0 AND PF>gate_min_profit_factor
    gate_min_profit_factor: float = 1.0
    gate_worst_fold_min_pf: float = 0.9
    gate_require_mean_sharpe_positive: bool = True

    # Embargo removes bars on each side of every split boundary.  EMA-200 (the
    # longest lookback) retains ~37% of a bar's weight after 200 steps, so a
    # 200-bar embargo makes cross-boundary leakage negligible.
    split_embargo_bars: int = 200

    # Indicators/features.
    atr_period: int = 14
    rsi_period: int = 14
    warmup_bars: int = 250

    # ── Bracket action space ─────────────────────────────────────────────────
    # direction ∈ {hold/flat, buy, sell}; SL = ATR multiple; TP = R-multiple of
    # the SL risk.  The agent picks direction + bracket shape; lot size is
    # fixed-fractional (risk_fraction of equity), never an agent choice.
    sl_atr_multipliers: Tuple[float, ...] = (1.0, 1.5)
    tp_r_multipliers: Tuple[float, ...] = (1.5, 2.0)

    # Backtest/account model.
    initial_equity: float = 10_000.0
    risk_fraction: float = 0.005  # 0.5% equity risked per trade.
    spread_price: float = 0.20    # XAUUSD price units; adjust to your broker.
    slippage_price: float = 0.02  # XAUUSD price units per side.
    commission_per_trade: float = 0.01

    # ── Reward shaping (env_bracket.py, the three terms) ─────────────────────
    # Term 1: realized PnL normalized by the per-trade risk budget (±1R at TP/SL).
    # Term 2: mark-to-market nudge while a trade floats (weight below).
    # Term 3: per-bar time penalty while a trade is held.
    reward_mtm_weight: float = 0.01
    holding_penalty: float = 0.000002   # reference repo used 2e-5; both "tiny"

    # ── PPO regularisation (generalisation-first preset) ─────────────────────
    # The thin XAUUSD signal lets PPO memorise the train period if it
    # over-optimises, so this preset trains "softer" — more exploration, fewer
    # passes per rollout, a KL brake, a decaying LR, and explicit weight decay —
    # and leans on the best-checkpoint selector to stop before the late collapse.
    ppo_learning_rate: float = 6e-5
    ppo_lr_schedule: str = "linear"        # "linear" decays LR → 0; "constant" holds it
    ppo_ent_coef: float = 0.03
    ppo_n_epochs: int = 5
    ppo_clip_range: float = 0.1
    ppo_target_kl: Optional[float] = 0.025
    ppo_weight_decay: float = 1e-5
    ppo_net_arch: Tuple[int, ...] = (128, 64)

    # ── MetaTrader 5 bridge ──────────────────────────────────────────────────
    mt5_symbol: str = "XAUUSD"
    mt5_magic: int = 260831            # tags this bot's orders/positions
    mt5_deviation_points: int = 30     # max slippage accepted by order_send
    mt5_order_comment: str = "rl-bracket"
    # Terminal/login left as None → attach to the already-running, logged-in
    # terminal.  Fill in for headless starts (or use environment variables
    # MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_TERMINAL_PATH).
    mt5_terminal_path: Optional[str] = None
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    # History fetched at bot start for indicator warmup (must comfortably cover
    # warmup_bars + EMA-200 lookback at H1 → 600 H1 bars ≈ 5 weeks).
    live_warmup_h1_bars: int = 900

    # ── Derived / read-only properties ──────────────────────────────────────

    @property
    def pandas_tf(self) -> str:
        """Canonical pandas resample rule for decision_timeframe."""
        rule = _TF_TO_PANDAS.get(self.decision_timeframe)
        if rule is None:
            raise ValueError(
                f"Unknown decision_timeframe '{self.decision_timeframe}'. "
                f"Recognised values: {sorted(_TF_TO_PANDAS.keys())}"
            )
        return rule

    @property
    def pandas_execution_tf(self) -> str:
        """Canonical pandas rule for execution_timeframe."""
        rule = _TF_TO_PANDAS.get(self.execution_timeframe)
        if rule is None:
            raise ValueError(
                f"Unknown execution_timeframe '{self.execution_timeframe}'. "
                f"Recognised values: {sorted(_TF_TO_PANDAS.keys())}"
            )
        return rule

    @property
    def periods_per_year(self) -> int:
        """Trading bars per year at decision_timeframe — used for annualising
        Sharpe, Sortino, Calmar."""
        return _BARS_PER_YEAR.get(self.pandas_tf, 6_003)


CFG = ProjectConfig()
