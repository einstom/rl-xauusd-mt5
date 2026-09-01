# rl-xauusd-mt5 — project context for Claude

RL bracket-trading bot for XAUUSD. Decisions on H1, intrabar TP/SL simulated
on M1, PPO (stable-baselines3) with sliding walk-forward validation, direct
MetaTrader 5 execution. Owner's workflow: **research (train/backtest) on this
macOS machine, live trading on a Windows box**. Baseline architecture:
ZiadFrancis/Reinforcement_Trading_Part_2. Full run guide: README.md.

## Commands

```bash
source .venv/bin/activate                 # ALWAYS activate first ((base) conda lacks deps)
python features.py                        # data load + 25-feature collinearity audit
python train_ppo.py --mode single --timesteps 120000 --n-envs 4 --device cpu   # smoke
caffeinate -i python train_ppo.py --mode sliding --timesteps 1000000 --n-envs 4 --device cpu
python final_holdout_eval.py              # sealed holdout — run ONCE per frozen model
python live_trading.py --dry-run          # needs MT5 access (Windows or gateway)
```

Dataset: `data/XAUUSD_M1.csv` or `.csv.gz` (IC Markets M1 export; loader reads
both formats + gz, path resolves against the project dir). Regenerate with
`mql5/ExportHistoryCSV.mq5` inside the MT5 terminal.

## Architecture (one line each)

- `config.py` — every knob: timeframes, splits, brackets, risk, PPO regularisation, MT5.
- `data_loader.py` — CSV/gz loading, EET/EEST tz-safe close-time indexing, resampling, embargoed splits, sliding folds.
- `features.py` — exactly 25 stationary features; audit helper.
- `env_bracket.py` — gym env; 3-term reward (realized-PnL/risk, 0.01 MTM nudge, holding penalty).
- `train_ppo.py` — PPO training, consistency checkpoint selection, sliding walk-forward, deployment gate.
- `evaluate.py`, `model_artifacts.py` — metrics; artifact resolution.
- `final_holdout_eval.py` — one-shot sealed OOS eval.
- `mt5_data.py`, `mt5_execution.py`, `live_trading.py` — MT5 bridge + live H1 loop.
- `mt5_compat.py`, `mt5_gateway.py` — HTTP gateway so non-Windows Python can reach MT5 (research needs none of this).

## Hard rules — do not relax these

1. **Never feed absolute prices** (O/H/L/C, raw EMA/BB levels) to the agent.
   Observation = the 25 audited features + 6 position-state features, nothing else.
2. **Keep the feature count at 25 and re-run `python features.py` after any
   feature change**; drop any pair with |r| ≥ 0.96.
3. **The sealed holdout (last `test_frac`) is revealed only by
   `final_holdout_eval.py`, once per frozen model.** Never tune against it.
4. **The walk-forward gate decides deployability** (`gate_passed` in
   `models/run_info.json`). A failed gate is a valid result — never weaken the
   gate thresholds to make a model pass.
5. **Only one training run at a time** — they share `models/`. Check
   `pgrep -fl train_ppo` before starting; a second run corrupts artifacts.
6. Position sizing is fixed-fractional (`risk_fraction`); the agent only
   chooses direction and bracket shape. Keep it that way.
7. The `MetaTrader5` pip package imports only on Windows. On macOS the
   pipeline runs data-from-CSV; live/download needs the gateway or Windows.
8. Timestamps: bars indexed by CLOSE time in `Europe/Helsinki` (broker
   EET/EEST). Any new data source must be converted to that convention.

## Results to look at after a sliding run

`models/sliding_walk_forward_summary.csv` (per-fold TEST metrics),
`models/sliding_oos_equity.csv` (stitched OOS equity), gate verdict in the
console + `models/run_info.json` / `NO_DEPLOY.txt`.

## Deploying to Windows (production)

Copy the repo + `models/run_info.json` + `models/best_model/*` to the Windows
box with the logged-in terminal → `pip install -r requirements.txt` →
`python live_trading.py --dry-run` → demo account first. Live refuses a model
whose gate failed (`--force` for demo experiments only).
