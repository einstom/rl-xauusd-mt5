# RL Bracket-Trading Bot for XAUUSD with a Direct MetaTrader 5 Bridge

A Reinforcement-Learning trading pipeline for Gold (XAUUSD) that decides on
**H1**, simulates intrabar TP/SL hits on **M1**, and connects Python directly
to **MetaTrader 5** for both historical data and live bracket-order execution.

Baseline architecture: [ZiadFrancis/Reinforcement_Trading_Part_2](https://github.com/ZiadFrancis/Reinforcement_Trading_Part_2)
(feature set, environment, walk-forward and checkpoint-selection design),
extended here with the MT5 integration layer.

## Design rules (enforced in code)

| Rule | Where |
| --- | --- |
| H1 decisions, M1 intrabar TP/SL simulation (pessimistic: SL first if both touch) | `env_bracket.py` |
| No absolute prices in the observation — only ATR-normalized distances, ratios, bounded oscillators, sin/cos time, session flags | `features.py` |
| Exactly **25 non-collinear features**; `rsi_centered`, `roc5_atr`, `body_atr`, `ema50_slope5_atr` dropped by the \|r\| ≥ 0.96 audit | `features.py` (`python features.py` re-runs the audit) |
| Action = MultiDiscrete(direction ∈ {hold, buy, sell}, SL ∈ {1.0, 1.5}×ATR, TP ∈ {1.5, 2.0}×R) | `env_bracket.py`, `config.py` |
| Fixed-fractional sizing (0.5% equity risk); the agent never picks lot size | `env_bracket.py`, `mt5_execution.py` |
| Reward = realized-PnL/risk ±1R + 0.01·unrealized-R nudge − per-bar time penalty | `env_bracket.py` (`step()`, the "three terms" block) |
| Sliding walk-forward: train 5y → val 6m (checkpoint selection) → test 6m (true OOS) → slide 6m | `train_ppo.py`, `data_loader.py` |
| Checkpoint = best **minimum** of drawdown-penalised train-tail/val scores ("best of the worst"), with a do-nothing guard | `train_ppo.py` (`_ConsistencyEvalCallback`) |
| Sealed final holdout, revealed exactly once | `final_holdout_eval.py` |
| Broker EET/EEST time parsed via `Europe/Helsinki`, bars indexed by close time | `data_loader.py`, `mt5_data.py` |

## Repository layout

```
config.py            All knobs: timeframes, splits, brackets, risk, PPO, MT5.
data_loader.py       CSV loading, tz-safe indexing, resampling, temporal splits,
                     sliding walk-forward folds (embargoed).
features.py          The 25-feature observation set + collinearity audit.
env_bracket.py       Gymnasium env: H1 decisions, M1 fills, 3-term reward.
evaluate.py          Sharpe/Sortino/Calmar/PF/drawdown metrics.
train_ppo.py         PPO training, consistency checkpoint selection,
                     sliding walk-forward + deployment gate.
model_artifacts.py   run_info.json / model / VecNormalize resolution.
final_holdout_eval.py  One-shot sealed out-of-sample evaluation.
mt5_data.py          MT5 bridge: historical download → CSV, live closed bars,
                     EET/EEST-safe time conversion.
mt5_execution.py     order_send bracket orders (entry+SL+TP), fixed-fractional
                     lots from tick value, filling-mode/requote retries.
mt5_compat.py        MT5 backend resolver: real MetaTrader5 package (Windows)
                     or the HTTP gateway (macOS/Linux, MT5_GATEWAY_URL).
mt5_gateway.py       Stdlib-only HTTP gateway run inside the Wine/Windows
                     Python next to the terminal (macOS/Linux setups).
live_trading.py      Live H1 loop: bars → features → policy → bracket order.
```

## Step-by-step: run the whole system

Training can happen on any OS. Steps 2 and 8–10 need a running, logged-in MT5
terminal with algo trading enabled — directly on **Windows**, or on **macOS**
via the gateway (see "Running on macOS" below).

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

### 2. Get M1 history from MT5 → CSV

Two equivalent paths; both land at `data/XAUUSD_M1.csv`:

**a) Scripted** (Windows, or macOS via the gateway):

```bash
python mt5_data.py check                          # verify terminal + symbol
python mt5_data.py download --years 10            # writes data/XAUUSD_M1.csv
```

**b) Manual export from the MT5 app** — no Python-side MT5 access needed, the
easiest path on macOS for research: in the terminal press `Ctrl+U` (View →
Symbols) → select XAUUSD → *Bars* tab → request the M1 range → *Export Bars*,
and save the file as `data/XAUUSD_M1.csv`. The loader auto-detects MT5's
native tab-separated `<DATE> <TIME> <OPEN> …` format as well as the
comma-separated `Time (EET),Open,…` flavour.

**c) One-click MQL5 script** — most reliable, pulls the full server history:
open `mql5/ExportHistoryCSV.mq5` in MetaEditor (F4), compile (F7), open any
XAUUSD chart, drag the script from Navigator → Scripts onto it, set the start
year, OK. The file lands in *File → Open Data Folder →* `MQL5/Files/XAUUSD_M1.csv`.

Broker history depth varies — set *Options → Charts → Max bars in chart* to
Unlimited in MT5 first. If your broker's server time is not EET/EEST, change
`CFG.source_tz` (IC Markets is EET/EEST).

To train in a cloud session (Claude Code on the web), get the dataset into
the repo: `gzip -9 data/XAUUSD_M1.csv`, then
`git add -f data/XAUUSD_M1.csv.gz && git commit -m "data" && git push`
(`-f` because `data/` is gitignored; GitHub caps files at 100 MB — the gzip
of ~10 years of M1 fits). The loader reads the `.gz` transparently.

### 3. Audit the features (optional but recommended)

```bash
python features.py
# → "25 observation features … Collinearity audit PASSED: no pair with |r| >= 0.96."
```

### 4. Smoke-test the pipeline on a small slice

```bash
# temporarily set max_days_for_demo = 365 in config.py, then:
python train_ppo.py --mode single --timesteps 100000 --n-envs 1
# revert max_days_for_demo = None afterwards
```

### 5. Full training: sliding walk-forward

```bash
python train_ppo.py --mode sliding --timesteps 3000000 --n-envs 4
```

Per fold: train 5y → select the checkpoint on the next 6m by
`min(q_train, q_val)` with `q = reward − dd_penalty·maxDD%` → evaluate on the
following untouched 6m test window → slide 6m. Outputs:

- `models/sliding/fold_k/` — per-fold model, best checkpoint, eval log
- `models/sliding_walk_forward_summary.csv` — per-fold TEST metrics
- `models/sliding_oos_equity.csv` — stitched continuous OOS equity curve
- `models/run_info.json`, `models/best_model/` — final fold promoted to the
  production slot, with `gate_passed` true/false (and `NO_DEPLOY.txt` when the
  consistency gate failed — live_trading refuses such a model)

### 6. Sealed out-of-sample confirmation — run ONCE

```bash
python final_holdout_eval.py
```

This is the only script that touches the last 10% of history. If the number
disappoints, go back to research; do not iterate against this split.

### 7. Copy artifacts to the trading machine

Copy the whole project + `models/` (at minimum `models/run_info.json`,
`models/best_model/best_model.zip`, `models/best_model/best_model_vecnorm.pkl`)
to the Windows box with the MT5 terminal.

### 8. Dry-run live decisions (no orders)

```bash
python live_trading.py --dry-run
```

Watches for each new closed H1 bar, rebuilds the 25 features from MT5 history,
prints the action + intended bracket. Let it run through a few bars.

### 9. Go live on a DEMO account

```bash
python live_trading.py
```

The loop places market bracket orders via `mt5.order_send` (entry + SL + TP
attached), sized to risk `CFG.risk_fraction` (0.5%) of current equity, tagged
with magic `CFG.mt5_magic`. TP/SL rest server-side; the bot re-syncs open
positions each hour and survives restarts (`live_state.json`, reconstructed
from MT5 if lost).

### 10. Retrain on schedule

The walk-forward already simulates "retrain every 6 months, trade the next 6".
Live, repeat steps 2 → 5 → 6 → 7 on that cadence so the deployed model is
never older than one step window.

## Running on macOS (MT5 under Wine) — the gateway

**Research only (train + backtest) needs none of this.** Export M1 history
straight from the MT5 app (step 2b above) and run steps 3–6 with native macOS
Python — the whole training/backtest pipeline is OS-agnostic. The gateway
below is only for driving the terminal from Python on a Mac: automated
downloads or live trading. If production lives on a Windows box, set the
gateway aside entirely.

The `MetaTrader5` pip package is **Windows-only**: it talks to the terminal
over Windows IPC, so native macOS Python can never import it — even though
the MT5 app itself runs fine on a Mac (the official macOS build is a Wine
wrapper). Installing torch/SB3 into Wine's Windows Python is not realistic,
so this project splits the work in two processes:

```
┌───────────── Wine bottle (same one as the MT5 terminal) ─────────────┐
│  Windows Python:  mt5_gateway.py   (stdlib + MetaTrader5 + numpy)    │
└───────────────────────────── HTTP 127.0.0.1 ─────────────────────────┘
┌───────────────────────── native macOS Python ────────────────────────┐
│  MT5_GATEWAY_URL=http://127.0.0.1:8787                               │
│  mt5_data.py / mt5_execution.py / live_trading.py  (torch, SB3, …)   │
└──────────────────────────────────────────────────────────────────────┘
```

1. **Find the Wine prefix of your MT5 app.** For the official MetaQuotes
   .dmg it is typically
   `~/Library/Application Support/net.metaquotes.wine.metatrader5` and the
   bundled wine binary lives inside the `MetaTrader 5.app` bundle (paths vary
   by version — `find /Applications/MetaTrader\ 5.app -name "wine*" -type f`
   locates it; CrossOver/PlayOnMac bottles have their own prefix paths).
2. **Install Windows Python into that same bottle** (python.org
   `python-3.11.x-amd64.exe`, run it with that wine binary and the
   `WINEPREFIX` above), then inside it:
   `wine python.exe -m pip install MetaTrader5 numpy`.
3. **Start the gateway in the bottle** (MT5 terminal already running &
   logged in): `wine python.exe mt5_gateway.py --port 8787`.
4. **On the macOS side** (this repo, native Python):

   ```bash
   export MT5_GATEWAY_URL=http://127.0.0.1:8787
   python mt5_data.py check
   python mt5_data.py download --years 10
   python live_trading.py --dry-run
   ```

Every `mt5.*` call is forwarded over localhost JSON-RPC; nothing else about
the pipeline changes. Optionally set the same `MT5_GATEWAY_TOKEN` on both
sides to require an auth header. For unattended 24/7 live trading a small
Windows VPS remains the more robust option — a sleeping MacBook trades
nothing.

## Notes and warnings

- **The gate can (and often should) say no.** `gate_passed=false` means the
  edge was not consistent across out-of-sample windows. Live trading refuses
  to start; `--force` overrides for demo experiments only.
- Spread/slippage/commission in `config.py` must match your broker before any
  backtest number is meaningful.
- The `MetaTrader5` Python package is Windows-only; on macOS/Linux run the
  gateway inside the terminal's Wine bottle (section above) or use a Windows
  VPS for steps 2 and 8–10.
- Nothing here is financial advice; trade a demo account until you have months
  of dry-run/demo evidence.
