"""Reveal the sealed final-test split and evaluate the frozen model ONCE.

This is the only script allowed to touch the last CFG.test_frac of the data.
Everything upstream (training, checkpoint selection, the walk-forward gate)
never saw these bars.  Run it a single time after the model is frozen; if the
result disappoints, the honest path is back to research — not re-running this
script with tweaks until it passes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from config import CFG
from model_artifacts import load_run_info, resolve_project_path, resolve_sb3_model_path
from train_ppo import _rollout_on_split, load_datasets


def main(out_dir: str = "outputs/final_holdout", models_dir: str = "models") -> None:
    print("Revealing the sealed test split and writing holdout-only artifacts.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _, run_info = load_run_info(models_dir)
    model_path = resolve_sb3_model_path(run_info["model_path"], ".")
    vecnorm_path = resolve_project_path(run_info["vecnorm_path"], ".")
    best_model_path = Path(models_dir) / "best_model" / "best_model.zip"
    if best_model_path.exists():
        print(f"Using best validation checkpoint: {best_model_path}")
        model_path = best_model_path
        # Pair the best checkpoint with the normalisation it was selected under.
        if "best_model_vecnorm_path" in run_info:
            vecnorm_path = resolve_project_path(run_info["best_model_vecnorm_path"], ".")
        elif (Path(models_dir) / "best_model" / "best_model_vecnorm.pkl").exists():
            vecnorm_path = Path(models_dir) / "best_model" / "best_model_vecnorm.pkl"
    else:
        print(f"Using final checkpoint from run_info.json: {model_path}")

    if run_info.get("gate_passed") is False:
        print("WARNING: the walk-forward consistency gate FAILED for this model "
              "(gate_passed=false). The holdout number below is informational only.")

    m1, feature_cols, _train_feat, _val_feat, test_feat = load_datasets()
    print(f"Sealed test window: {test_feat.index.min()} → {test_feat.index.max()}  "
          f"({len(test_feat):,} bars)")

    model = PPO.load(str(model_path))
    equity, trades, report = _rollout_on_split(model, vecnorm_path, m1, feature_cols, test_feat)

    equity.to_csv(out_path / "rl_test_equity_curve.csv")
    trades.to_csv(out_path / "rl_test_trade_log.csv", index=False)
    report_df = pd.DataFrame(
        [{"scenario": "rl_test", "metric": k, "value": v} for k, v in report.items()]
    )
    report_df.to_csv(out_path / "holdout_report.csv", index=False)

    print(report_df.set_index("metric")["value"])
    print(f"Saved holdout artifacts to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
