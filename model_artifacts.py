"""Locate saved model artifacts (model zip + VecNormalize pkl + run_info)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_project_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a project-relative artifact path without forcing it to exist."""
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved


def resolve_sb3_model_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve an SB3 model path across both `.zip` and no-extension saves.

    Slugs contain decimal points, so Path.suffix is non-empty and SB3 may not
    auto-append `.zip` on write depending on how the path was passed.
    """
    requested = resolve_project_path(path, base_dir=base_dir)
    candidates = [requested]

    requested_str = str(requested)
    if requested_str.lower().endswith(".zip"):
        candidates.append(Path(requested_str[:-4]))
    else:
        candidates.append(Path(f"{requested}.zip"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Stable-Baselines3 model artifact not found. Tried:\n{tried}")


def load_run_info(models_dir: str | Path = "models") -> tuple[Path, dict[str, Any]]:
    models_path = Path(models_dir)
    info_path = models_path / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found.\nRun train_ppo.py first, then retry."
        )
    return info_path, json.loads(info_path.read_text())
