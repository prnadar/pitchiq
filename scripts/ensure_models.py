"""Ensure trained models exist before the server starts.

Models are committed under backend/models/saved/, so a normal deploy is a no-op.
Training only runs if a model file is missing (e.g. a fresh checkout without LFS).
Used as the Render build step; safe to run locally too.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAVED_DIR = ROOT / "backend" / "models" / "saved"
REQUIRED = (
    "xgb_model.pkl",
    "lgb_model.pkl",
    "lr_model.pkl",
    "signal_feature_idx.pkl",
    "signal_feature_names.pkl",
)


def main() -> int:
    missing = [name for name in REQUIRED if not (SAVED_DIR / name).exists()]
    if not missing:
        print(f"All {len(REQUIRED)} model files present in {SAVED_DIR} — skipping training.")
        return 0

    print(f"Missing model files: {', '.join(missing)} — training now.")
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    from backend.models.train import train_and_evaluate

    train_and_evaluate()

    still_missing = [name for name in REQUIRED if not (SAVED_DIR / name).exists()]
    if still_missing:
        print(f"ERROR: training finished but these are still missing: {', '.join(still_missing)}")
        return 1
    print("Training complete — all model files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
