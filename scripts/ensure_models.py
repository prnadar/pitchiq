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


def _stale_reason() -> str | None:
    """Why the committed models cannot be trusted, or None if they are fine.

    Pickles carry the library version they were written with. Unpickling under
    a different scikit-learn emits InconsistentVersionWarning and, per sklearn's
    own docs, can silently produce invalid results — so treat it as stale and
    retrain rather than serving predictions we cannot vouch for.
    """
    missing = [name for name in REQUIRED if not (SAVED_DIR / name).exists()]
    if missing:
        return f"missing model files: {', '.join(missing)}"

    import warnings

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # older sklearn without the warning class
        InconsistentVersionWarning = None  # type: ignore[assignment]

    from backend.models.predict import EnsemblePredictor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            EnsemblePredictor(SAVED_DIR).load()
        except Exception as exc:
            return f"models failed to load: {type(exc).__name__}: {exc}"

    if InconsistentVersionWarning is not None:
        mismatches = [
            str(w.message) for w in caught
            if issubclass(w.category, InconsistentVersionWarning)
        ]
        if mismatches:
            return f"library version mismatch ({len(mismatches)} estimator(s)): {mismatches[0]}"
    return None


def main() -> int:
    reason = _stale_reason()
    if reason is None:
        print(f"Models in {SAVED_DIR} load cleanly under the installed libraries — skipping training.")
        return 0

    print(f"Retraining because: {reason}")
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
