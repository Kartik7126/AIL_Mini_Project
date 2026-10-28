from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from frauddet.config import BEST_MODEL_PATH, MODELS_DIR, RANDOM_STATE, TARGET
from frauddet.data import load_transactions, split_supervised
from frauddet.evaluate import evaluate_scores, pick_threshold, save_metrics
from frauddet.features import fit_preprocessor, transform_frame
from frauddet.models import build_detectors, decision_scores
from frauddet.plots import plot_amount_by_class, plot_class_balance, plot_pr_curves, plot_roc_curves


def _subsample_for_slow_models(X: np.ndarray, max_rows: int = 20_000) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx]


def train(use_synthetic_if_missing: bool = True) -> dict[str, Any]:
    df = load_transactions()
    if not use_synthetic_if_missing and len(df) < 100_000:
        raise RuntimeError("Expected the full Kaggle CSV; only a small/synthetic file is present.")

    plot_class_balance(df)
    plot_amount_by_class(df)

    train_df, val_df, test_df = split_supervised(df)
    preprocessor = fit_preprocessor(train_df)
    X_train = transform_frame(preprocessor, train_df)
    X_val = transform_frame(preprocessor, val_df)
    X_test = transform_frame(preprocessor, test_df)
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    # Semi-supervised: fit one-class models on legitimate traffic only.
    X_normal = transform_frame(preprocessor, train_df[train_df[TARGET] == 0])

    detectors = build_detectors()
    metrics: dict[str, Any] = {
        "n_rows": int(len(df)),
        "fraud_rate": float(df[TARGET].mean()),
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "models": {},
    }
    curve_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, Any] = {}

    for name, estimator in detectors.items():
        print(f"Fitting {name}...")
        fit_X = X_normal
        if name in {"one_class_svm", "local_outlier_factor", "autoencoder"}:
            fit_X = _subsample_for_slow_models(X_normal)
        estimator.fit(fit_X)
        val_scores = decision_scores(estimator, X_val)
        threshold = pick_threshold(y_val, val_scores)
        test_scores = decision_scores(estimator, X_test)
        report = evaluate_scores(y_test, test_scores, threshold)
        metrics["models"][name] = report
        curve_payload[name] = (y_test, test_scores)
        fitted[name] = {"estimator": estimator, "threshold": threshold, "report": report}
        print(
            f"  {name}: PR-AUC={report['pr_auc']:.3f}  ROC-AUC={report['roc_auc']:.3f}  "
            f"F1={report['f1']:.3f}  recall={report['recall']:.3f}"
        )

    # Supervised reference (uses labels). Not the project goal, but a useful ceiling.
    print("Fitting random_forest (supervised baseline)...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_X = _subsample_for_slow_models(X_train, max_rows=80_000)
    rng = np.random.default_rng(RANDOM_STATE)
    rf_idx = rng.choice(len(X_train), size=len(rf_X), replace=False) if len(X_train) > len(rf_X) else np.arange(len(X_train))
    rf.fit(X_train[rf_idx], train_df[TARGET].to_numpy()[rf_idx])
    rf_val = rf.predict_proba(X_val)[:, 1]
    rf_thr = pick_threshold(y_val, rf_val)
    rf_test = rf.predict_proba(X_test)[:, 1]
    metrics["models"]["random_forest_supervised"] = evaluate_scores(y_test, rf_test, rf_thr)
    curve_payload["random_forest_supervised"] = (y_test, rf_test)

    anomaly_names = [n for n in fitted]
    best_name = max(anomaly_names, key=lambda n: fitted[n]["report"]["pr_auc"])
    best = fitted[best_name]
    bundle = {
        "name": best_name,
        "preprocessor": preprocessor,
        "estimator": best["estimator"],
        "threshold": best["threshold"],
        "feature_engineer": "frauddet.features.engineer",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BEST_MODEL_PATH)
    metrics["best_anomaly_model"] = best_name
    save_metrics(metrics)
    plot_pr_curves(curve_payload)
    plot_roc_curves(curve_payload)
    print(f"Best anomaly detector: {best_name} -> {BEST_MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train()
