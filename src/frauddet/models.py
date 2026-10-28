from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

from frauddet.config import FRAUD_RATE, RANDOM_STATE


class ReconstructionAutoencoder(BaseEstimator, OutlierMixin):
    """Shallow autoencoder: reconstruction error is the anomaly score.

    sklearn-compatible: `decision_function` is higher for inliers (like IsolationForest).
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (16, 8, 16),
        max_iter: int = 15,
        contamination: float = FRAUD_RATE,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=3,
        )
        self.mlp_.fit(X, X)
        errors = self._errors(X)
        self.offset_ = float(np.quantile(errors, 1.0 - self.contamination))
        return self

    def _errors(self, X) -> np.ndarray:
        recon = self.mlp_.predict(X)
        return np.mean((np.asarray(X) - recon) ** 2, axis=1)

    def decision_function(self, X) -> np.ndarray:
        """Higher means more normal."""
        return self.offset_ - self._errors(X)

    def score_samples(self, X) -> np.ndarray:
        return -self._errors(X)

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)


def build_detectors(contamination: float = FRAUD_RATE) -> dict[str, BaseEstimator]:
    """Unsupervised / one-class models. Labels are not used at fit time."""
    return {
        "isolation_forest": IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        ),
        "one_class_svm": OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=min(0.05, max(contamination * 5, 0.01)),
        ),
        "autoencoder": ReconstructionAutoencoder(
            contamination=contamination,
            max_iter=12,
        ),
    }


def decision_scores(estimator, X) -> np.ndarray:
    """Unified anomaly scores: larger => more likely fraud.

    sklearn outlier estimators use the opposite convention (larger => more inlier).
    """
    if hasattr(estimator, "decision_function"):
        return -np.asarray(estimator.decision_function(X), dtype=float)
    if hasattr(estimator, "score_samples"):
        return -np.asarray(estimator.score_samples(X), dtype=float)
    pred = np.asarray(estimator.predict(X))
    return (pred == -1).astype(float)
