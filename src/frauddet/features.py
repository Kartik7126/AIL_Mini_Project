from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from frauddet.config import PCA_FEATURES


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Hour"] = (out["Time"] / 3600.0) % 24.0
    out["LogAmount"] = np.log1p(out["Amount"].clip(lower=0))
    return out


FEATURE_COLUMNS = ["Time", "Amount", "Hour", "LogAmount", *PCA_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Scale raw columns. PCA components V1–V28 are already standardized in the source data."""
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Time", "Amount", "Hour", "LogAmount"]),
            ("pca", "passthrough", PCA_FEATURES),
        ],
        remainder="drop",
    )


def transform_frame(preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    engineered = engineer(df)
    return preprocessor.transform(engineered)


def fit_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(engineer(train_df))
    return preprocessor


def make_feature_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
