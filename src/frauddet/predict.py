from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from frauddet.config import BEST_MODEL_PATH
from frauddet.features import transform_frame
from frauddet.models import decision_scores


def load_bundle(path: Path = BEST_MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m frauddet train")
    return joblib.load(path)


def score_frame(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    X = transform_frame(bundle["preprocessor"], df)
    scores = decision_scores(bundle["estimator"], X)
    out = df.copy()
    out["anomaly_score"] = scores
    out["predicted_fraud"] = (scores >= bundle["threshold"]).astype(int)
    return out


def score_csv(input_csv: Path, output_csv: Path | None = None) -> Path:
    df = pd.read_csv(input_csv)
    scored = score_frame(df)
    dest = output_csv or input_csv.with_name(input_csv.stem + "_scored.csv")
    scored.to_csv(dest, index=False)
    return dest
