from pathlib import Path

import pandas as pd

from frauddet.config import TARGET
from frauddet.data import split_supervised, write_synthetic_dataset
from frauddet.evaluate import evaluate_scores, pick_threshold
from frauddet.features import fit_preprocessor, transform_frame
from frauddet.models import ReconstructionAutoencoder, build_detectors, decision_scores


def test_synthetic_split_and_scores(tmp_path: Path):
    csv = write_synthetic_dataset(tmp_path / "synth.csv", n=2_000)
    df = pd.read_csv(csv)
    assert set(df[TARGET].unique()) == {0, 1}
    train, val, test = split_supervised(df)
    prep = fit_preprocessor(train)
    X_normal = transform_frame(prep, train[train[TARGET] == 0])
    X_val = transform_frame(prep, val)
    X_test = transform_frame(prep, test)

    model = build_detectors()["isolation_forest"]
    model.fit(X_normal)
    val_scores = decision_scores(model, X_val)
    thr = pick_threshold(val[TARGET].to_numpy(), val_scores)
    report = evaluate_scores(test[TARGET].to_numpy(), decision_scores(model, X_test), thr)
    assert 0.0 <= report["pr_auc"] <= 1.0
    assert 0.0 <= report["roc_auc"] <= 1.0
    assert report["fraud_support"] >= 1


def test_autoencoder_decision_shape():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    ae = ReconstructionAutoencoder(hidden_layer_sizes=(6, 3, 6), max_iter=5, contamination=0.05)
    ae.fit(X)
    scores = ae.decision_function(X)
    assert scores.shape == (200,)
    pred = ae.predict(X)
    assert set(pred).issubset({-1, 1})
