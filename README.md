# Credit Card Fraud Detection

Identify fraudulent card transactions as **anomalies**: rare events that do not look like the bulk of legitimate spend.

The source data is the public [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) set (also mirrored by TensorFlow). About **284,807** European card transactions from two days, of which **492 (0.172%)** are fraud. Features `V1`–`V28` are PCA components; only `Time` and `Amount` are raw.

## Why anomaly detection?

A classifier that always predicts “legitimate” is already ~99.8% accurate. Ranking quality (**PR-AUC**) and recall at a usable false-positive rate matter more than accuracy. Unsupervised / one-class models fit the *normal* class and flag deviations — which matches how a fraud team would score traffic when labels are delayed or incomplete.

## Approach

1. Engineer `Hour` (from `Time`) and `LogAmount`, then scale the non-PCA columns.
2. Fit detectors **only on legitimate train rows** (semi-supervised one-class):
   - Isolation Forest
   - Local Outlier Factor (novelty mode)
   - One-Class SVM (subsampled)
   - Shallow reconstruction autoencoder (MLP, reconstruction error)
3. Pick a score threshold on a labelled validation split by **F1**.
4. Report **PR-AUC**, ROC-AUC, precision, recall, and confusion counts on a held-out test set.
5. Compare against a supervised Random Forest as a ceiling, not as the primary method.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m frauddet download    # ~150 MB CSV into data/creditcard.csv
python -m frauddet train       # fits models, writes models/ and reports/
python -m frauddet metrics
python -m frauddet predict --input path/to/transactions.csv
python -m frauddet ui          # http://127.0.0.1:7860
```

If the download cannot reach the public mirror, the CLI writes a smaller **synthetic** dataset so the pipeline still runs.

Score the bundled demo file:

```bash
python -m frauddet predict --input examples/sample_transactions.csv
```

EDA notebook: `notebooks/01_eda_and_detection.ipynb`.

## Results on a held-out 30% test split

Trained one-class models on legitimate train rows only (labels unused at fit time). Thresholds chosen by F1 on validation. **PR-AUC** is the headline metric because the positive class is ~0.17%.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local Outlier Factor | 0.623 | 0.943 | 0.744 | 0.628 | 0.681 |
| Autoencoder | 0.303 | 0.948 | 0.694 | 0.169 | 0.272 |
| One-Class SVM | 0.252 | 0.927 | 0.281 | 0.541 | 0.370 |
| Isolation Forest | 0.078 | 0.942 | 0.121 | 0.284 | 0.170 |
| Random Forest (supervised ceiling) | 0.797 | 0.940 | 0.851 | 0.770 | 0.809 |

LOF is the saved production detector (`models/best_detector.joblib`). The supervised forest is a labelled upper bound, not the project’s primary method. Plots live in `reports/`.

## Project layout

```
src/frauddet/    download, features, models, train, predict, local UI
notebooks/       exploration
tests/           synthetic-data smoke tests
data/            CSV (gitignored)
models/          best detector joblib
reports/         metrics JSON + plots
```

## Notes

This is a research-style detector, not a payment-authorization system. PCA features cannot be inverted to raw merchant fields, and a production stack would add streaming features, rules, and human review.
