from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RAW_CSV = DATA_DIR / "creditcard.csv"
SYNTHETIC_CSV = DATA_DIR / "creditcard_synthetic.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_detector.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE = 0.20  # fraction of the train split, used to pick a decision threshold

# Dataset prior (~0.172% fraud). Used as IsolationForest contamination.
FRAUD_RATE = 0.00172

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", "Amount", *PCA_FEATURES]
TARGET = "Class"

DATASET_URLS = [
    # Public copy of the Kaggle ULB dataset (same schema: Time, V1–V28, Amount, Class)
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
]
