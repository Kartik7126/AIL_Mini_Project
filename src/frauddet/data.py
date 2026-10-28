from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from frauddet.config import (
    DATASET_URLS,
    FRAUD_RATE,
    RANDOM_STATE,
    RAW_CSV,
    SYNTHETIC_CSV,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


def download_kaggle_dataset(dest: Path = RAW_CSV) -> Path:
    """Download the ULB credit-card dataset. Falls back to a local synthetic copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest

    last_error: Exception | None = None
    for url in DATASET_URLS:
        try:
            print(f"Downloading dataset from {url}")
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 1_000_000:
                print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
                return dest
            dest.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 — network / HTTP errors are expected
            last_error = exc
            dest.unlink(missing_ok=True)
            print(f"Download failed: {exc}")

    print("Could not download the public dataset; writing a synthetic stand-in.")
    if last_error:
        print(f"Last error: {last_error}")
    return write_synthetic_dataset(SYNTHETIC_CSV)


def write_synthetic_dataset(dest: Path = SYNTHETIC_CSV, n: int = 20_000) -> Path:
    """Imbalanced stand-in so the pipeline can run without the 150 MB Kaggle file."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = max(40, int(n * FRAUD_RATE * 4))  # slightly richer than real prior for demo
    n_ok = n - n_fraud

    def _block(size: int, loc: float, scale: float, amount_loc: float) -> pd.DataFrame:
        v = rng.normal(loc=loc, scale=scale, size=(size, 28))
        time = rng.uniform(0, 172_800, size=size)
        amount = np.abs(rng.lognormal(mean=amount_loc, sigma=1.3, size=size))
        cols = {"Time": time, **{f"V{i}": v[:, i - 1] for i in range(1, 29)}, "Amount": amount}
        return pd.DataFrame(cols)

    ok = _block(n_ok, loc=0.0, scale=1.0, amount_loc=3.2)
    ok[TARGET] = 0
    fraud = _block(n_fraud, loc=1.8, scale=2.4, amount_loc=4.5)
    fraud[TARGET] = 1
    frame = pd.concat([ok, fraud], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    print(f"Wrote synthetic dataset {dest} with {n_fraud} fraud rows of {n}")
    return dest


def load_transactions(path: Path | None = None) -> pd.DataFrame:
    csv = path or (RAW_CSV if RAW_CSV.exists() else SYNTHETIC_CSV)
    if not csv.exists():
        csv = download_kaggle_dataset()
        if not RAW_CSV.exists():
            csv = SYNTHETIC_CSV
    df = pd.read_csv(csv)
    missing = {"Time", "Amount", TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"{csv} is missing columns: {sorted(missing)}")
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test. Labels are only used for evaluation and thresholding."""
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE, stratify=train_val[TARGET], random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
