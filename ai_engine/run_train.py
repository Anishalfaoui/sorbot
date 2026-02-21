"""Quick training script for Sorbot v3.0."""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ml_core.data_loader import fetch_all_timeframes
from ml_core.feature_eng import build_dataset
from ml_core.trainer import train_model

print("Fetching data...")
data = fetch_all_timeframes()

print("Building features...")
htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
dataset = build_dataset(data["1h"], include_target=True, htf_data=htf_data)

print(f"Dataset shape: {dataset.shape}")
tgt = dataset["target"].value_counts().to_dict()
print(f"Target distribution: {tgt}")

print("\nTraining model...")
meta = train_model(dataset)

print(f"\nCV metrics: {meta['cv_metrics']}")
print(f"Final metrics: {meta['final_metrics']}")
print(f"Top features: {meta['top_features'][:10]}")
print("\nDone!")
