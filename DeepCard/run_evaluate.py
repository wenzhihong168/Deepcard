"""Evaluate a trained DeepCard ensemble on a held-out / external CSV.

Usage:
    python run_evaluate.py --test_csv data/internal_test.csv --artifacts artifacts
"""
import argparse
import glob
import os
import pickle

import pandas as pd
import torch

from config import MODEL, MULTICLASS_TASKS, BINARY_TASKS
from feature_engineering import add_derived_features
from data import order_features
from model import DeepCard
from train import ensemble_predict
from evaluate import evaluate_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    df = add_derived_features(pd.read_csv(args.test_csv))
    X = order_features(df)
    with open(os.path.join(args.artifacts, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    Xs = scaler.transform(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path in sorted(glob.glob(os.path.join(args.artifacts, "deepcard_fold*.pt"))):
        m = DeepCard(MODEL).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        models.append(m)
    if not models:
        raise SystemExit("No fold checkpoints found in artifacts directory.")

    preds = ensemble_predict(models, Xs, device=device)
    y_true = {t: df[t].to_numpy() for t in (MULTICLASS_TASKS + BINARY_TASKS)}
    results = evaluate_all(y_true, preds)

    for task, metrics in results.items():
        pretty = {k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in metrics.items()}
        print(f"{task}: {pretty}")


if __name__ == "__main__":
    main()
