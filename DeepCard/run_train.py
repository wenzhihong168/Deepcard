"""Train DeepCard with 5-fold CV and save the ensemble + scaler.

Usage:
    python run_train.py --train_csv data/train.csv --out artifacts
"""
import argparse
import os
import pickle

import pandas as pd
import torch

from config import MULTICLASS_TASKS, BINARY_TASKS
from feature_engineering import add_derived_features, ZScoreScaler
from data import order_features
from train import cross_validate


def load_xy(csv_path):
    df = add_derived_features(pd.read_csv(csv_path))
    X = order_features(df)
    labels = {t: df[t].to_numpy() for t in (MULTICLASS_TASKS + BINARY_TASKS)}
    return X, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, labels = load_xy(args.train_csv)
    scaler = ZScoreScaler().fit(X)
    Xs = scaler.transform(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = cross_validate(Xs, labels, device=device)

    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(args.out, f"deepcard_fold{i}.pt"))
    with open(os.path.join(args.out, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved {len(models)} fold models + scaler to '{args.out}'")


if __name__ == "__main__":
    main()
