"""Dataset, fixed feature ordering, augmentation, and SMOTE oversampling."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEATURE_ORDER, MULTICLASS_TASKS, BINARY_TASKS, TRAIN


class EchoDataset(Dataset):
    """Standardized feature matrix + per-task labels.

    Multi-class task labels are encoded 0..4 (the five severity grades);
    binary task labels are encoded 0/1. See DATA_FORMAT.md for the schema.
    """

    def __init__(self, X, labels):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.labels = {t: torch.tensor(np.asarray(v)) for t, v in labels.items()}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        y = {t: v[i] for t, v in self.labels.items()}
        return self.X[i], y


def order_features(df: pd.DataFrame) -> np.ndarray:
    """Return the feature matrix arranged in the fixed clinical 1-D order."""
    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[FEATURE_ORDER].to_numpy(dtype=float)


def add_gaussian_noise(x: torch.Tensor, std: float = TRAIN["gaussian_noise_std"]):
    """Augmentation: inject Gaussian noise to simulate measurement uncertainty."""
    return x + torch.randn_like(x) * std


def _physiologically_valid(row, feat_index) -> bool:
    """Rule-based filter for synthetic SMOTE samples.

    Discards anatomically/hemodynamically impossible instances, e.g. the
    end-systolic dimension/volume must be smaller than the end-diastolic one.
    """
    try:
        if row[feat_index["LVESD"]] >= row[feat_index["LVEDD"]]:
            return False
        if row[feat_index["LVESV"]] >= row[feat_index["LVEDV"]]:
            return False
    except KeyError:
        pass
    return True


def smote_oversample(X, y, random_state=42):
    """SMOTE oversampling for one label vector, followed by a rule-based
    physiological-plausibility filter. TRAINING DATA ONLY.

    The manuscript applies SMOTE exclusively to the training split and never to
    the internal test set or the external validation cohort.
    """
    from imblearn.over_sampling import SMOTE
    feat_index = {f: i for i, f in enumerate(FEATURE_ORDER)}
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(np.asarray(X), np.asarray(y))
    keep = np.array([_physiologically_valid(r, feat_index) for r in X_res])
    return X_res[keep], y_res[keep]
