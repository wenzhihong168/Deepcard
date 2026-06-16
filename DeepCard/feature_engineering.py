"""Derived hemodynamic / structural indices and z-score standardization.

Implements the formulas described in the manuscript section "Quantitative
parameter extraction and feature engineering".
"""
import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the derived indices used by DeepCard and append them as columns.

    Raw inputs expected (when available): LVEDV, LVESV, LVEDD, LVESD, IVSd,
    LVPWd, MV_E, MV_A, e_septal, e_lateral, AV_Vmax, TR_Vmax, RAP, weight_kg,
    height_cm.
    """
    df = df.copy()

    # Left-ventricular systolic function
    if {"LVEDV", "LVESV"}.issubset(df.columns):
        df["LVEF"] = (df["LVEDV"] - df["LVESV"]) / df["LVEDV"] * 100.0
    if {"LVEDD", "LVESD"}.issubset(df.columns):
        df["FS"] = (df["LVEDD"] - df["LVESD"]) / df["LVEDD"] * 100.0

    # Diastolic ratios
    if {"MV_E", "MV_A"}.issubset(df.columns):
        df["E_A_ratio"] = df["MV_E"] / df["MV_A"]
    if {"MV_E", "e_septal", "e_lateral"}.issubset(df.columns):
        e_mean = (df["e_septal"] + df["e_lateral"]) / 2.0
        df["E_e_ratio"] = df["MV_E"] / e_mean

    # Simplified Bernoulli: peak gradient = 4 * Vmax^2  (V in m/s -> mmHg)
    if "AV_Vmax" in df.columns:
        df["AV_peak_grad"] = 4.0 * df["AV_Vmax"] ** 2

    # Right-ventricular systolic pressure = 4 * (TR Vmax)^2 + RAP
    if {"TR_Vmax", "RAP"}.issubset(df.columns):
        df["RVSP"] = 4.0 * df["TR_Vmax"] ** 2 + df["RAP"]

    # Du Bois body-surface area (weight in kg, height in cm)
    if {"weight_kg", "height_cm"}.issubset(df.columns):
        df["BSA"] = 0.007184 * df["weight_kg"] ** 0.425 * df["height_cm"] ** 0.725

    # ASE left-ventricular mass and relative wall thickness (dimensions in cm)
    if {"LVEDD", "IVSd", "LVPWd"}.issubset(df.columns):
        df["LVM"] = 0.8 * (1.04 * ((df["LVEDD"] + df["IVSd"] + df["LVPWd"]) ** 3
                                   - df["LVEDD"] ** 3)) + 0.6
        df["RWT"] = (2.0 * df["LVPWd"]) / df["LVEDD"]

    return df


class ZScoreScaler:
    """z-score standardization with statistics fit on the TRAINING set only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
