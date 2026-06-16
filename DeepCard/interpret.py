"""SHAP-based interpretability (global and local feature importance).

Implements the manuscript "Interpretability analysis": SHAP values decompose a
prediction into per-feature contributions; global importance is the mean
absolute SHAP value across the dataset.
"""
import numpy as np
import torch

from config import FEATURE_ORDER


class _TaskWrapper(torch.nn.Module):
    """Expose a single task's logits so SHAP can explain one head at a time."""

    def __init__(self, base, task):
        super().__init__()
        self.base = base
        self.task = task

    def forward(self, x):
        out, _ = self.base(x)
        logit = out[self.task]
        return logit.unsqueeze(1) if logit.dim() == 1 else logit


def shap_global_importance(model, X_background, X_explain, task, device="cpu"):
    """Return {feature: mean_abs_SHAP} for one task using GradientExplainer."""
    import shap

    wrapper = _TaskWrapper(model, task).to(device).eval()
    bg = torch.tensor(np.asarray(X_background), dtype=torch.float32, device=device)
    ex = torch.tensor(np.asarray(X_explain), dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(wrapper, bg)
    sv = np.asarray(explainer.shap_values(ex))
    # Average |SHAP| over samples (and classes for multi-class outputs)
    mean_abs = np.mean(np.abs(sv), axis=tuple(range(sv.ndim - 1)))
    mean_abs = np.asarray(mean_abs).reshape(-1)[:len(FEATURE_ORDER)]
    return dict(zip(FEATURE_ORDER, mean_abs))
