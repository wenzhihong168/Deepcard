"""Composite weighted multi-task loss.

Total loss = weighted sum of individual task losses:
  * multi-class tasks  -> weighted cross-entropy (class weights inverse to freq.)
  * binary tasks       -> weighted BCE-with-logits (pos_weight inverse to freq.)
See manuscript "Model training strategy and optimization".
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MULTICLASS_TASKS, BINARY_TASKS


class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights=None, pos_weights=None, task_weights=None):
        super().__init__()
        self.class_weights = class_weights or {}   # {task: tensor(NUM_SEVERITY)}
        self.pos_weights = pos_weights or {}       # {task: tensor(1)}
        self.task_weights = task_weights or {}     # {task: float}

    def forward(self, outputs, targets):
        total = 0.0
        logs = {}
        for t in MULTICLASS_TASKS:
            w = self.class_weights.get(t)
            if w is not None:
                w = w.to(outputs[t].device)
            loss = F.cross_entropy(outputs[t], targets[t].long(), weight=w)
            total = total + self.task_weights.get(t, 1.0) * loss
            logs[t] = float(loss.detach())
        for t in BINARY_TASKS:
            pw = self.pos_weights.get(t)
            if pw is not None:
                pw = pw.to(outputs[t].device)
            loss = F.binary_cross_entropy_with_logits(
                outputs[t], targets[t].float(), pos_weight=pw)
            total = total + self.task_weights.get(t, 1.0) * loss
            logs[t] = float(loss.detach())
        return total, logs


def compute_class_weights(labels, num_classes):
    """Inverse-frequency class weights for weighted cross-entropy."""
    counts = np.bincount(np.asarray(labels).astype(int), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
