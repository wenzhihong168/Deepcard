"""Training: stratified 5-fold cross-validation, Adam + cosine annealing + L2,
ensemble by averaging fold probabilities. See manuscript "Cross-validation and
evaluation logic".
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from config import (MULTICLASS_TASKS, BINARY_TASKS, NUM_SEVERITY,
                    MODEL, TRAIN, SEED)
from model import DeepCard
from losses import MultiTaskLoss, compute_class_weights
from data import EchoDataset, add_gaussian_noise


def _make_loss(labels):
    """Build a MultiTaskLoss with inverse-frequency weights from the labels."""
    class_weights, pos_weights = {}, {}
    for t in MULTICLASS_TASKS:
        class_weights[t] = compute_class_weights(labels[t], NUM_SEVERITY)
    for t in BINARY_TASKS:
        y = np.asarray(labels[t]).astype(int)
        n_pos = max(int(y.sum()), 1)
        n_neg = max(len(y) - int(y.sum()), 1)
        pos_weights[t] = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    return MultiTaskLoss(class_weights, pos_weights)


def _train_epoch(model, loader, criterion, optimizer, device, noise_std):
    model.train()
    for X, y in loader:
        X = add_gaussian_noise(X.to(device), noise_std)      # augmentation
        y = {t: v.to(device) for t, v in y.items()}
        optimizer.zero_grad()
        out, _ = model(X)
        loss, _ = criterion(out, y)
        loss.backward()
        optimizer.step()


def cross_validate(X, labels, device="cpu"):
    """Stratified 5-fold CV. Returns the list of trained fold models (ensemble)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X = np.asarray(X)
    strat = np.asarray(labels[MULTICLASS_TASKS[0]]).astype(int)   # stratify on a primary task
    skf = StratifiedKFold(n_splits=TRAIN["n_folds"], shuffle=True, random_state=SEED)

    models = []
    for fold, (tr, _) in enumerate(skf.split(X, strat)):
        model = DeepCard(MODEL).to(device)
        fold_labels = {t: np.asarray(v)[tr] for t, v in labels.items()}
        criterion = _make_loss(fold_labels)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"],
                                     weight_decay=TRAIN["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAIN["cosine_T_max"])

        ds = EchoDataset(X[tr], fold_labels)
        loader = DataLoader(ds, batch_size=TRAIN["batch_size"], shuffle=True)
        for _ in range(TRAIN["epochs"]):
            _train_epoch(model, loader, criterion, optimizer, device,
                         TRAIN["gaussian_noise_std"])
            scheduler.step()
        models.append(model)
        print(f"[fold {fold + 1}/{TRAIN['n_folds']}] training complete")
    return models


@torch.no_grad()
def ensemble_predict(models, X, device="cpu"):
    """Average predicted probabilities across the fold models."""
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    mc = {t: [] for t in MULTICLASS_TASKS}
    bn = {t: [] for t in BINARY_TASKS}
    for m in models:
        m.eval()
        out, _ = m(Xt)
        for t in MULTICLASS_TASKS:
            mc[t].append(torch.softmax(out[t], dim=1).cpu().numpy())
        for t in BINARY_TASKS:
            bn[t].append(torch.sigmoid(out[t]).cpu().numpy())
    preds = {t: np.mean(mc[t], axis=0) for t in MULTICLASS_TASKS}
    preds.update({t: np.mean(bn[t], axis=0) for t in BINARY_TASKS})
    return preds
