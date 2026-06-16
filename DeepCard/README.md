# DeepCard

Code for **DeepCard**, a multi-task deep-learning framework that produces a
standardized, reproducible interpretation of pre-measured quantitative
echocardiographic parameters. DeepCard takes 39 standardized echocardiographic
measurements and jointly predicts 17 diagnostic outputs (8 multi-class severity
gradings + 9 binary classifications) using a residual 1-D CNN with multi-head
self-attention.

> This repository accompanies the article *"DeepCard enables standardized
> interpretation of multiparameter cardiac ultrasound measurements and disease
> classification"* (iScience, 2026).

## Why the data are not here

The raw echocardiographic images and clinical records are protected by
patient-privacy regulations and institutional policies under Chinese healthcare
data-protection laws and therefore **cannot be redistributed**. This repository
contains the **complete model and analysis code** plus a data-format
specification (`DATA_FORMAT.md`) so the pipeline can be run on appropriately
approved data. De-identified data may be requested from the lead contact subject
to IRB approval and a data transfer agreement (see the paper).

## Repository structure

```
DeepCard/
├── config.py               # features (39), tasks (17), hyperparameters
├── feature_engineering.py  # derived indices (LVEF, FS, Bernoulli, RVSP, BSA, LVM, RWT) + z-score
├── data.py                 # dataset, fixed feature ordering, augmentation, SMOTE + physiological filter
├── model.py                # DeepCard: residual 1-D CNN + multi-head attention + multi-task heads
├── losses.py               # weighted multi-task loss (class/positive weights)
├── train.py                # stratified 5-fold CV, Adam + cosine annealing + L2, ensemble
├── evaluate.py             # sensitivity/specificity/precision/F1/accuracy/AUC + bootstrap 95% CI
├── interpret.py            # SHAP global/local feature importance
├── run_train.py            # CLI: train + save ensemble
├── run_evaluate.py         # CLI: evaluate ensemble on a held-out/external CSV
├── DATA_FORMAT.md          # input schema (no data included)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
```

## Usage

Prepare CSV files following `DATA_FORMAT.md`, then:

```bash
# Train (stratified 5-fold CV; saves fold checkpoints + scaler to ./artifacts)
python run_train.py --train_csv data/train.csv --out artifacts

# Evaluate on the internal test set and the external cohort
python run_evaluate.py --test_csv data/internal_test.csv --artifacts artifacts
python run_evaluate.py --test_csv data/external.csv      --artifacts artifacts
```

Interpretability (SHAP global importance for one task):

```python
import pandas as pd, torch
from config import MODEL
from model import DeepCard
from feature_engineering import add_derived_features
from data import order_features
from interpret import shap_global_importance

df = add_derived_features(pd.read_csv("data/internal_test.csv"))
X = order_features(df)
model = DeepCard(MODEL); model.load_state_dict(torch.load("artifacts/deepcard_fold0.pt"))
imp = shap_global_importance(model, X[:50], X[:50], task="mitral_regurgitation")
print(sorted(imp.items(), key=lambda kv: -kv[1])[:10])
```

## Architecture summary

* **Input** — 39 standardized measurements arranged in a fixed, clinically
  defined 1-D order (physiologically coupled parameters in contiguous blocks).
* **Backbone** — conv stem → 5 residual 1-D conv blocks (kernel 3, BatchNorm,
  ReLU, dropout, skip connections).
* **Attention** — 8-head self-attention over the 39 feature positions, then
  global average pooling.
* **Heads** — 8 multi-class severity heads (softmax) + 9 binary heads (sigmoid).
* **Training** — weighted multi-task loss, Adam + L2 (weight decay), cosine-
  annealing LR, Gaussian-noise augmentation, and stratified 5-fold CV with
  ensemble averaging. Class imbalance is handled by the inverse-frequency
  weighted loss; a SMOTE utility (training only, with a physiological-
  plausibility filter) is provided for the per-task oversampling experiments.

## Reproducibility notes

Hyperparameters are centralized in `config.py` and the global seed is `42`.
The block-wise ordering of the 39 features is defined in `config.FEATURE_ORDER`
and matches Supplemental Table S* of the paper.

## Citation

```bibtex
@article{deepcard2026,
  title   = {DeepCard enables standardized interpretation of multiparameter cardiac ultrasound measurements and disease classification},
  author  = {Wen, Zhihong and Liu, Xiangpeng and Liu, Yi and Tang, Wenbo and An, Kang and Guo, Shengli},
  journal = {iScience},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
