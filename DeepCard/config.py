"""Central configuration for DeepCard.

All feature names, task definitions, and hyperparameters live here so that the
model, training, and evaluation code stay in sync. Values follow the description
in the manuscript (Method details: "Deep learning model architecture",
"Model training strategy and optimization", and "Quantitative parameter
extraction and feature engineering").
"""

SEED = 42

# ----------------------------------------------------------------------------
# Input features (39), arranged in a FIXED, clinically defined 1-D order.
# Because 1-D convolutions impose a locality bias, neighbouring features are
# physiologically coupled (contiguous blocks: LV systolic, LV diastolic,
# valvular hemodynamics, right-heart/pulmonary, structural). Features tagged as
# derived are computed in feature_engineering.py.
# ----------------------------------------------------------------------------
FEATURE_ORDER = [
    # Block 1 - LV structure & systolic function
    "LVEDD", "LVESD", "IVSd", "LVPWd", "LVEDV", "LVESV",
    "LVEF",            # derived
    "FS",              # derived
    "SV",
    # Block 2 - LV diastolic function
    "MV_E", "MV_A",
    "E_A_ratio",       # derived
    "e_septal", "e_lateral",
    "E_e_ratio",       # derived
    "LAVI", "DT",
    # Block 3 - Valvular hemodynamics
    "AV_Vmax",
    "AV_peak_grad",    # derived (simplified Bernoulli)
    "AV_mean_grad", "AVA", "MV_Vmax", "MR_jet_area", "AR_PHT", "MV_VTI",
    # Block 4 - Right heart & pulmonary circulation
    "TR_Vmax",
    "RVSP",            # derived
    "PA_diam", "RAP", "TAPSE", "RV_diam", "IVC_diam", "IVC_collapse",
    # Block 5 - Structural / other
    "Ao_root", "LA_diam",
    "LVM",             # derived
    "RWT",             # derived
    "BSA",             # derived
    "PE_amount",
]
assert len(FEATURE_ORDER) == 39, "FEATURE_ORDER must contain exactly 39 features"

DERIVED_FEATURES = ["LVEF", "FS", "E_A_ratio", "E_e_ratio",
                    "AV_peak_grad", "RVSP", "LVM", "RWT", "BSA"]

# ----------------------------------------------------------------------------
# Multi-task definition: 8 multi-class (severity) + 9 binary = 17 tasks.
# ----------------------------------------------------------------------------
SEVERITY_CLASSES = ["Mild", "Mild-Moderate", "Moderate", "Moderate-Severe", "Severe"]
NUM_SEVERITY = len(SEVERITY_CLASSES)

MULTICLASS_TASKS = [
    "mitral_regurgitation",
    "tricuspid_regurgitation",
    "aortic_regurgitation",
    "aortic_stenosis",
    "pulmonary_regurgitation",
    "lv_diastolic_dysfunction",
    "pulmonary_hypertension",
    "lv_systolic_dysfunction",
]   # 8 tasks, each with NUM_SEVERITY classes

BINARY_TASKS = [
    "pulmonary_artery_pressure_abnormal",
    "left_atrial_enlargement",
    "right_atrial_enlargement",
    "left_ventricular_enlargement",
    "right_ventricular_enlargement",
    "left_ventricular_hypertrophy",
    "interventricular_septal_thickening",
    "pericardial_effusion",
    "regional_wall_motion_abnormality",
]   # 9 tasks

# ----------------------------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------------------------
MODEL = dict(
    in_channels=1,
    hidden_channels=64,
    num_res_blocks=5,      # five sequential residual blocks
    kernel_size=3,         # 1-D conv kernel size = 3, stride = 1
    attn_heads=8,          # 8 parallel self-attention heads
    dropout=0.3,
)

# ----------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------
TRAIN = dict(
    batch_size=32,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,        # L2 regularization
    cosine_T_max=100,         # cosine-annealing LR schedule
    gaussian_noise_std=0.05,  # Gaussian-noise augmentation
    n_folds=5,                # stratified 5-fold cross-validation
    # Multi-task class imbalance is handled primarily by the inverse-frequency
    # weighted loss (losses.py). The SMOTE utility in data.py (training only,
    # with a physiological-plausibility filter) reproduces the per-task
    # oversampling experiments described in the manuscript.
    use_smote=False,
)

BOOTSTRAP_N = 1000            # bootstrap iterations for 95% CIs
