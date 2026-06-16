# Data format

The clinical echocardiographic data are **not** included in this repository
because they are protected by patient-privacy regulations and institutional
policies under Chinese healthcare data-protection laws. This file documents the
input schema so that the code can be run on appropriately approved data.

## Input CSV

Each row is one (de-identified) patient. The loader
(`feature_engineering.add_derived_features` + `data.order_features`) expects the
following columns.

### Raw measured columns
These are measured by certified sonographers (ASE/EACVI protocols). Derived
indices are computed automatically from them.

```
LVEDD, LVESD, IVSd, LVPWd, LVEDV, LVESV, SV,
MV_E, MV_A, e_septal, e_lateral, LAVI, DT,
AV_Vmax, AV_mean_grad, AVA, MV_Vmax, MR_jet_area, AR_PHT, MV_VTI,
TR_Vmax, PA_diam, RAP, TAPSE, RV_diam, IVC_diam, IVC_collapse,
Ao_root, LA_diam, PE_amount,
weight_kg, height_cm
```

### Derived columns (computed by the code; need not be supplied)
```
LVEF, FS, E_A_ratio, E_e_ratio, AV_peak_grad, RVSP, LVM, RWT, BSA
```
The 39 model inputs, in fixed clinical order, are defined in `config.FEATURE_ORDER`.

## Label columns

One column per task.

* 8 multi-class severity tasks — encoded `0..4`
  (`0=Mild, 1=Mild-Moderate, 2=Moderate, 3=Moderate-Severe, 4=Severe`):
  `mitral_regurgitation, tricuspid_regurgitation, aortic_regurgitation,
  aortic_stenosis, pulmonary_regurgitation, lv_diastolic_dysfunction,
  pulmonary_hypertension, lv_systolic_dysfunction`

* 9 binary tasks — encoded `0/1` (absent/present):
  `pulmonary_artery_pressure_abnormal, left_atrial_enlargement,
  right_atrial_enlargement, left_ventricular_enlargement,
  right_ventricular_enlargement, left_ventricular_hypertrophy,
  interventricular_septal_thickening, pericardial_effusion,
  regional_wall_motion_abnormality`

## Cohorts
* `train.csv` — 400 patients (training)
* `internal_test.csv` — 100 patients (internal test)
* `external.csv` — 102 patients (external validation)

SMOTE is applied to the training split only; the internal test and external
cohorts are never resampled.
