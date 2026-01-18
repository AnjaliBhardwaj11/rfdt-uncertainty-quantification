# RF Digital Twin Uncertainty Modeling (Sensitivity-Aware)

This repository contains the code accompanying our work on **sensitivity-aware uncertainty modeling for RF Digital Twins (RFDTs)** using RF-3D Gaussian Splatting (RF-3DGS).  
The objective is to **quantify beam-wise predictive uncertainty** by propagating physically meaningful parameter sensitivities of the RFDT.

The pipeline is explicitly divided into **offline calibration** and **online deployment / evaluation**, reflecting practical RFDT usage.

---

## Repository Structure and Pipeline Overview

The overall workflow consists of the following stages:

1. **RFDT forward prediction**
2. **Parametric sensitivity analysis (offline)**
3. **Sensitivity aggregation**
4. **Uncertainty fitting and propagation**
5. **Oracle-based validation**

---

## Script Descriptions and Execution Order

### A. RFDT Forward Modeling

#### `rfdt_forward.py`
**Purpose**  
Runs the RF Digital Twin forward model to predict beam-wise received energy on the DT beam grid.

**Outputs**
- `dt_outputs.pt`
  - `E_hat`: predicted beam-wise energy (DT resolution)

**When to run**
- Once per trained RF-3DGS scene
- Required by all downstream scripts

---

#### `A_rx_forward_sweep.py`
**Purpose**  
Computes oracle MPC-based beam energies by aggregating individual MPC contributions for a given receiver location.

**Outputs**
- `rx0_mpc.pt`
  - `c2`: per-MPC, per-beam power contributions

**When to run**
- Required for uncertainty validation
- Can be rerun for different receiver locations

---

### B. Parametric Sensitivity Analysis (Offline Calibration)

These steps are **computationally expensive** and should be run **once per RFDT model**.

---

#### `rfdt_param_sensitivity_and_plot_SH_l6.py`
**Purpose**  
Computes beam-wise sensitivity of predicted log-energy with respect to a selected RFDT parameter block.

**Parameter blocks (run separately)**
- `position`
- `sh` (spherical harmonic coefficients, l ≤ 6)
- `opacity`

**Outputs**
- `param_sensitivity_position_az33_el11.pt`
- `param_sensitivity_sh_az33_el11.pt`
- `param_sensitivity_opacity_az33_el11.pt`

Each file contains:
- `var_logE`: beam-wise variance of log-energy due to that parameter

**When to run**
- Run **three times**, once per parameter block

---

#### `combine_param_sensitivity.py`
**Purpose**  
Aggregates individual parameter sensitivities and produces a **beam-wise parametric dominance plot**.

**Inputs**
- `param_sensitivity_position_az33_el11.pt`
- `param_sensitivity_sh_az33_el11.pt`
- `param_sensitivity_opacity_az33_el11.pt`

**Outputs**
- `param_sensitivity_total_az33_el11.pt`
- Plot: *Beam-wise parametric sensitivity comparison*

This plot demonstrates that:
- Position perturbations dominate uncertainty
- SH coefficients contribute secondary effects
- Opacity has negligible impact

**When to run**
- Once after all three sensitivity files are available

---

### C. Uncertainty Modeling

#### `fit_density_uncertainty_sensitivity.py`
**Purpose**  
Fits a mapping from combined sensitivity magnitude to uncertainty scale parameters.

**Outputs**
- `beam_uncertainty_params_sensitivity.pt`

**When to run**
- Offline calibration
- Once per RFDT model

---

#### `apply_density_uncertainty_sensitivity.py`
**Purpose**  
Applies the fitted uncertainty model to DT predictions to generate beam-wise uncertainty scales.

**Outputs**
- `dt_uncertainty_beam_sensitivity.pt`
  - `E_hat`: predicted energy
  - `q_total`: beam-wise uncertainty scale

**When to run**
- Once after fitting
- Used during online validation

---

### D. Online Validation and Visualization

#### `confidence_interval.py`
**Purpose**  
Validates sensitivity-aware uncertainty intervals against oracle MPC-based beam energies.

**What it performs**
- Projects oracle energies onto DT beam grid
- Evaluates empirical coverage vs oracle MPC count
- Computes beam-wise confidence
- Visualizes:
  - Sensitivity-aware uncertainty intervals
  - Beam-wise uncertainty width

**Inputs**
- `dt_uncertainty_beam_sensitivity.pt`
- `rx0_mpc.pt`

**When to run**
- Online / evaluation stage
- Can be rerun with different MPC budgets or uncertainty scales

---

## Offline vs Online Summary

### Offline (Run Once per Scene)
- `rfdt_forward.py`
- `rfdt_param_sensitivity_and_plot_SH_l6.py` (×3: position, SH, opacity)
- `combine_param_sensitivity.py`
- `fit_density_uncertainty_sensitivity.py`
- `apply_density_uncertainty_sensitivity.py`

### Online / Evaluation
- `A_rx_forward_sweep.py`
- `confidence_interval.py`

---

## Key Experimental Outputs

The core results of this repository are:

1. **Beam-wise parametric sensitivity comparison**  
   (from `combine_param_sensitivity.py`)

2. **Sensitivity-aware uncertainty interval visualization**  
   (from `confidence_interval.py`)

Together, these demonstrate physically grounded, beam-dependent uncertainty modeling in RF Digital Twins.

---

## Notes

- DT beam grid resolution: **AZ = 33, EL = 11**
- SH expansion truncated at **l ≤ 6**
- Uncertainty is modeled in the **log-energy domain**
- Assumes a trained RF-3DGS point cloud as input

---

## Citation

If you use this code, please cite the corresponding paper (to be updated upon publication).

---

This repository is intentionally focused on **interpretability and physical consistency** rather than system-level performance benchmarking.
