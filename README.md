# RF Digital Twin Uncertainty Modeling (Sensitivity-Aware)

This repository implements a **sensitivity-aware uncertainty modeling pipeline for RF Digital Twins (RFDTs)** built on RF-3D Gaussian Splatting (RF-3DGS).

The code is organized around **two clearly separated stages**:

- **Step 1: Offline calibration with high beam resolution**  
  (used to *learn* and *fit* uncertainty behavior)
- **Step 2: Deployment-time evaluation with DT beam resolution**  
  (used to *apply* and *validate* uncertainty)

The separation is intentional and reflects how an RFDT would be used in practice.

---

## High-Level Pipeline Summary

### Step 1: Offline Calibration (High Beam Resolution)

1. `A_rx_forward_sweep.py`  
   Generates oracle MPC-based beam energies.

2. `rfdt_forward.py`  
   Runs RFDT forward prediction.

3. `rfdt_param_sensitivity_and_plot_SH_l6.py`  
   Computes parametric sensitivities (position / SH / opacity).

4. `combine_param_sensitivity.py`  
   Aggregates sensitivities into total beam-wise uncertainty.

5. `fit_density_uncertainty_sensitivity.py`  
   Fits uncertainty scaling parameters.

### Step 2: Deployment & Validation (DT Beam Resolution)

1. `rfdt_forward.py`  
   Runs the RF Digital Twin forward model at the **deployment beam resolution**, producing beam-wise predicted energy.

2. `rfdt_param_sensitivity_and_plot_SH_l6.py`  
   Computes **beam-wise parametric sensitivities** (position / SH / opacity) at the deployment beam grid using first-order differentiation.

3. `combine_param_sensitivity.py`  
   Aggregates individual parameter sensitivities into a **total beam-wise sensitivity / uncertainty proxy** for the deployment setting.

4. `apply_density_uncertainty_sensitivity.py`  
   Applies the **uncertainty scaling learned during offline calibration** to obtain beam-wise uncertainty scales for deployment.

5. `confidence_interval.py`  
   Validates the sensitivity-aware uncertainty intervals against **oracle MPC-based beam energies**, reporting coverage, confidence statistics, and producing the main evaluation plots.

---

## Detailed Explanation of Each Step

## Step 1: Offline Calibration (High Beam Resolution)

This step is run **once per environment / RF-3DGS model**.  
It is computationally expensive and should not be repeated during deployment.

### 1. `A_rx_forward_sweep.py`
**Purpose**  
Computes *oracle* beam energies using a **high-resolution beam grid** by aggregating individual MPC contributions.

**Role in pipeline**
- Acts as the *ground-truth oracle*
- Provides fine-grained beam energy statistics

**Output**
- `rx0_mpc.pt`
  - `c2`: per-MPC, per-beam power contributions

---

### 2. `rfdt_forward.py`
**Purpose**  
Runs the RF Digital Twin forward model to predict beam-wise received energy.

**Role in pipeline**
- Generates RFDT predictions corresponding to the oracle beams

**Output**
- `dt_outputs.pt`
  - `E_hat`: predicted beam-wise energy

---

### 3. `rfdt_param_sensitivity_and_plot_SH_l6.py`
**Purpose**  
Computes **beam-wise parametric sensitivity** of predicted log-energy.

**How it works**
- Uses automatic differentiation
- Perturbs one parameter block at a time

**Parameter blocks (run separately)**
- `position`
- `sh` (spherical harmonics, l ≤ 6)
- `opacity`

**Outputs**
- `param_sensitivity_position_az33_el11.pt`
- `param_sensitivity_sh_az33_el11.pt`
- `param_sensitivity_opacity_az33_el11.pt`

Each file contains:
- `var_logE`: variance of log-energy per beam

---

### 4. `combine_param_sensitivity.py`
**Purpose**  
Combines individual parameter sensitivities into a **total beam-wise sensitivity**.

**Role in pipeline**
- Aggregates contributions from:
  - Geometry (position)
  - Angular scattering (SH)
  - Attenuation (opacity)

**Outputs**
- `param_sensitivity_total_az33_el11.pt`
- Beam-wise parametric sensitivity comparison plot

This plot shows that **position dominates uncertainty**, justifying beam-wise modeling.

---

### 5. `fit_density_uncertainty_sensitivity.py`
**Purpose**  
Fits a mapping from total sensitivity magnitude to an uncertainty scale.

**Role in pipeline**
- Learns how sensitivity translates to predictive uncertainty
- Produces parameters reused during deployment

**Output**
- `beam_uncertainty_params_sensitivity.pt`

---

## Step 2: Deployment & Validation (DT Beam Resolution)

This step represents **online RFDT usage** at the deployment beam resolution.

---

### 1. `rfdt_forward.py`
**Purpose**  
Predicts beam-wise energy on the **DT beam grid**.

**Output**
- `dt_outputs.pt`

---

### 2. `rfdt_param_sensitivity_and_plot_SH_l6.py`
**Purpose**  
Computes parametric sensitivities **at deployment resolution**.

**Role in pipeline**
- Reuses the same sensitivity formulation
- Now evaluated on DT beams instead of high-resolution beams

**Outputs**
- `param_sensitivity_*.pt` (DT resolution)

---

### 3. `combine_param_sensitivity.py`
**Purpose**  
Aggregates sensitivities to obtain deployment-time beam-wise uncertainty proxies.

**Output**
- `param_sensitivity_total_az33_el11.pt`

---

### 4. `apply_density_uncertainty_sensitivity.py`
**Purpose**  
Applies the uncertainty model learned in Step 1 to DT predictions.

**Role in pipeline**
- Converts sensitivity magnitudes into uncertainty intervals

**Output**
- `dt_uncertainty_beam_sensitivity.pt`
  - `E_hat`: predicted energy
  - `q_total`: beam-wise uncertainty scale

---

### 5. `confidence_interval.py`
**Purpose**  
Validates sensitivity-aware uncertainty against oracle MPC-based energies.

**What it evaluates**
- Empirical coverage vs oracle MPC count
- Beam-wise confidence
- Sensitivity-aware uncertainty interval visualization
- Uncertainty width across beams

This script produces the **main validation figures** used in the paper.

---

## Key Design Principles

- **Offline / online separation**  
  Expensive calibration is done once; deployment remains lightweight.

- **Beam-wise uncertainty**  
  Uncertainty is not uniform and varies significantly across beams.

- **Physical interpretability**  
  Dominant uncertainty sources are explicitly identified (geometry > SH > opacity).

- **Log-energy domain modeling**  
  Ensures numerical stability and multiplicative uncertainty handling.

---

## Notes

- DT beam grid: **AZ = 33, EL = 11**
- Oracle beam grid: higher resolution (e.g., AZ = 72, EL = 24)
- SH expansion truncated at **l ≤ 6**
- Assumes a trained RF-3DGS point cloud as input

---

## Citation

If you use this code, please cite the corresponding paper (to be updated upon publication).

---

This repository is intentionally focused on **physically grounded uncertainty modeling** rather than end-to-end communication system optimization.
