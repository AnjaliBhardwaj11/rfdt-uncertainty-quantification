# rfdt-uncertainty-quantification

This repository contains the reference implementation for the uncertainty
quantification framework proposed in:

**Sensitivity-Aware Uncertainty Quantification for RF Digital Twin Beam Predictions**

The code implements a modular, physics-aware pipeline for quantifying beam-wise
prediction uncertainty in RF Digital Twins (RF-DTs).

The focus of this repository is **uncertainty modeling and calibration**.
It does **not** reimplement the RF Digital Twin itself.

---

## What This Repository Provides

The code supports the following components:

- Structural uncertainty calibration using oracle multipath references
- Parametric uncertainty propagation via first-order sensitivity analysis
- Beam-wise uncertainty interval construction in the log-energy domain
- Empirical coverage evaluation across multipath richness levels
- Visualization of beam-wise uncertainty behavior

All uncertainty modeling is performed **on top of RF Digital Twin predictions**.

---

## Dependency on RF-3DGS (SunLab)

This repository assumes access to RF Digital Twin predictions generated using
**RF-3DGS**, developed by SunLab (University of Georgia).

The RF-3DGS implementation, along with example scene data and `.ply` files, can
be obtained from:

https://github.com/SunLab-UGA/RF-3DGS

You should use the SunLab repository to:
- Reconstruct the propagation environment
- Generate RF-DT beam-wise energy predictions
- Produce oracle multipath component (MPC) outputs

The outputs from RF-3DGS are then consumed by the uncertainty pipeline provided
in this repository.
---

## Data Directory

The `data/` directory is expected to contain intermediate artifacts produced by
RF-3DGS and by different stages of the uncertainty pipeline, such as:

- RF-DT beam-wise predicted energies
- Oracle multipath component (MPC) tensors
- Sensitivity tensors and intermediate uncertainty terms

See `data/README.md` for details on expected file formats and naming conventions.

Large datasets and scene reconstructions are **not** included in this repository.

---

## Pipeline Overview

The intended execution flow is as follows:

1. Generate RF-DT beam-wise energy predictions using RF-3DGS
2. Compute parametric sensitivities via automatic differentiation
3. Fit structural uncertainty quantiles using oracle multipath references
4. Combine structural and parametric uncertainty components
5. Construct beam-wise uncertainty intervals
6. Visualize empirical coverage and beam-wise uncertainty behavior

Each Python script in the repository corresponds to a specific stage in this
pipeline.

---

## Reproducibility Notes

- All uncertainty modeling is performed in the **log-energy domain**
- Structural uncertainty is calibrated empirically (distribution-free)
- Parametric uncertainty is propagated using first-order sensitivity analysis
- The oracle model is used **only offline** for calibration and validation

Exact reproduction requires access to RF-3DGS outputs generated using the same
scene, beam codebook, and oracle multipath configuration.

---

## Citation

If you use this code or build upon it, please cite it.

---

## License

This code is provided for research and academic use.
Please respect the license terms of the RF-3DGS repository for upstream components.
