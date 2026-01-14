# rfdt-uncertainty-quantification

This repository contains the reference implementation for the uncertainty quantification framework proposed in:

**"Sensitivity-Aware Uncertainty Quantification for RF Digital Twin Beam Predictions"**

The code provides a modular pipeline for:
- Structural uncertainty calibration using oracle multipath references
- Parametric uncertainty propagation via first-order sensitivity analysis
- Beam-wise uncertainty interval construction in the log-energy domain
- Visualization and validation of empirical coverage and beam-wise behavior

This repository operates **on top of RF Digital Twin predictions** and does not reimplement the RF-DT itself.

---

## Dependencies

The RF Digital Twin forward model and scene reconstruction are based on **RF-3DGS**, developed by SunLab.

The RF-3DGS implementation and example scene data can be obtained from: https://github.com/SunLab-UGA/RF-3DGS

This repository assumes access to:
- RF-DT beam-wise energy predictions
- Oracle multipath component (MPC) outputs

---

## Repository Structure
├── rfdt_forward.py
├── rfdt_param_sensitivity_and_plot_SH_l6.py
├── fit_density_uncertainty_sensitivity.py
├── apply_density_uncertainty_sensitivity.py
├── combine_param_sensitivity.py
├── confidence_interval.py
├── plot_uncertainty_visualization.py
├── plot_uncertainty_visualization_presentation.py
├── A_rx_forward_sweep.py
├── data/
│ └── README.md


---

## Pipeline Overview

The intended execution flow is:

1. Generate RF-DT beam-wise energy predictions (via RF-3DGS)
2. Compute parametric sensitivities using automatic differentiation
3. Fit structural uncertainty quantiles using oracle references
4. Combine structural and parametric uncertainty components
5. Construct beam-wise prediction intervals
6. Visualize coverage and beam-wise uncertainty behavior

Each script corresponds to a specific stage of this pipeline.

---

## Citation

If you use this code, please cite the associated paper.
