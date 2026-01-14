## Data Directory

This directory is reserved for lightweight example data and intermediate artifacts produced by the uncertainty quantification pipeline proposed in this work.

Large scene assets, point clouds (.ply files), RF-3DGS models, and oracle ray-tracing outputs are not included in this repository due to size and licensing constraints.
Users interested in reproducing the RF Digital Twin (RF-DT) forward model can obtain the RF-3DGS implementation and example scene data from the official SunLab repository: https://github.com/SunLab-UGA/RF-3DGS
The scripts in this repository operate on beam-wise energy predictions produced by an RF-DT (e.g., RF-3DGS) and corresponding oracle references, and are written such that users can directly plug in predictions generated using the SunLab RF-3DGS pipeline or any equivalent RF Digital Twin model.
