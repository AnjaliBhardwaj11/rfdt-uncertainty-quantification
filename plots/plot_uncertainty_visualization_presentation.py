#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_PLOT = 200
GAMMA = 1

AZ_DT = 33
EL_DT = 11

dt = torch.load("dt_uncertainty_beam_sensitivity.pt", map_location=DEVICE)

E_hat = dt["E_hat"]
q_total = dt["q_total"]

mpc = torch.load("rx0_mpc.pt", map_location=DEVICE)
c2 = mpc["c2"]

num_mpcs, B_oracle = c2.shape

mpc_power = c2.sum(dim=1)
mpc_idx = torch.argsort(mpc_power, descending=True)

K_eff = min(K_PLOT, num_mpcs)
E_oracle = c2[mpc_idx[:K_eff]].sum(dim=0)

def beam_angles(num_az, num_el):
    az = np.linspace(-np.pi, np.pi, num_az, endpoint=False)
    el = np.linspace(0, np.pi / 2, num_el)
    AZ, EL = np.meshgrid(az, el, indexing="xy")
    return torch.tensor(
        np.stack([AZ.ravel(), EL.ravel()], axis=1),
        dtype=torch.float32,
        device=DEVICE
    )

ang_oracle = beam_angles(72, 24)
ang_dt = beam_angles(AZ_DT, EL_DT)

d_az = ang_oracle[:, None, 0] - ang_dt[None, :, 0]
d_el = ang_oracle[:, None, 1] - ang_dt[None, :, 1]
idx = torch.argmin(d_az**2 + d_el**2, dim=1)

E_oracle_dt = torch.zeros(E_hat.numel(), device=DEVICE)
E_oracle_dt.index_add_(0, idx, E_oracle)

log_E_hat = torch.log(E_hat + 1e-12)
log_E_lo = log_E_hat - GAMMA * q_total
log_E_hi = log_E_hat + GAMMA * q_total
log_E_orc = torch.log(E_oracle_dt + 1e-12)

order = torch.argsort(log_E_hat, descending=True)

log_E_hat = log_E_hat[order]
log_E_lo  = log_E_lo[order]
log_E_hi  = log_E_hi[order]
log_E_orc = log_E_orc[order]

x = np.arange(log_E_hat.numel())

log_E_hat = log_E_hat.detach().cpu().numpy()
log_E_lo  = log_E_lo.detach().cpu().numpy()
log_E_hi  = log_E_hi.detach().cpu().numpy()
log_E_orc = log_E_orc.detach().cpu().numpy()

# PLOT

plt.figure(figsize=(11, 4.5))

plt.plot(
    x,
    log_E_hat,
    linewidth=2.2,
    label=r"Predicted $\log \hat{E}_b$"
)

plt.fill_between(
    x,
    log_E_lo,
    log_E_hi,
    alpha=0.30,
    label="Uncertainty interval"
)

plt.scatter(
    x,
    log_E_orc,
    s=14,
    color="red",
    alpha=0.75,
    label=rf"Oracle $\log E_b^{{(K={K_PLOT})}}$"
)

plt.xlabel("Beam index (sorted by predicted energy)")
plt.ylabel("Log energy")
plt.title(
    rf"Sensitivity-aware log-energy uncertainty (K={K_PLOT}, $\gamma$={GAMMA})"
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
