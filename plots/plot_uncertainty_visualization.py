#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AZ_DT = 33
EL_DT = 11

K_PLOT = 8000
GAMMA  = 1.0

dt = torch.load("dt_outputs.pt", map_location=DEVICE)
E_hat = dt["E_hat"]

unc = torch.load(
    "dt_uncertainty_beam_sensitivity.pt",
    map_location=DEVICE
)

q_total = unc["q_total"]
mu = torch.tensor(0.0, device=DEVICE)

E_lo = E_hat * torch.exp(mu - GAMMA * q_total)
E_hi = E_hat * torch.exp(mu + GAMMA * q_total)

mpc = torch.load("rx0_mpc.pt", map_location=DEVICE)
c2 = mpc["c2"]

mpc_power = c2.sum(dim=1)
mpc_idx = torch.argsort(mpc_power, descending=True)

c2_K = c2[mpc_idx[:K_PLOT]]
E_oracle = c2_K.sum(dim=0)

order = torch.argsort(E_hat, descending=True)

E_hat_s     = E_hat[order]
E_lo_s      = E_lo[order]
E_hi_s      = E_hi[order]
E_oracle_s  = E_oracle[order]

E_hat_p = E_hat_s.detach().cpu().numpy()
E_lo_p  = E_lo_s.detach().cpu().numpy()
E_hi_p  = E_hi_s.detach().cpu().numpy()
E_or_p  = E_oracle_s.detach().cpu().numpy()

x = np.arange(len(E_hat_p))

plt.figure(figsize=(11, 5))

plt.plot(
    x, E_hat_p,
    lw=2,
    label="Predicted $\\hat{E}_b$"
)

plt.fill_between(
    x,
    E_lo_p,
    E_hi_p,
    alpha=0.3,
    label="Uncertainty interval"
)

plt.scatter(
    x,
    E_or_p,
    s=12,
    c="red",
    label=f"Oracle $E_b^{{(K={K_PLOT})}}$"
)

plt.yscale("log")
plt.xlabel("Beam index (sorted by predicted energy)")
plt.ylabel("Energy")
plt.title(
    f"Beam-wise uncertainty visualization "
    f"(K={K_PLOT}, γ={GAMMA})"
)

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
