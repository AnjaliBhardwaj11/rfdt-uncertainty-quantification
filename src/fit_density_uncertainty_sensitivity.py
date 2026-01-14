#!/usr/bin/env python3
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AZ_REF = 72
EL_REF = 24

K_LIST = [50, 100, 200, 500, 8000]

dt = torch.load("dt_outputs.pt", map_location=DEVICE)
E_hat = dt["E_hat"]
B = E_hat.numel()

mpc = torch.load("rx0_mpc.pt", map_location=DEVICE)
c2 = mpc["c2"]

num_mpcs, B_check = c2.shape
assert B_check == B, "Beam count mismatch between DT and MPC oracle"

mpc_power = c2.sum(dim=1)
mpc_idx = torch.argsort(mpc_power, descending=True)


sens = torch.load(
    "param_sensitivity_total_az72_el24.pt",
    map_location=DEVICE
)

var_pos = sens["var_logE_total"]
assert var_pos.numel() == B, "Sensitivity beam count mismatch"

S = torch.sqrt(var_pos)
S_med = torch.median(S)

if S_med <= 0:
    raise RuntimeError("Median phase sensitivity is zero")

s = S / S_med

eps_all = []

for K in K_LIST:
    K_eff = min(K, num_mpcs)

    c2_K = c2[mpc_idx[:K_eff]]

 
    E_oracle_K = c2_K.sum(dim=0)

    eps_raw = torch.log(
        (E_oracle_K + 1e-12) /
        (E_hat + 1e-12)
    )

    eps_norm = eps_raw / s

    eps_all.append(eps_norm)

eps_all = torch.cat(eps_all)

mu = torch.median(eps_all)

Q = 0.85                                 # unchanged
q_beam = torch.quantile(torch.abs(eps_all - mu), Q)

params = {
    "mu": mu.item(),
    "q_beam": q_beam.item(),
    "quantile": Q,
    "K_list": K_LIST,
    "AZ_REF": AZ_REF,
    "EL_REF": EL_REF,
    "calibration": "sensitivity-normalized",
}

torch.save(params, "beam_uncertainty_params_sensitivity.pt")

print("Saved beam_uncertainty_params_sensitivity.pt")
print("Calibration summary:")
print(f"  mu        = {mu.item():.4e}")
print(f"  q_beam    = {q_beam.item():.4e}")
print(f"  quantile  = {Q}")
print(f"  K_list    = {K_LIST}")
print(f"  median(s) = {torch.median(s).item():.3f}")
