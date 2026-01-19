#!/usr/bin/env python3
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dt = torch.load("dt_outputs.pt", map_location=DEVICE)
E_hat = dt["E_hat"]                    # [B]
B = E_hat.numel()

p = torch.load("beam_uncertainty_params_sensitivity.pt")

mu     = torch.tensor(p["mu"], device=DEVICE)
q_beam = torch.tensor(p["q_beam"], device=DEVICE)

AZ_REF = p["AZ_REF"]
EL_REF = p["EL_REF"]

AZ_DT = 33
EL_DT = 11

dphi_ref   = 2.0 * np.pi / AZ_REF
dtheta_ref = (np.pi / 2.0) / EL_REF

dphi   = 2.0 * np.pi / AZ_DT
dtheta = (np.pi / 2.0) / EL_DT

ALPHA_PHI   = 0.9
ALPHA_THETA = 0.2

inflation = (
    1.0
    + ALPHA_PHI   * abs(np.log(dphi / dphi_ref))
    + ALPHA_THETA * abs(np.log(dtheta / dtheta_ref))
)

inflation = torch.tensor(inflation, device=DEVICE)

sens = torch.load(
    f"param_sensitivity_total_az{AZ_DT}_el{EL_DT}.pt",
    map_location=DEVICE
)

var_sen = sens["var_logE_total"]
assert var_sen.numel() == B, "Beam count mismatch"

S = torch.sqrt(var_sen)
S_med = torch.median(S)

if S_med <= 0:
    raise RuntimeError("Median phase sensitivity is zero")

s = S / S_med

q_total = q_beam * inflation * s

E_lo = E_hat * torch.exp(- q_total)
E_hi = E_hat * torch.exp(+ q_total)

torch.save(
    {
        "E_hat": E_hat,
        "E_lo":  E_lo,
        "E_hi":  E_hi,
        "q_total": q_total,
        "inflation": inflation,
        "sensitivity_scale": s,
    },
    "dt_uncertainty_beam_sensitivity.pt"
)

print("Saved dt_uncertainty_beam_sensitivity.pt")
print("-----------------------------------------")
print(f"Median sensitivity scale (should be ~1): {torch.median(s).item():.3f}")
print()
print("q_total stats:")
print(f"  min: {q_total.min().item():.3e}")
print(f"  med: {q_total.median().item():.3e}")
print(f"  max: {q_total.max().item():.3e}")
