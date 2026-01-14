#!/usr/bin/env python3
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AZ_ORACLE = 72
EL_ORACLE = 24

AZ_DT = 33
EL_DT = 11

K_LIST = [50, 100, 200, 500, 8000]

GAMMA_LIST = [1.0, 0.8, 0.6, 0.4, 0.2]

print(f"\nDT beam grid:")
print(f"  AZ_DT = {AZ_DT}")
print(f"  EL_DT = {EL_DT}")
print(f"  Total DT beams = {AZ_DT * EL_DT}")

def beam_angles(num_az, num_el, device):
    az = np.linspace(-np.pi, np.pi, num_az, endpoint=False)
    el = np.linspace(0, np.pi / 2, num_el)
    AZ, EL = np.meshgrid(az, el, indexing="xy")
    return torch.tensor(
        np.stack([AZ.ravel(), EL.ravel()], axis=1),
        dtype=torch.float32,
        device=device
    )

def project_oracle_to_dt(E_oracle, ang_oracle, ang_dt):
    d_az = ang_oracle[:, None, 0] - ang_dt[None, :, 0]
    d_el = ang_oracle[:, None, 1] - ang_dt[None, :, 1]
    dist2 = d_az**2 + d_el**2
    idx = torch.argmin(dist2, dim=1)

    E_dt = torch.zeros(len(ang_dt), device=E_oracle.device)
    E_dt.index_add_(0, idx, E_oracle)
    return E_dt

bound = torch.load("dt_uncertainty_beam_sensitivity.pt", map_location=DEVICE)

E_hat   = bound["E_hat"]
q_total = bound["q_total"]

params = torch.load("beam_uncertainty_params_sensitivity.pt")
mu = torch.tensor(params["mu"], device=DEVICE)

B_DT = E_hat.shape[0]

mpc = torch.load("rx0_mpc.pt", map_location=DEVICE)
c2 = mpc["c2"]

num_mpcs, B_oracle = c2.shape

ang_oracle = beam_angles(AZ_ORACLE, EL_ORACLE, DEVICE)
ang_dt     = beam_angles(AZ_DT, EL_DT, DEVICE)

mpc_power = c2.sum(dim=1)
mpc_idx   = torch.argsort(mpc_power, descending=True)

results = []

for gamma in GAMMA_LIST:

    print(f"\n==============================")
    print(f"Uncertainty scale γ = {gamma:.2f}")
    print(f"==============================")

    E_lo = E_hat * torch.exp(mu - gamma * q_total)
    E_hi = E_hat * torch.exp(mu + gamma * q_total)

    hit_count = torch.zeros(B_DT, device=DEVICE)

    for K in K_LIST:
        K_eff = min(K, num_mpcs)

        c2_K = c2[mpc_idx[:K_eff]]

        E_oracle_fine = c2_K.sum(dim=0)

        E_oracle_dt = project_oracle_to_dt(
            E_oracle_fine, ang_oracle, ang_dt
        )

        inside = (E_oracle_dt >= E_lo) & (E_oracle_dt <= E_hi)

        coverage = inside.float().mean().item()
        hit_count += inside.float()

        max_violation = torch.abs(
            E_oracle_dt -
            torch.clamp(E_oracle_dt, E_lo, E_hi)
        ).max().item()

        print(f"MPC K = {K_eff:5d} | coverage = {coverage:.4f} | max viol = {max_violation:.2e}")

    confidence_beam = hit_count / len(K_LIST)

    print("Beam-wise confidence:")
    print(f"  Min : {confidence_beam.min().item():.3f}")
    print(f"  Mean: {confidence_beam.mean().item():.3f}")
    print(f"  Max : {confidence_beam.max().item():.3f}")

    results.append({
        "gamma": gamma,
        "confidence_beam": confidence_beam
    })

torch.save(
    {
        "results": results,
        "GAMMA_LIST": GAMMA_LIST,
        "K_LIST": K_LIST,
        "AZ_DT": AZ_DT,
        "EL_DT": EL_DT
    },
    "beam_confidence_gamma_sweep.pt"
)

print("\nSaved beam_confidence_gamma_sweep.pt")
print("Gamma sweep validation complete.")
