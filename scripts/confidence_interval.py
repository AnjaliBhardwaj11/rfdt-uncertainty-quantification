#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12

AZ_ORACLE = 72
EL_ORACLE = 24
AZ_DT = 33
EL_DT = 11

K_LIST = [50, 100, 200, 500, 8000]
GAMMA_LIST = [1.0, 0.8, 0.6, 0.4, 0.2]
K_PLOT = 200   # representative oracle fidelity for visualization

print(f"\nDT beam grid:")
print(f"  AZ_DT = {AZ_DT}")
print(f"  EL_DT = {EL_DT}")
print(f"  Total DT beams = {AZ_DT * EL_DT}")

# --------------------------------------------------
# Beam angle helpers
# --------------------------------------------------
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

# --------------------------------------------------
# Load DT prediction and uncertainty
# --------------------------------------------------
bound = torch.load("dt_uncertainty_beam_sensitivity.pt", map_location=DEVICE)
E_hat   = bound["E_hat"]
q_total = bound["q_total"]

# Log-domain quantities (for plotting)
z_hat = torch.log(E_hat + EPS)

# --------------------------------------------------
# Load oracle MPC data
# --------------------------------------------------
mpc = torch.load("rx0_mpc.pt", map_location=DEVICE)
c2 = mpc["c2"]

num_mpcs, _ = c2.shape
ang_oracle = beam_angles(AZ_ORACLE, EL_ORACLE, DEVICE)
ang_dt     = beam_angles(AZ_DT, EL_DT, DEVICE)

mpc_power = c2.sum(dim=1)
mpc_idx   = torch.argsort(mpc_power, descending=True)

# --------------------------------------------------
# Sort beams by predicted energy
# --------------------------------------------------
order = torch.argsort(z_hat, descending=True)
z_hat_s = z_hat[order]
q_s = q_total[order]

# --------------------------------------------------
# Gamma sweep
# --------------------------------------------------
results = []

for gamma in GAMMA_LIST:

    print(f"\n==============================")
    print(f"Uncertainty scale γ = {gamma:.2f}")
    print(f"==============================")

    # Interval bounds (log domain)
    z_lo = z_hat - gamma * q_total
    z_hi = z_hat + gamma * q_total

    hit_count = torch.zeros_like(z_hat)

    for K in K_LIST:
        K_eff = min(K, num_mpcs)
        c2_K = c2[mpc_idx[:K_eff]]
        E_oracle_fine = c2_K.sum(dim=0)

        E_oracle_dt = project_oracle_to_dt(
            E_oracle_fine, ang_oracle, ang_dt
        )
        z_oracle = torch.log(E_oracle_dt + EPS)

        inside = (z_oracle >= z_lo) & (z_oracle <= z_hi)
        hit_count += inside.float()

        coverage = inside.float().mean().item()
        max_violation = torch.abs(
            z_oracle - torch.clamp(z_oracle, z_lo, z_hi)
        ).max().item()

        print(
            f"MPC K = {K_eff:5d} | "
            f"coverage = {coverage:.4f} | "
            f"max viol = {max_violation:.2e}"
        )

    confidence_beam = hit_count / len(K_LIST)

    print("Beam-wise confidence:")
    print(f"  Min : {confidence_beam.min().item():.3f}")
    print(f"  Mean: {confidence_beam.mean().item():.3f}")
    print(f"  Max : {confidence_beam.max().item():.3f}")

    # --------------------------------------------------
    # MAIN FIGURE: prediction + uncertainty + oracle
    # --------------------------------------------------
    # Oracle for representative K
    c2_K = c2[mpc_idx[:min(K_PLOT, num_mpcs)]]
    E_oracle_dt = project_oracle_to_dt(
        c2_K.sum(dim=0), ang_oracle, ang_dt
    )
    z_oracle_s = torch.log(E_oracle_dt + EPS)[order]

    z_lo_s = z_lo[order]
    z_hi_s = z_hi[order]

    # Highlight top 10% uncertainty beams
    interval_width = 2 * gamma * q_s
    thresh = torch.quantile(interval_width, 0.90)
    high_unc = interval_width >= thresh
    idx = torch.where(high_unc)[0].cpu().numpy()

    plt.figure(figsize=(11, 4.5))
    x = np.arange(len(z_hat_s))

    plt.plot(x, z_hat_s.detach().cpu(), lw=2, label="Predicted log $\\hat{E}_b$")
    plt.fill_between(
        x,
        z_lo_s.detach().cpu(),
        z_hi_s.detach().cpu(),
        alpha=0.30,
        label="Uncertainty interval"
    )
    plt.scatter(
        x,
        z_oracle_s.detach().cpu(),
        s=14,
        c="red",
        label=rf"Oracle log $E_b^{{(K={K_PLOT})}}$"
    )

    # Shade high-uncertainty regions
    if len(idx) > 0:
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                plt.axvspan(start, prev, color="gray", alpha=0.15)
                start = i
                prev = i
        plt.axvspan(start, prev, color="gray", alpha=0.15)

    plt.plot([], [], color="gray", alpha=0.3, label="High-uncertainty beams")

    plt.xlabel("Beam index (sorted by predicted energy)")
    plt.ylabel("Log energy")
    plt.title(
        rf"Sensitivity-aware log-energy uncertainty (K={K_PLOT}, $\gamma$={gamma:.1f})"
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    results.append({
        "gamma": gamma,
        "confidence_beam": confidence_beam.detach().cpu()
    })

# --------------------------------------------------
# Save results
# --------------------------------------------------
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
