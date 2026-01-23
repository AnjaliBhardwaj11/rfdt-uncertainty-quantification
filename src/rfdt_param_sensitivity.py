#!/usr/bin/env python3

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from plyfile import PlyData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AZ = 72
EL = 24
BEAM_BATCH_SIZE = 32

PARAM_BLOCK = "position"    # "position" | "opacity" | "sh"

SIGMA_POS = 0.02      # meters
SIGMA_OPACITY = 0.2   # log-opacity std
SIGMA_SH = 0.1        # SH coefficient std

EPS = 1e-12

# RF constants
c0 = 3e8
FC = 28e9
lam = c0 / FC
k0 = 2 * np.pi / lam

NX = 8
NY = 8
RX = torch.tensor([0.0, 0.0, 1.5], device=DEVICE)

PLY_PATH = (
    "/home/anjali/Documents/Anjali_AML_Project/RF/"
    "RF-3DGS/output/rf-3dgs_MPC_test/"
    "point_cloud/iteration_40000/point_cloud.ply"
)

dt = torch.load("dt_outputs.pt", map_location=DEVICE)
E_hat = dt["E_hat"].to(DEVICE)
B = E_hat.numel()
assert B == AZ * EL, "Beam count mismatch"

def load_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"]

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=1),
        dtype=torch.float32, device=DEVICE
    )

    alpha = torch.tensor(v["opacity"], device=DEVICE).clamp_min(1e-12)

    sh_keys = [k for k in v.data.dtype.names if k.startswith("f_")]
    sh = torch.tensor(
        np.stack([v[k] for k in sh_keys], axis=1),
        dtype=torch.float32, device=DEVICE
    )

    assert sh.shape[1] >= 48

    return means, alpha, sh


means, alpha, sh = load_ply(PLY_PATH)

means.requires_grad_(PARAM_BLOCK == "position")
alpha.requires_grad_(PARAM_BLOCK == "opacity")
sh.requires_grad_(PARAM_BLOCK == "sh")


def double_factorial(n):
    if n <= 0:
        return 1.0
    out = 1.0
    for k in range(n, 0, -2):
        out *= k
    return out


def associated_legendre(l, m, x):
    P_mm = ((-1.0)**m) * double_factorial(2*m - 1) * (1 - x**2)**(m / 2)
    if l == m:
        return P_mm

    P_mmp1 = x * (2*m + 1) * P_mm
    if l == m + 1:
        return P_mmp1

    P_lm_prev = P_mm
    P_lm = P_mmp1

    for n in range(m + 2, l + 1):
        P_lm_next = (
            (2*n - 1) * x * P_lm - (n + m - 1) * P_lm_prev
        ) / (n - m)
        P_lm_prev = P_lm
        P_lm = P_lm_next

    return P_lm


def real_sh_l3(sh, dirs):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x)
    cos_t = torch.cos(theta)

    Y = []

    # l = 0 (DC)
    l = 0
    m = 0
    K = math.sqrt(1 / (4 * math.pi))
    Y.append(torch.full_like(cos_t, K))

    # l = 1..3
    for l in range(1, 4):
        for m in range(-l, l + 1):
            P_lm = associated_legendre(l, abs(m), cos_t)
            K = math.sqrt(
                (2*l + 1) / (4*math.pi)
                * math.factorial(l - abs(m)) / math.factorial(l + abs(m))
            )

            if m > 0:
                Y_lm = math.sqrt(2) * K * P_lm * torch.cos(m * phi)
            elif m < 0:
                Y_lm = math.sqrt(2) * K * P_lm * torch.sin(abs(m) * phi)
            else:
                Y_lm = K * P_lm

            Y.append(Y_lm)

    # Stack SH basis: (N, 16)
    Y = torch.stack(Y, dim=1)

    # Combine 3 channels (exactly like RF-3DGS)
    # sh reshaped to (N, 3, 16)
    sh = sh.view(sh.shape[0], 3, 16)

    # Field-domain directional gain (sum over SH)
    g = (sh * Y[:, None, :]).sum(dim=2).norm(dim=1)

    return g


def generate_beams():
    d = lam / 2
    az = np.linspace(-np.pi, np.pi, AZ, endpoint=False)
    el = np.linspace(0, np.pi / 2, EL)

    beams = []
    for th in el:
        for ph in az:
            w = np.zeros((NX, NY), dtype=np.complex64)
            for i in range(NX):
                for j in range(NY):
                    w[i, j] = np.exp(
                        1j * k0 * d * (
                            i * np.sin(th) * np.cos(ph)
                            + j * np.sin(th) * np.sin(ph)
                        )
                    )
            w = w.reshape(-1)
            w /= np.linalg.norm(w)
            beams.append(w)

    return torch.from_numpy(np.stack(beams)).to(torch.complex64).to(DEVICE)


beams = generate_beams()

coords = torch.tensor(
    [[i * lam / 2, j * lam / 2, 0.0] for i in range(NX) for j in range(NY)],
    device=DEVICE
)


def forward_log_energy(beam_idx):
    vec = means - RX
    dist = torch.linalg.norm(vec, dim=1)
    dirs = vec / dist[:, None]

    g = real_sh_l3(sh, dirs)
    field = torch.sqrt(alpha) * g * torch.exp(-1j * k0 * dist)

    steer = torch.exp(1j * k0 * (dirs @ coords.T))
    proj = steer @ beams[beam_idx].conj().T

    y = (field[:, None] * proj).sum(dim=0)
    E = torch.abs(y) ** 2

    return torch.log(E + EPS)


var_logE = torch.zeros(B, device=DEVICE)
beam_ids = torch.arange(B, device=DEVICE)

for i in range(0, B, BEAM_BATCH_SIZE):
    batch = beam_ids[i:i + BEAM_BATCH_SIZE]

    logE = forward_log_energy(batch)

    for j, b in enumerate(batch):
        logE[j].backward(retain_graph=True)

        if PARAM_BLOCK == "position":
            grad = means.grad
            var = (grad ** 2).sum() * SIGMA_POS**2
            means.grad.zero_()

        elif PARAM_BLOCK == "opacity":
            grad = alpha.grad * alpha
            var = (grad ** 2).sum() * SIGMA_OPACITY**2
            alpha.grad.zero_()

        elif PARAM_BLOCK == "sh":
            grad = sh.grad[:, :48]
            var = (grad ** 2).sum() * SIGMA_SH**2
            sh.grad.zero_()

        var_logE[b] = var

out_file = f"param_sensitivity_{PARAM_BLOCK}_az{AZ}_el{EL}.pt"
torch.save({"var_logE": var_logE}, out_file)

print(f"Saved {out_file}")
print(f"Min var(logE): {var_logE.min().item():.3e}")
print(f"Max var(logE): {var_logE.max().item():.3e}")

plt.figure(figsize=(10, 4))
plt.plot(torch.sqrt(var_logE).detach().cpu().numpy(), linewidth=2)
plt.xlabel("Beam index")
plt.ylabel("Std(log E)")
plt.title(f"{PARAM_BLOCK.upper()} sensitivity (SH l≤6)")
plt.grid(True)
plt.tight_layout()
plt.show()
