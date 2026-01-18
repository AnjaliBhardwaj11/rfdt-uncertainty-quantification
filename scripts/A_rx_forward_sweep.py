#!/usr/bin/env python3
import torch
import numpy as np
from plyfile import PlyData
from scipy.special import sph_harm_y as sph_harm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- RF constants ----------------
c0 = 3e8
FC = 28e9
lam = c0 / FC
k0 = 2 * np.pi / lam

NX = 8
NY = 8

# ---------------- Single RX ----------------
RX = torch.tensor([0.0, 0.0, 1.5], device=DEVICE)

# ---------------- Paths ----------------
PLY_PATH = (
    "/home/anjali/Documents/Anjali_AML_Project/RF/"
    "RF-3DGS/output/rf-3dgs_MPC_test/"
    "point_cloud/iteration_40000/point_cloud.ply"
)

# ---------------- Utilities ----------------
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

    return means, alpha, sh


def sh_gain(sh, dirs):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = torch.acos(torch.clamp(z, -1, 1))
    phi = torch.atan2(y, x)

    sh_np = sh.cpu().numpy()
    th = theta.cpu().numpy()
    ph = phi.cpu().numpy()

    out = np.zeros(len(th), dtype=np.complex64)
    idx = 0
    L = int(np.sqrt(sh_np.shape[1]) - 1)

    for l in range(L + 1):
        for m in range(-l, l + 1):
            out += sh_np[:, idx] * sph_harm(m, l, ph, th)
            idx += 1

    return torch.tensor(np.abs(out), device=DEVICE)


def generate_beams():
    d = lam / 2
    az = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    el = np.linspace(0, np.pi / 2, 24)

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

    return torch.tensor(beams, dtype=torch.complex64, device=DEVICE)


# ---------------- Main ----------------
means, alpha, sh = load_ply(PLY_PATH)
beams = generate_beams()

coords = torch.tensor(
    [[i * lam / 2, j * lam / 2, 0.0] for i in range(NX) for j in range(NY)],
    device=DEVICE
)

vec = means - RX
dist = torch.linalg.norm(vec, dim=1)
dirs = vec / dist[:, None]

g = sh_gain(sh, dirs)
field = torch.sqrt(alpha) * g * torch.exp(-1j * k0 * dist)

steer = torch.exp(1j * k0 * (dirs @ coords.T))
proj = steer @ beams.conj().T

# Incoherent MPC power aggregation
c2 = torch.abs(field[:, None] * proj) ** 2
E_full = c2.sum(dim=0)

torch.save(
    {
        "E_full": E_full,
        "c2": c2
    },
    "rx0_mpc.pt"
)

print("Saved rx0_mpc.pt")
