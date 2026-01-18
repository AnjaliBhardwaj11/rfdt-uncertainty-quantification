#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AZ = 33
EL = 11

POS_FILE = f"param_sensitivity_position_az{AZ}_el{EL}.pt"
OPA_FILE = f"param_sensitivity_opacity_az{AZ}_el{EL}.pt"
SH_FILE  = f"param_sensitivity_sh_az{AZ}_el{EL}.pt"

OUT_FILE = f"param_sensitivity_total_az{AZ}_el{EL}.pt"

sens_pos = torch.load(POS_FILE, map_location=DEVICE)["var_logE"]
sens_opa = torch.load(OPA_FILE, map_location=DEVICE)["var_logE"]
sens_sh  = torch.load(SH_FILE,  map_location=DEVICE)["var_logE"]

B = AZ * EL
assert sens_pos.numel() == B, "Position sensitivity beam count mismatch"
assert sens_opa.numel() == B, "Opacity sensitivity beam count mismatch"
assert sens_sh.numel()  == B, "SH sensitivity beam count mismatch"

var_total = sens_pos + sens_opa + sens_sh
s_total = torch.sqrt(var_total)

torch.save(
    {
        "var_logE_total": var_total,
        "s_total": s_total,
        "AZ": AZ,
        "EL": EL,
        "components": {
            "position": sens_pos,
            "opacity": sens_opa,
            "sh": sens_sh
        }
    },
    OUT_FILE
)

print(f"Saved {OUT_FILE}")
print("---------------------------------")
print("Total sensitivity stats:")
print(f"  min(s_b): {s_total.min().item():.3e}")
print(f"  med(s_b): {s_total.median().item():.3e}")
print(f"  max(s_b): {s_total.max().item():.3e}")

# ==================================================
# Beam-wise parametric sensitivity comparison
# ==================================================


std_pos = torch.sqrt(sens_pos).detach().cpu()
std_sh  = torch.sqrt(sens_sh).detach().cpu()
std_opa = torch.sqrt(sens_opa).detach().cpu()

# Optional: sort beams by dominant (position) sensitivity
order = torch.argsort(std_pos, descending=True)
std_pos = std_pos[order]
std_sh  = std_sh[order]
std_opa = std_opa[order]

plt.figure(figsize=(10, 4))
plt.plot(std_pos, label="Position", linewidth=2)
plt.plot(std_sh,  label="SH (l ≤ 6)", linewidth=2)
plt.plot(std_opa, label="Opacity", linewidth=2)

plt.xlabel("Beam index")
plt.ylabel("Std of log(E)")
plt.title("Beam-wise parametric sensitivity comparison (AZ=33, EL=11)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

