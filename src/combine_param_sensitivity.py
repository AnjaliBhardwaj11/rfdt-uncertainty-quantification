#!/usr/bin/env python3

import torch

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
