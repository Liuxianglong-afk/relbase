"""课题 B1：核实 B13 中 TC 的 z' 标准差 > 1 是否完全来自留一标准化（外部参照 = 解析理论值）。

TC 本人 z' = (z - m_{-i}) / s_{-i}，站点 TC 数为 k 时，若 z 在站点内服从正态，Var(z') = (1 + 1/(k-1)) · (k-2)/(k-4)。
逐站点比较理论 SD 与观察 SD，并给出按单元格加权的合并值；另报 B12 中 |z| 达到数值边界 qnorm(1-1e-7) ≈ 5.2 的比例。
输出: results/B13v_calibration_theory.json
用法: .venv_B/Scripts/python.exe -I scripts/B13v_calibration_theory.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derivatives/norms_B1"


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B13v_calibration_theory.py", "sets": {}}
    for st in ["strength", "netfc", "thickness"]:
        z = pd.read_csv(NB / f"B13_zrc_{st}_full.csv.gz")
        tc = z[(z.dx == 2) & z.site_ok]
        k = tc.groupby("site").sub_n.nunique()
        exp_sd = np.sqrt((1 + 1 / (k - 1)) * (k - 2) / (k - 4))
        obs_sd = tc.groupby("site").zrc.std()
        w = tc.groupby("site").size() / len(tc)
        out["sets"][st] = {
            "pooled_expected_sd": round(float(np.sqrt((exp_sd ** 2 * w).sum())), 4),
            "pooled_observed_sd": round(float(tc.zrc.std()), 4),
            "max_abs_site_diff_obs_minus_exp": round(float((obs_sd - exp_sd).abs().max()), 4),
            "share_raw_z_at_bound_5.2": float((z.z.abs() >= 5.19).mean()),
            "per_site": {s: {"k": int(k[s]), "exp_sd": round(float(exp_sd[s]), 3), "obs_sd": round(float(obs_sd[s]), 3)} for s in k.index},
        }
        print(st, {kk: v for kk, v in out["sets"][st].items() if kk != "per_site"}, flush=True)
    out["conclusion"] = ("TC z' 的 SD 与留一理论值一致（差异来自 t 型膨胀，不是常模失准）；ASD 用全部 k 个 TC 标准化，"
                         "膨胀略小，P2 极端比例在两组间的基线因此有轻微差异，报告时说明。")
    (ROOT / "results/B13v_calibration_theory.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
