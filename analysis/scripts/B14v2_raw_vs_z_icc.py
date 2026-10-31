"""课题 B1：核实 B14 的低 ICC 是否由常模/站点居中造成。对同一批被试，比较三种量的前后两半 ICC(2,1) 中位数:
  raw   B11 原始 FC 强度（未建模）
  rawc  原始强度按站点减去站点均值（只去站点差异）
  zrc   B13 站点居中后的 z（B14 主分析所用）
若 raw 本身就低，低信度来自数据长度/测量本身，而非建模步骤。
输出: results/B14v2_raw_vs_z_icc.json
用法: .venv_B/Scripts/python.exe -I scripts/B14v2_raw_vs_z_icc.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from B14_reliability import icc21, load_pair  # noqa: E402


def main():
    w1, w2 = load_pair("strength", "half1", "half2")
    meta = w1.index.to_frame(index=False)
    raw = {c: pd.read_csv(ROOT / f"derivatives/features_B/B11_strength_{c}.csv").set_index(["cohort", "sub_n"]) for c in ["half1", "half2"]}
    key = pd.MultiIndex.from_frame(meta[["cohort", "sub_n"]])
    r1, r2 = raw["half1"].loc[key, w1.columns].values, raw["half2"].loc[key, w1.columns].values
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B14v2_raw_vs_z_icc.py", "results": {}}
    for coh in ["abide1", "abide2"]:
        for g, lab in [(2, "TC"), (1, "ASD")]:
            sel = ((meta.cohort == coh) & (meta.dx == g)).values
            site = meta.site.values[sel]
            a, b = r1[sel], r2[sel]
            ac, bc = a.copy(), b.copy()
            for s in np.unique(site):
                ix = site == s
                m = np.vstack([a[ix], b[ix]]).mean(0)
                ac[ix] -= m
                bc[ix] -= m
            res = {"n": int(sel.sum()),
                   "raw": round(float(np.median(icc21(a, b))), 4),
                   "rawc_site_centered": round(float(np.median(icc21(ac, bc))), 4),
                   "zrc": round(float(np.median(icc21(w1.values[sel], w2.values[sel]))), 4)}
            out["results"][f"{coh}_{lab}"] = res
            print(coh, lab, res, flush=True)
    (ROOT / "results/B14v2_raw_vs_z_icc.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
