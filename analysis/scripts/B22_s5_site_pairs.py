"""课题 B1 事后核查（2026-10-31，登记于方案第 12 节）：S5 的 ASD > TC 个体间重叠是否来自站点居中方式。

机制假设: TC 用同站点留一 TC 均值居中（同站点 TC 之间诱导负相关，重叠被压低）；ASD 用同站点全体 TC 均值居中
（站点均值的估计误差为同站点 ASD 共享，重叠被抬高）。两者都只作用于同站点被试对。
检验: 将两两 Jaccard 拆为同站点对与跨站点对；跨站点对不受上述机制影响。
ASD − TC 差异的 P 值: 站点内置换诊断标签 10,000 次（与 B15 S5 相同的置换方案），分别对全部对与跨站点对计算。
输入: derivatives/norms_B1/B13_zrc_strength_full.csv.gz（site_ok 站点）；输出: results/B22_s5_site_pairs.json
用法: .venv_B/Scripts/python.exe -I scripts/B22_s5_site_pairs.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
NPERM = 10000


def pair_jaccard(z, thr=2.0):
    x = np.hstack([z > thr, z < -thr]).astype(np.float64)
    inter = x @ x.T
    s = x.sum(1)
    union = s[:, None] + s[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.where(union > 0, inter / union, np.nan)
    return j


def group_means(J, site, lab, code):
    m = lab == code
    sub = J[np.ix_(m, m)]
    st = site[m]
    iu = np.triu_indices(m.sum(), 1)
    v, same = sub[iu], (st[:, None] == st[None, :])[iu]
    ok = ~np.isnan(v)
    return float(v[ok].mean()), float(v[ok & same].mean()), float(v[ok & ~same].mean())


def main():
    z = pd.read_csv(ROOT / "derivatives/norms_B1/B13_zrc_strength_full.csv.gz")
    z = z[z.site_ok]
    w = z.pivot_table(index=["cohort", "sub_n", "site", "dx"], columns="feature", values="zrc").dropna()
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B22_s5_site_pairs.py", "nperm": NPERM, "results": {}}
    for c in ["abide1", "abide2"]:
        wc = w[w.index.get_level_values("cohort") == c]
        site = wc.index.get_level_values("site").to_numpy()
        dx = wc.index.get_level_values("dx").to_numpy()
        J = pair_jaccard(wc.values)
        asd, tc = group_means(J, site, dx, 1), group_means(J, site, dx, 2)
        obs = np.array([asd[k] - tc[k] for k in range(3)])
        rng = np.random.default_rng(42)
        null = np.empty((NPERM, 3))
        for b in range(NPERM):
            lab = dx.copy()
            for s in np.unique(site):
                ix = np.where(site == s)[0]
                lab[ix] = rng.permutation(lab[ix])
            a, t = group_means(J, site, lab, 1), group_means(J, site, lab, 2)
            null[b] = [a[k] - t[k] for k in range(3)]
        p = [(1 + (np.abs(null[:, k]) >= abs(obs[k])).sum()) / (NPERM + 1) for k in range(3)]
        out["results"][c] = {lab: {"ASD": round(asd[k], 5), "TC": round(tc[k], 5), "ASD_minus_TC": round(float(obs[k]), 5), "perm_p_two_sided": round(float(p[k]), 5)}
                             for k, lab in enumerate(["all_pairs", "same_site_pairs", "between_site_pairs"])}
        print(c, json.dumps(out["results"][c]), flush=True)
    (ROOT / "results/B22_s5_site_pairs.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
