"""课题 B1 事后分析（2026-11-01）：连接水平跨站点重叠的"每人固定极端数"检验。

动机: B28 中 ABIDE II 的 ASD 跨站点 Jaccard 高于 TC，但 ASD 的极端率也更高（7.7% vs 6.0%）；
      每人极端数越多，Jaccard 机械性越大。此处每人取 |z| 最大的 k 条连接（k = 全部 TC 的 |z| > 2 比例 × 71,631），
      记录其符号，使每人极端数相同，再比较跨站点 Jaccard（站点内置换诊断 10,000 次）。分区水平同样计算以作对照。
输入: derivatives/norms_B1/edges/edge_zrc_full.npy；derivatives/norms_B1/B13_zrc_strength_full.csv.gz；B15_subject_level.csv
输出: results/B28b_edge_fixed_count.json
用法: .venv_B/Scripts/python.exe -I scripts/B28b_edge_fixed_count.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def topk_indicator(z, k):
    """每行取 |z| 最大的 k 个特征，按符号编码为 2P 维指示矩阵。"""
    n, p = z.shape
    idx = np.argpartition(-np.abs(z), k - 1, axis=1)[:, :k]
    x = np.zeros((n, 2 * p), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    sign_neg = (z[rows, cols] < 0).astype(int)
    x[rows, cols + sign_neg * p] = 1
    return x


def site_perm(site, rng):
    perm = np.arange(len(site))
    for s in np.unique(site):
        ix = np.where(site == s)[0]
        perm[ix] = rng.permutation(ix)
    return perm


def compare(z, lab, site, k, nperm=10000):
    x = topk_indicator(z, k)
    inter = x @ x.T
    J = inter / (2 * k - inter)
    iu = np.triu_indices(len(z), 1)
    v = J[iu]
    diff_site = (site[:, None] != site[None, :])[iu]

    def gm(lb):
        same = (lb[:, None] == lb[None, :])[iu] & diff_site
        g = lb[iu[0]]
        return float(v[same & (g == 1)].mean()), float(v[same & (g == 2)].mean())
    a, t = gm(lab)
    rng = np.random.default_rng(42)
    null = np.array([np.subtract(*gm(lab[site_perm(site, rng)])) for _ in range(nperm)])
    p = (1 + np.sum(np.abs(null) >= abs(a - t))) / (nperm + 1)
    return {"k": int(k), "between_site_ASD": round(a, 5), "between_site_TC": round(t, 5), "diff": round(a - t, 5), "perm_p": round(float(p), 4)}


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B28b_edge_fixed_count.py"}
    sl = pd.read_csv(ROOT / "derivatives/norms_B1/B15_subject_level.csv")
    ze = np.load(ROOT / "derivatives/norms_B1/edges/edge_zrc_full.npy")
    tc = (sl.dx == 2).to_numpy()
    rate = float((np.abs(ze[tc]) > 2).mean())
    k_edge = int(round(rate * ze.shape[1]))
    zp = pd.read_csv(ROOT / "derivatives/norms_B1/B13_zrc_strength_full.csv.gz")
    zp = zp[zp.site_ok].pivot_table(index=["cohort", "sub_n"], columns="feature", values="zrc")
    zp = zp.loc[list(zip(sl.cohort, sl.sub_n))].to_numpy()
    k_parcel = int(round(float((np.abs(zp[tc]) > 2).mean()) * zp.shape[1]))
    out["tc_extreme_rate_edges"] = round(rate, 4)
    for c in ["abide1", "abide2"]:
        m = (sl.cohort == c).to_numpy()
        lab = np.where(tc[m], 2, 1)
        site = sl.site.to_numpy()[m]
        out[c] = {"edges": compare(ze[m], lab, site, k_edge), "parcels": compare(zp[m], lab, site, k_parcel)}
        print(c, out[c], flush=True)
    (ROOT / "results/B28b_edge_fixed_count.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
