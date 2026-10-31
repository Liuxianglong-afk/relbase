"""课题 B1 主要终点 P1、P2 及 S1、S3（protocol_B1 v1.0 第 4、5 节）。

输入: derivatives/norms_B1/B13_zrc_<set>_<cond>.csv.gz（站点重新居中后的 z，只用 site_ok 站点）
配对: P1/P2 = strength 的 half1 vs half2；S1 = strength 的 run1 vs run2；S3 = netfc 的 half1 vs half2（另报 netfc run1 vs run2）
P1: 每个特征的 ICC(2,1)（两向随机、绝对一致、单次测量），分别在 TC、ASD 中计算；汇总为特征 ICC 的中位数与四分位距。
    中位 ICC 的 95% CI：按被试 bootstrap 2,000 次，seed 42/7/123/2024 各一次（报告各 seed 的 CI 与跨 seed 均值）。
    ASD − TC 的中位 ICC 差：在队列内按站点分层置换诊断标签 10,000 次（seed 42），双侧 P。
P2: 极端偏离 |z| > 2。正向一致率（第一段为极端的被试–特征单元中，第二段同方向极端的比例；两个方向取平均）、
    对称正向一致率 2·both / (n1 + n2)、三分类（负极端 / 非极端 / 正极端）Cohen κ。
分析单位: 每个队列（ABIDE I、ABIDE II）分别报告；S1 样本小，合并两个队列报告。
输出: results/B14_reliability.json；derivatives/norms_B1/B14_icc_per_feature.csv（特征 × 配对 × 队列 × 组）
用法: .venv_B/Scripts/python.exe -I scripts/B14_reliability.py [--nperm 10000] [--nboot 2000] [--variant fd30|oddeven|gordon|combat]
  变体: 读 derivatives/norms_B1/sens_<变体>/，只做 P1P2_strength_half；输出 results/B14_reliability_<变体>.json
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derivatives/norms_B1"
SEEDS = [42, 7, 123, 2024]
PAIRS = {"P1P2_strength_half": ("strength", "half1", "half2"), "S1_strength_run": ("strength", "run1", "run2"),
         "S3_netfc_half": ("netfc", "half1", "half2"), "S3_netfc_run": ("netfc", "run1", "run2")}


def icc21(a, b):
    """a, b: n × F。返回每列的 ICC(2,1)。"""
    n = a.shape[0]
    x = np.stack([a, b], axis=2)
    m = x.mean(axis=(0, 2))
    rm = x.mean(axis=2)
    cm = x.mean(axis=0)
    ssr = 2 * ((rm - m) ** 2).sum(0)
    ssc = n * ((cm - m[:, None]) ** 2).sum(1)
    sst = ((x - m[None, :, None]) ** 2).sum(axis=(0, 2))
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc, sse / (n - 1)
    return (msr - mse) / (msr + mse + 2 * (msc - mse) / n)


def extreme_agreement(a, b, thr=2.0):
    ca = np.sign(a) * (np.abs(a) > thr)
    cb = np.sign(b) * (np.abs(b) > thr)
    ea, eb = ca != 0, cb != 0
    both = (ea & eb & (ca == cb)).sum()
    pa_ab = both / max(ea.sum(), 1)
    pa_ba = both / max(eb.sum(), 1)
    cats = [-1, 0, 1]
    po = (ca == cb).mean()
    pe = sum((ca == c).mean() * (cb == c).mean() for c in cats)
    return {"n_extreme_seg1": int(ea.sum()), "n_extreme_seg2": int(eb.sum()), "n_both_same_sign": int(both),
            "rate_extreme_seg1": round(float(ea.mean()), 5), "rate_extreme_seg2": round(float(eb.mean()), 5),
            "pos_agreement_mean_dir": round(float((pa_ab + pa_ba) / 2), 4),
            "pos_agreement_symmetric": round(float(2 * both / max(ea.sum() + eb.sum(), 1)), 4),
            "kappa_3cat": round(float((po - pe) / (1 - pe)), 4)}


def load_pair(st, c1, c2, nb=None):
    nb = NB if nb is None else nb
    z1 = pd.read_csv(nb / f"B13_zrc_{st}_{c1}.csv.gz")
    z2 = pd.read_csv(nb / f"B13_zrc_{st}_{c2}.csv.gz")
    z1, z2 = z1[z1.site_ok], z2[z2.site_ok]
    w1 = z1.pivot_table(index=["cohort", "sub_n", "site", "dx"], columns="feature", values="zrc")
    w2 = z2.pivot_table(index=["cohort", "sub_n", "site", "dx"], columns="feature", values="zrc")
    idx = w1.index.intersection(w2.index)
    cols = w1.columns.intersection(w2.columns)
    w1, w2 = w1.loc[idx, cols], w2.loc[idx, cols]
    ok = w1.notna().all(1) & w2.notna().all(1)
    return w1[ok], w2[ok]


def summarize(a, b, groups, sites, nboot, nperm, do_perm):
    res = {}
    icc_by_g = {}
    for g, lab in [(2, "TC"), (1, "ASD")]:
        sel = groups == g
        A, B = a[sel], b[sel]
        icc = icc21(A, B)
        icc_by_g[lab] = icc
        boot = {}
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            meds = np.empty(nboot)
            for i in range(nboot):
                ix = rng.integers(0, A.shape[0], A.shape[0])
                meds[i] = np.median(icc21(A[ix], B[ix]))
            boot[str(sd)] = [round(float(x), 4) for x in np.percentile(meds, [2.5, 97.5])]
        q = np.percentile(icc, [25, 50, 75])
        res[lab] = {"n": int(sel.sum()), "median_icc": round(float(q[1]), 4), "iqr": [round(float(q[0]), 4), round(float(q[2]), 4)],
                    "ci95_by_seed": boot, "ci95_mean_over_seeds": [round(float(np.mean([v[0] for v in boot.values()])), 4),
                                                                    round(float(np.mean([v[1] for v in boot.values()])), 4)],
                    "share_features_icc_ge_0.5": round(float((icc >= 0.5).mean()), 4),
                    "share_features_icc_ge_0.75": round(float((icc >= 0.75).mean()), 4),
                    "extreme": extreme_agreement(A, B)}
    obs = float(np.median(icc_by_g["ASD"]) - np.median(icc_by_g["TC"]))
    res["ASD_minus_TC_median_icc"] = round(obs, 4)
    if do_perm:
        rng = np.random.default_rng(42)
        null = np.empty(nperm)
        site_ix = [np.where(sites == s)[0] for s in np.unique(sites)]
        for i in range(nperm):
            gp = groups.copy()
            for ix in site_ix:
                gp[ix] = rng.permutation(gp[ix])
            null[i] = np.median(icc21(a[gp == 1], b[gp == 1])) - np.median(icc21(a[gp == 2], b[gp == 2]))
        res["perm_p_two_sided"] = round(float((np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)), 5)
        res["perm_null_sd"] = round(float(null.std()), 4)
    return res, icc_by_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--variant", default="main")
    a = ap.parse_args()
    nb = NB if a.variant == "main" else NB / f"sens_{a.variant}"
    pairs = PAIRS if a.variant == "main" else {"P1P2_strength_half": PAIRS["P1P2_strength_half"]}
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B14_reliability.py",
           "nboot": a.nboot, "nperm": a.nperm, "variant": a.variant, "results": {}}
    rows = []
    for name, (st, c1, c2) in pairs.items():
        w1, w2 = load_pair(st, c1, c2, nb)
        if w1.empty:
            continue
        meta = w1.index.to_frame(index=False)
        units = [("ABIDE_I", meta.cohort == "abide1"), ("ABIDE_II", meta.cohort == "abide2")] if c1 == "half1" else [("pooled", meta.cohort.notna())]
        out["results"][name] = {}
        for uname, sel in units:
            sel = sel.values
            res, icc = summarize(w1.values[sel], w2.values[sel], meta.dx.values[sel], meta.site.values[sel],
                                 a.nboot, a.nperm, do_perm=True)
            res["n_features"] = int(w1.shape[1])
            out["results"][name][uname] = res
            for lab, v in icc.items():
                rows.append(pd.DataFrame({"pair": name, "unit": uname, "group": lab, "feature": w1.columns, "icc": v}))
            print(name, uname, {k: (v["median_icc"], v["n"]) for k, v in res.items() if isinstance(v, dict)},
                  "diff", res["ASD_minus_TC_median_icc"], "p", res.get("perm_p_two_sided"), flush=True)
    pd.concat(rows).to_csv(nb / "B14_icc_per_feature.csv", index=False)
    (ROOT / ("results/B14_reliability.json" if a.variant == "main" else f"results/B14_reliability_{a.variant}.json")).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
