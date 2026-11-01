"""课题 B1 事后分析（2026-11-02，双专家视角内评补做 ①③）：一致性 ICC(3,1) 敏感性与偏离图的多变量信度。

a. ICC(3,1)（两向混合、一致性、单次测量）：与 ICC(2,1) 的差别是不惩罚两次测量间的系统性均值偏移。
   跨 run 用前半常模打分，run 偏离有共同均值偏移（约 +0.55），绝对一致 ICC 会因此受罚；分半两段各自建模，预期两者接近。
b. 被试识别率（Finn 2015 式指纹识别）：前半偏离图与所有人后半偏离图求 Pearson 相关，本人相关最高即识别成功；
   两个方向取平均；机会水平 = 1/n。另报本人相关与他人相关的均值差（差异可识别性，Amico & Goñi 2018）。
c. Discriminability（Bridgeford 2021）：每个被试每次测量，与本人另一次测量的欧氏距离小于与他人各次测量距离的比例，取平均；
   机会水平 0.5。
特征集: 分区 FC 强度偏离（379）、网络 FC 偏离（91）、连接偏离（71,631，B28 线性常模）；分队列 × 组别，及队列合并组别。
输出: results/B29_icc31_fingerprint.json
用法: .venv_B/Scripts/python.exe -I scripts/B29_icc31_fingerprint.py
"""
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("B14", ROOT / "scripts/B14_reliability.py")
B14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B14)


def icc31(a, b):
    n = a.shape[0]
    x = np.stack([a, b], axis=2)
    m = x.mean(axis=(0, 2))
    rm = x.mean(axis=2)
    cm = x.mean(axis=0)
    ssr = 2 * ((rm - m) ** 2).sum(0)
    ssc = n * ((cm - m[:, None]) ** 2).sum(1)
    sst = ((x - m[None, :, None]) ** 2).sum(axis=(0, 2))
    mse = (sst - ssr - ssc) / (n - 1)
    msr = ssr / (n - 1)
    return (msr - mse) / (msr + mse)


def rowcorr(a, b):
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return a @ b.T / a.shape[1]


def sqdist(a, b):
    return (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T


def multivariate(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = len(a)
    R = rowcorr(a, b)
    ident = (np.mean(R.argmax(1) == np.arange(n)) + np.mean(R.argmax(0) == np.arange(n))) / 2
    off = ~np.eye(n, dtype=bool)
    self_r, other_r = float(np.median(np.diag(R))), float(np.median(R[off]))
    Dab = sqdist(a, b)
    Daa, Dbb = sqdist(a, a), sqdist(b, b)
    w = np.diag(Dab)
    disc = []
    for i in range(n):
        for row, same in ((np.r_[Dab[i], Daa[i]], Daa), (np.r_[Dab[:, i], Dbb[i]], Dbb)):
            others = np.delete(row, [i, n + i])
            disc.append(np.mean(others > w[i]))
    return {"n": n, "identification": round(float(ident), 4), "chance": round(1 / n, 4),
            "median_r_self": round(self_r, 4), "median_r_other": round(other_r, 4),
            "differential_identifiability": round(float(np.mean(np.diag(R)) - np.mean(R[off])), 4),
            "discriminability": round(float(np.mean(disc)), 4)}


def units(meta):
    for c, cn in [("abide1", "ABIDE I"), ("abide2", "ABIDE II")]:
        for g, gn in [(2, "TC"), (1, "ASD")]:
            yield f"{cn} {gn}", ((meta.cohort == c) & (meta.dx == g)).to_numpy()
        yield f"{cn} all", (meta.cohort == c).to_numpy()


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B29_icc31_fingerprint.py"}
    res = {}
    for name, (st, c1, c2) in {"strength_half": ("strength", "half1", "half2"), "strength_run": ("strength", "run1", "run2"),
                               "netfc_half": ("netfc", "half1", "half2")}.items():
        w1, w2 = B14.load_pair(st, c1, c2)
        meta = w1.index.to_frame(index=False)
        a, b = w1.to_numpy(float), w2.to_numpy(float)
        r = {}
        sel_list = list(units(meta)) if c1 == "half1" else [(f"pooled {g}", (meta.dx == k).to_numpy()) for g, k in [("TC", 2), ("ASD", 1)]] + [("pooled all", np.ones(len(meta), bool))]
        for lab, sel in sel_list:
            v = {"median_icc21": round(float(np.median(B14.icc21(a[sel], b[sel]))), 4),
                 "median_icc31": round(float(np.median(icc31(a[sel], b[sel]))), 4),
                 "mean_shift_seg2_minus_seg1": round(float((b[sel] - a[sel]).mean()), 4)}
            v.update(multivariate(a[sel], b[sel]))
            r[lab] = v
            print(name, lab, v, flush=True)
        res[name] = r
    sl = pd.read_csv(ROOT / "derivatives/norms_B1/B15_subject_level.csv")
    e1 = np.load(ROOT / "derivatives/norms_B1/edges/edge_zrc_half1.npy", mmap_mode="r")
    e2 = np.load(ROOT / "derivatives/norms_B1/edges/edge_zrc_half2.npy", mmap_mode="r")
    r = {}
    for lab, sel in units(sl):
        ix = np.where(sel)[0]
        r[lab] = multivariate(np.asarray(e1[ix]), np.asarray(e2[ix]))
        print("edges_half", lab, r[lab], flush=True)
    res["edges_half"] = r
    out["results"] = res
    (ROOT / "results/B29_icc31_fingerprint.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
