"""课题 B1：B12 交叉拟合 z 值的站点重新居中与常模校准检验（protocol_B 第 7 节，protocol_B1 v0.2 第 6 节）。

重新居中（每个特征集 × 数据长度 × 特征 × 站点）:
  TC 本人: 用同站点其余 TC 的 z 均值与标准差（留一）；ASD: 用同站点全部 TC。
  本地 TC ≥ 10: z' = (z - 均值) / 标准差；本地 TC < 10: 该站点不校正，标记 site_ok=False（只进入敏感性分析）。
  run1/run2: 用 half1 的站点参数（同站点全部 TC 的 half1 z；TC 本人留一），因为跨 run 子样本每站点 TC 太少。
校准检验（只用 TC、site_ok 站点）: 每个特征 z' 的均值、标准差、|z'| ≤ 1.645 覆盖率（理想 0.90）、|z'| > 2 比例（理想约 0.046）；
  汇总为各特征集 × 数据长度的中位数，并按站点汇总。本步骤不输出任何 ASD 统计量。
输出: derivatives/norms_B1/B13_zrc_<set>_<cond>.csv.gz（长表：cohort, sub_n, site, dx, feature, z, zrc, site_ok）
      results/B13_calibration.json
用法: .venv_B/Scripts/python.exe -I scripts/B13_site_recenter_calibration.py [--variant fd30|oddeven|gordon|combat]
  变体: 读写 derivatives/norms_B1/sens_<变体>/，只有 strength 的 full/half1/half2；fd30 用 pass_fd30 定样本与站点 TC 数；
  combat 已在特征层面做站点协调，不再重新居中（zrc = z），site_ok 仍按主分析规则标记以保持样本一致。
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derivatives/norms_B1"
MIN_TC = 10
SETS = {"strength": ["full", "half1", "half2", "run1", "run2"], "netfc": ["full", "half1", "half2", "run1", "run2"],
        "thickness": ["full"]}


def site_params(z):
    """z: 含 site, dx, feature, key, z 的长表。返回每行的居中参数（TC 留一）。"""
    tc = z[z.dx == 2]
    g = tc.groupby(["site", "feature"]).z
    agg = pd.DataFrame({"n": g.size(), "s1": g.sum(), "s2": g.apply(lambda v: (v ** 2).sum())}).reset_index()
    out = z.merge(agg, on=["site", "feature"], how="left")
    is_tc = out.dx == 2
    n = out.n - is_tc
    s1 = out.s1 - np.where(is_tc, out.z, 0)
    s2 = out.s2 - np.where(is_tc, out.z ** 2, 0)
    mu = s1 / n
    var = (s2 - n * mu ** 2) / (n - 1)
    out["mu"], out["sd"], out["k"] = mu, np.sqrt(var), n
    return out[["key", "feature", "mu", "sd", "k"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    v = ap.parse_args().variant
    global NB, SETS
    if v != "main":
        NB = ROOT / "derivatives/norms_B1" / f"sens_{v}"
        SETS = {"strength": ["full", "half1", "half2"]}
    pcol = "pass_fd30" if v == "fd30" else "pass_fd20"
    qc = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_rest_qc_*.csv"))[-1])
    cov = qc.loc[qc[pcol], ["cohort", "sub_n", "site", "dx_group"]].rename(columns={"dx_group": "dx"})
    site_ntc = cov[cov.dx == 2].groupby("site").size()
    ok_sites = set(site_ntc[site_ntc >= MIN_TC].index)
    summary = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B13_site_recenter_calibration.py",
               "min_local_tc": MIN_TC, "sites_ok": sorted(ok_sites),
               "sites_excluded_lt10_tc": {s: int(n) for s, n in site_ntc.items() if s not in ok_sites},
               "n_subjects_in_excluded_sites": int(cov[~cov.site.isin(ok_sites)].shape[0]), "calibration": {}}
    for st, conds in SETS.items():
        half1_params = None
        for cond in conds:
            f = NB / f"B12_z_{st}_{cond}.csv.gz"
            if not f.exists():
                continue
            z = pd.read_csv(f).merge(cov, on=["cohort", "sub_n"])
            z["key"] = z.cohort + "_" + z.sub_n.astype(str)
            if cond in ("run1", "run2"):
                p = half1_params
            else:
                p = site_params(z)
                if cond == "half1":
                    half1_params = p
            z = z.merge(p, on=["key", "feature"], how="left")
            z["site_ok"] = z.site.isin(ok_sites)
            z["zrc"] = np.where(z.site_ok, z.z if v == "combat" else (z.z - z.mu) / z.sd, np.nan)
            z[["cohort", "sub_n", "site", "dx", "feature", "z", "zrc", "site_ok"]].to_csv(NB / f"B13_zrc_{st}_{cond}.csv.gz", index=False)

            tc = z[(z.dx == 2) & z.site_ok]
            per_feat = tc.groupby("feature").zrc.agg(mean="mean", sd="std",
                                                     cover90=lambda v: (v.abs() <= 1.645).mean(),
                                                     ext2=lambda v: (v.abs() > 2).mean())
            per_site = tc.groupby("site").zrc.agg(mean="mean", sd="std")
            summary["calibration"][f"{st}_{cond}"] = {
                "n_tc": int(tc.key.nunique()), "n_features": int(per_feat.shape[0]),
                "median_over_features": {k: round(float(v), 4) for k, v in per_feat.median().items()},
                "range_mean": [round(float(per_feat["mean"].min()), 3), round(float(per_feat["mean"].max()), 3)],
                "range_sd": [round(float(per_feat["sd"].min()), 3), round(float(per_feat["sd"].max()), 3)],
                "raw_z_tc_median_sd": round(float(z[z.dx == 2].groupby("feature").z.std().median()), 4),
                "per_site_sd_range": [round(float(per_site["sd"].min()), 3), round(float(per_site["sd"].max()), 3)],
            }
            print(st, cond, json.dumps(summary["calibration"][f"{st}_{cond}"]["median_over_features"]), flush=True)
    summary["variant"] = v
    (ROOT / ("results/B13_calibration.json" if v == "main" else f"results/B13_calibration_{v}.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
