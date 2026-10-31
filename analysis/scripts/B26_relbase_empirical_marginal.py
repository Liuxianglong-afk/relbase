"""课题 B1 事后分析（2026-10-31，登记于方案第 12 节）：relbase 经验边缘分布（高斯 copula）基线。

动机: B25 显示实测偏离尾部重于高斯（|z| > 3 约为高斯预期的 2 倍），高斯基线在严阈值下低估极端值的偶合。
做法: 模拟仍用 relbase 的高斯潜变量（空间相关 R_signal/R_noise、逐分区信度），再按经验分布做分位数映射。
1. 校准: (i) 潜变量信度 r 与映射后实测尺度 ICC 的关系——模拟两半（s = 0），比较映射后中位 ICC 与输入 r；
   若偏差 > 0.01，按比例校准逐分区 r 使映射后中位 ICC 等于实测；(ii) 用半长边缘分布复现实测 P2（同 B20 验证）。
2. 阈值 1.5 / 2.0 / 2.5 / 3.0：各组用自身全长 z 的经验边缘、自身 n，比较跨站点 Jaccard 与 s = 0 零分布（500 次），
   并报告模拟与实测的极端率。
3. 主阈值 |z| > 2：ASD 各队列的基线曲线（s = 0…1）、隐含共享比例、检验效能 80% 的最小可检出 s。
种子: 曲线 42/7/123/2024 × 25 次；检验 42；效能 1000 + 100·s。
输出: results/B26_relbase_empirical_marginal.json
用法: .venv_B/Scripts/python.exe -I scripts/B26_relbase_empirical_marginal.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "relbase"))
from relbase import (BaselineModel, estimate_inputs, icc21, implied_shared_fraction, marginal_quantiles,  # noqa: E402
                     mean_jaccard, spearman_brown)

NB = ROOT / "derivatives/norms_B1"
sys.stdout.reconfigure(encoding="utf-8")
IDX = ["cohort", "sub_n", "site", "dx"]


def wide(cond):
    z = pd.read_csv(NB / f"B13_zrc_strength_{cond}.csv.gz")
    z = z[z.site_ok]
    return z.pivot_table(index=IDX, columns="feature", values="zrc").dropna()


def pos_agreement(a, b, thr=2.0):
    ca, cb = np.sign(a) * (np.abs(a) > thr), np.sign(b) * (np.abs(b) > thr)
    both = ((ca != 0) & (ca == cb)).sum()
    return float((both / max((ca != 0).sum(), 1) + both / max((cb != 0).sum(), 1)) / 2)


def two_halves(model, rng, n, r):
    """同一真实偏离（s = 0）的两次独立测量，映射到模型的边缘分布。"""
    rr = model._reliability(r)
    T = model._draw(rng, model.L_signal, n)
    out = []
    for _ in range(2):
        lat = np.sqrt(rr) * T + np.sqrt(1 - rr) * model._draw(rng, model.L_noise, n)
        out.append(model._to_marginal(lat) if model.marginal is not None else model.scale * lat)
    return out


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B26_relbase_empirical_marginal.py"}
    full, h1, h2 = wide("full"), wide("half1"), wide("half2")
    idx = full.index.intersection(h1.index).intersection(h2.index)
    full, h1, h2 = full.loc[idx], h1.loc[idx], h2.loc[idx]
    dx, coh, site = (idx.get_level_values(k).to_numpy() for k in ["dx", "cohort", "site"])
    groups = {f"{g}_{c}": (dx == code) & (coh == c) for g, code in [("ASD", 1), ("TC", 2)] for c in ["abide1", "abide2"]}
    tc = dx == 2
    est = estimate_inputs(full.values[tc], h1.values[tc], h2.values[tc])
    R_obs, R_noise = est["R_signal"], est["R_noise"]
    rel_half = np.clip(est["reliability_half"], 0, 1)
    b14 = json.loads((ROOT / "results/B14_reliability.json").read_text(encoding="utf-8"))["results"]["P1P2_strength_half"]
    p2_obs = float(np.mean([b14[u][g]["extreme"]["pos_agreement_mean_dir"] for u in ["ABIDE_I", "ABIDE_II"] for g in ["TC", "ASD"]]))
    obs_half_icc = float(np.median(icc21(h1.values[tc], h2.values[tc])))

    # 1. 校准（半长、TC 汇总；n = 全部 TC）
    m_half = BaselineModel(R_obs, R_noise, marginal=marginal_quantiles(np.vstack([h1.values, h2.values])))
    n_tc = int(tc.sum())

    def mapped_icc(scale_r, reps=20):
        rng = np.random.default_rng(42)
        v = []
        for _ in range(reps):
            a, b = two_halves(m_half, rng, n_tc, np.clip(rel_half * scale_r, 0, 1))
            v.append(np.median(icc21(a, b)))
        return float(np.mean(v))
    icc_at_1 = mapped_icc(1.0)
    factor = 1.0
    if abs(icc_at_1 - obs_half_icc) > 0.01:
        lo, hi = 0.5, 2.0
        for _ in range(20):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mapped_icc(mid) < obs_half_icc else (lo, mid)
        factor = (lo + hi) / 2
    rel_half_cal = np.clip(rel_half * factor, 0, 1)
    rel_full_cal = spearman_brown(rel_half_cal, 2)
    rng = np.random.default_rng(42)
    p2_sim = []
    for _ in range(25):
        a, b = two_halves(m_half, rng, 201, rel_half_cal)
        p2_sim.append(pos_agreement(a, b))
    out["calibration"] = {"observed_tc_half_median_icc": round(obs_half_icc, 4), "mapped_icc_uncalibrated": round(icc_at_1, 4),
                          "reliability_scale_factor": round(factor, 4), "mapped_icc_calibrated": round(mapped_icc(factor), 4),
                          "p2_simulated": round(float(np.mean(p2_sim)), 4), "p2_observed": round(p2_obs, 4),
                          "median_full_reliability_calibrated": round(float(np.median(rel_full_cal)), 4)}
    print("calibration", out["calibration"], flush=True)

    # 2. 阈值敏感性
    thr_res = {}
    for thr in [1.5, 2.0, 2.5, 3.0]:
        row = {}
        for k, m in groups.items():
            gm = BaselineModel(R_obs, R_noise, threshold=thr, marginal=marginal_quantiles(full.values[m]))
            obs = mean_jaccard(full.values[m], thr, blocks=site[m])
            t = gm.test_heterogeneous_null(obs, int(m.sum()), rel_full_cal, reps=500, seed=42)
            zsim = gm.simulate(np.random.default_rng(1), int(m.sum()), rel_full_cal, 0.0)
            row[k] = {"jaccard_between_site": round(obs, 5), "null_mean": round(t["null_mean"], 5), "p_one_sided_vs_s0": round(t["p_one_sided"], 4),
                      "extreme_rate_observed": round(float((np.abs(full.values[m]) > thr).mean()), 4),
                      "extreme_rate_simulated": round(float((np.abs(zsim) > thr).mean()), 4)}
        thr_res[f"thr={thr}"] = row
        print("thr", thr, {k: (v["jaccard_between_site"], v["null_mean"], v["p_one_sided_vs_s0"]) for k, v in row.items()}, flush=True)
    out["threshold_sensitivity"] = thr_res

    # 3. 主阈值：曲线、隐含共享比例、效能
    prim = {}
    for k in ["ASD_abide1", "ASD_abide2"]:
        m = groups[k]
        n = int(m.sum())
        gm = BaselineModel(R_obs, R_noise, marginal=marginal_quantiles(full.values[m]))
        curve = gm.baseline_curve(n, rel_full_cal, reps=25)
        obs = mean_jaccard(full.values[m], 2.0, blocks=site[m])
        q95 = float(np.quantile(gm.overlap_distribution(n, rel_full_cal, 0.0, reps=1000, seed=42), 0.95))
        power = {}
        for s in np.round(np.arange(0.02, 0.21, 0.02), 2):
            alt = gm.overlap_distribution(n, rel_full_cal, float(s), reps=200, seed=1000 + int(s * 100))
            power[float(s)] = round(float((alt > q95).mean()), 3)
        ss, pw = np.array(list(power)), np.array(list(power.values()))
        s80 = float(np.interp(0.8, pw, ss)) if pw.max() >= 0.8 else None
        prim[k] = {"n": n, "curve": [{kk: round(v, 5) for kk, v in c.items()} for c in curve],
                   "observed_between_site": round(obs, 5), "implied_shared_fraction": implied_shared_fraction(obs, curve),
                   "power_by_s": power, "min_detectable_s_power80": None if s80 is None else round(s80, 3)}
        print("primary", k, [round(c["mean"], 4) for c in curve], prim[k]["implied_shared_fraction"], s80, flush=True)
    out["primary_threshold"] = prim
    (ROOT / "results/B26_relbase_empirical_marginal.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
