"""课题 B1 事后分析（2026-10-31，登记于方案第 12 节）：信度匹配基线工具 relbase 的验证与应用。

1. 工具验证（外部事实源）
   a. relbase.icc21 与 B14 已独立验证的逐分区 ICC（derivatives/norms_B1/B14_icc_per_feature.csv）逐项比对；
   b. relbase.BaselineModel 用与 B20 相同的输入与种子，复现 results/B20_noise_sim.json 的网格数值（应完全相同）。
2. 应用
   a. 逐分区信度（TC 分半 ICC → Spearman–Brown 全长）代替统一信度，重估基线曲线与隐含共享比例（回应"各分区信度相同"的假设）；
   b. 第二个重叠指标：同一分区同向极端者的最大比例（max_prevalence，对应 Segal 2023 / Ilioska 2026 的报告方式）；
   c. 实测重叠是否高于"真实偏离完全个体化"（s = 0）的噪声预期：单侧 Monte Carlo 检验，每组 1,000 次；
      各组用自身 n 与 z 标准差；Jaccard 只用跨站点被试对（同站点对受站点居中误差影响，见 B22）。
3. 公开示例输入：relbase/examples/abide_glasser379_tc_inputs.npz（仅 TC 汇总相关矩阵与逐分区信度，不含任何被试标识）。
种子: 42 / 7 / 123 / 2024（曲线每种子 25 次）；检验种子 42。
输出: results/B21_relbase.json
用法: .venv_B/Scripts/python.exe -I scripts/B21_relbase_apply.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "relbase"))
from relbase import BaselineModel, estimate_inputs, icc21, implied_shared_fraction, max_prevalence, mean_jaccard  # noqa: E402

NB = ROOT / "derivatives/norms_B1"
sys.stdout.reconfigure(encoding="utf-8")


def wide(cond):
    z = pd.read_csv(NB / f"B13_zrc_strength_{cond}.csv.gz")
    z = z[z.site_ok]
    return z.pivot_table(index=["cohort", "sub_n", "site", "dx"], columns="feature", values="zrc").dropna()


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B21_relbase_apply.py"}
    full, h1, h2 = wide("full"), wide("half1"), wide("half2")
    idx = full.index.intersection(h1.index).intersection(h2.index)
    full, h1, h2 = full.loc[idx], h1.loc[idx], h2.loc[idx]
    feats = list(full.columns)
    dx, coh = idx.get_level_values("dx"), idx.get_level_values("cohort")
    site = idx.get_level_values("site").to_numpy()
    tc = dx == 2

    # 1a. ICC 与 B14 比对
    ref = pd.read_csv(NB / "B14_icc_per_feature.csv")
    ref = ref[ref.pair == "P1P2_strength_half"]
    diffs = {}
    for unit, c in [("ABIDE_I", "abide1"), ("ABIDE_II", "abide2")]:
        for g, code in [("TC", 2), ("ASD", 1)]:
            m = (coh == c) & (dx == code)
            mine = pd.Series(icc21(h1.values[m], h2.values[m]), index=feats)
            r = ref[(ref.unit == unit) & (ref.group == g)].set_index("feature").icc.reindex(feats)
            diffs[f"{unit}_{g}"] = {"n": int(m.sum()), "max_abs_diff": float((mine - r).abs().max()), "median_mine": round(float(mine.median()), 4)}
    out["validation_icc_vs_B14"] = diffs
    print("ICC vs B14", diffs, flush=True)

    # 1b. 复现 B20
    b20 = json.loads((ROOT / "results/B20_noise_sim.json").read_text(encoding="utf-8"))
    inp = b20["inputs"]
    R_obs = np.corrcoef(full.values[tc].T)
    R_noise = np.corrcoef(((h1.values[tc] - h2.values[tc]) / np.sqrt(2)).T)
    k_asd = float(full.values[~tc].std())
    n = int(inp["n_sim"])
    model = BaselineModel(R_obs, R_noise, scale=k_asd)
    r_full = round(float(inp["r_full_pred"]), 3)
    repro = {}
    for r in [r_full, 0.9]:
        curve = model.baseline_curve(n, r, reps=b20["reps_per_seed"])
        for c in curve:
            ref_v = b20["grid"][f"s={c['s']}|r={r}"]["mean"]
            repro[f"s={c['s']}|r={r}"] = {"relbase": round(c["mean"], 5), "B20": ref_v, "abs_diff": abs(round(c["mean"], 5) - ref_v)}
    out["validation_reproduce_B20"] = {"max_abs_diff": max(v["abs_diff"] for v in repro.values()), "cells": repro}
    print("B20 reproduction max diff", out["validation_reproduce_B20"]["max_abs_diff"], flush=True)

    # 3. 公开示例输入（仅 TC 汇总量）
    est = estimate_inputs(full.values[tc], h1.values[tc], h2.values[tc])
    ex = ROOT / "relbase/examples"
    ex.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ex / "abide_glasser379_tc_inputs.npz", features=np.array(feats), **est)
    rel_feat = np.clip(est["reliability_full"], 0, 1)
    out["per_feature_reliability_full"] = {"median": round(float(np.median(rel_feat)), 4),
                                           "q10_q90": [round(float(x), 4) for x in np.quantile(rel_feat, [0.1, 0.9])],
                                           "min": round(float(rel_feat.min()), 4), "max": round(float(rel_feat.max()), 4)}

    groups = {f"{g}_{c}": (dx == code) & (coh == c) for g, code in [("ASD", 1), ("TC", 2)] for c in ["abide1", "abide2"]}
    observed = {m: {k: f(full.values[v]) for k, v in groups.items()} for m, f in [("jaccard", mean_jaccard), ("max_prevalence", max_prevalence)]}
    # 站点居中的估计误差在同站点内共享（B22）：Jaccard 另报只含跨站点被试对的版本，检验与隐含共享比例以此为准
    observed["jaccard_between_site"] = {k: mean_jaccard(full.values[v], blocks=site[v]) for k, v in groups.items()}
    out["observed"] = {m: {k: round(v, 4) for k, v in d.items()} for m, d in observed.items()}

    # 2a/2b. 曲线：统一信度 vs 逐分区信度；两个指标
    curves = {}
    for metric in ["jaccard", "max_prevalence"]:
        for label, rel in [("uniform", r_full), ("per_feature", rel_feat)]:
            curve = model.baseline_curve(n, rel, metric=metric, reps=25)
            curves[f"{metric}|{label}"] = {
                "curve": [{k: round(v, 5) for k, v in c.items()} for c in curve],
                "implied_shared_fraction": {k: implied_shared_fraction(v, curve) for k, v in observed[metric].items()}}
            print(metric, label, [round(c["mean"], 4) for c in curve], flush=True)
    out["curves"] = curves

    # 2c. 高于"完全个体化"噪声预期的检验，及各组自身的隐含共享比例（逐分区信度；各组自身 n 与 z 标准差）
    tests, own = {}, {}
    for k, m in groups.items():
        sd_g = float(full.values[m].std())
        gm = BaselineModel(R_obs, R_noise, scale=sd_g)
        for metric, obs_key in [("jaccard", "jaccard_between_site"), ("max_prevalence", "max_prevalence")]:
            res = gm.test_heterogeneous_null(observed[obs_key][k], int(m.sum()), rel_feat, metric=metric, reps=1000, seed=42)
            tests[f"{metric}|{k}"] = {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in res.items()} | {"n": int(m.sum()), "scale": round(sd_g, 4)}
            curve = gm.baseline_curve(int(m.sum()), rel_feat, metric=metric, reps=25)
            own[f"{metric}|{k}"] = {"curve_means": [round(c["mean"], 5) for c in curve],
                                    "observed_used": obs_key,
                                    "implied_shared_fraction": implied_shared_fraction(observed[obs_key][k], curve)}
    out["test_vs_fully_individual"] = tests
    out["group_specific_curves"] = own
    print(json.dumps(own, indent=1), flush=True)
    print(json.dumps(tests, indent=1), flush=True)

    (ROOT / "results/B21_relbase.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
