"""课题 B1：敏感性分析 5（差异性排除的逆概率加权）、6（仅男性），以及 S6 症状关联（偏离预设）。

敏感性 6 仅男性: 用主分析 z（B13），只保留男性，重算 P1 中位 ICC（bootstrap 2,000 × seed 42/7/123/2024）与 P2。
敏感性 5 IPW: 总体 = 质控表中位于 site_ok 站点的全部被试；结局 = 通过主质控（pass_fd20）。
  logistic: 年龄 + 性别 + FIQ（队列内中位数插补 + 缺失指示）+ 站点 + 诊断；权重 = 1 / p，截断于第 1 和第 99 百分位，
  在纳入者中归一化到均值 1。加权 ICC(2,1) = 用权重计算的均方（权重全为 1 时与 B14 的 icc21 相同，脚本内自检）。
S6（偏离预设：P1 无 ICC ≥ 0.5 的分区）: 只在 ASD 中，全长 strength z 的两个全脑指标（|z| > 2 的分区数、|z| 的均值）
  与 ADOS CSS、SRS 原始总分的关联：OLS 指标 ~ 症状 + 年龄 + 性别 + 平均 FD + 站点，报告标准化 β 与 P；
  探索性逐分区 Spearman（379 项），BH-FDR。有效人数 < 100 时按方案只作描述（仍输出数字，标注）。
输出: results/B17_sens_males_ipw_S6.json
用法: .venv_B/Scripts/python.exe -I scripts/B17_sens_males_ipw_S6.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from B14_reliability import SEEDS, extreme_agreement, icc21, load_pair  # noqa: E402

NB = ROOT / "derivatives/norms_B1"


def icc21_w(a, b, w):
    """加权 ICC(2,1)：w 为被试权重（自动归一化为均值 1）。"""
    w = w / w.mean()
    n = a.shape[0]
    x = np.stack([a, b], axis=2)
    W = w[:, None, None]
    m = (W * x).sum(axis=(0, 2)) / (2 * w.sum())
    rm = x.mean(axis=2)
    cm = (W * x).sum(axis=0) / w.sum()
    ssr = 2 * (w[:, None] * (rm - m) ** 2).sum(0)
    ssc = n * ((cm - m[:, None]) ** 2).sum(1)
    sst = (W * (x - m[None, :, None]) ** 2).sum(axis=(0, 2))
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc, sse / (n - 1)
    return (msr - mse) / (msr + mse + 2 * (msc - mse) / n)


def summarize(a, b, nboot, w=None):
    f = (lambda A, B, ww: icc21(A, B)) if w is None else icc21_w
    ww = np.ones(a.shape[0]) if w is None else w
    icc = f(a, b, ww)
    ci = {}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        meds = [np.median(f(a[ix], b[ix], ww[ix])) for ix in (rng.integers(0, a.shape[0], a.shape[0]) for _ in range(nboot))]
        ci[str(sd)] = [round(float(x), 4) for x in np.percentile(meds, [2.5, 97.5])]
    return {"n": int(a.shape[0]), "median_icc": round(float(np.median(icc)), 4),
            "ci95_mean_over_seeds": [round(float(np.mean([v[0] for v in ci.values()])), 4), round(float(np.mean([v[1] for v in ci.values()])), 4)],
            "extreme": extreme_agreement(a, b) if w is None else "未加权，见主分析"}


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B17_sens_males_ipw_S6.py"}
    w1, w2 = load_pair("strength", "half1", "half2")
    meta = w1.index.to_frame(index=False)
    qc = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_rest_qc_*.csv"))[-1])
    ph = pd.read_csv(ROOT / "data/interim/abide_pheno_harmonized_20261014.csv")
    meta = meta.merge(qc[["cohort", "sub_n", "sex", "age", "meanfd_retained_fd20"]], on=["cohort", "sub_n"], how="left")

    # 自检：权重全为 1 时加权 ICC = icc21
    a0, b0 = w1.values[:200], w2.values[:200]
    out["selfcheck_weighted_icc_max_abs_diff"] = float(np.abs(icc21_w(a0, b0, np.ones(200)) - icc21(a0, b0)).max())
    assert out["selfcheck_weighted_icc_max_abs_diff"] < 1e-10

    # 敏感性 6：仅男性
    out["males_only"] = {}
    for unit, coh in [("ABIDE_I", "abide1"), ("ABIDE_II", "abide2")]:
        for g, lab in [(2, "TC"), (1, "ASD")]:
            sel = ((meta.cohort == coh) & (meta.dx == g) & (meta.sex == 1)).values
            out["males_only"][f"{unit}_{lab}"] = summarize(w1.values[sel], w2.values[sel], 2000)
            print("males", unit, lab, out["males_only"][f"{unit}_{lab}"]["median_icc"], flush=True)

    # 敏感性 5：IPW
    ok_sites = set(meta.site)
    pop = qc[qc.site.isin(ok_sites)].merge(ph[["cohort", "sub_n", "fiq"]], on=["cohort", "sub_n"], how="left")
    pop["fiq_miss"] = pop.fiq.isna().astype(int)
    pop["fiq_i"] = pop.fiq.fillna(pop.groupby("cohort").fiq.transform("median"))
    pop["y"] = pop.pass_fd20.astype(int)
    pop["female"] = (pop.sex == 2).astype(int)
    pop["asd"] = (pop.dx_group == 1).astype(int)
    lm = smf.glm("y ~ age + female + fiq_i + fiq_miss + C(site) + asd", data=pop, family=sm.families.Binomial()).fit()
    pop["p"] = lm.predict(pop)
    pop["w"] = 1 / pop.p
    inc = pop[pop.y == 1].copy()
    lo, hi = inc.w.quantile([0.01, 0.99])
    inc["w"] = inc.w.clip(lo, hi)
    inc["w"] = inc.w / inc.w.mean()
    wmap = inc.set_index(["cohort", "sub_n"]).w
    ww = wmap.reindex(pd.MultiIndex.from_frame(meta[["cohort", "sub_n"]])).values
    out["ipw"] = {"n_population": int(len(pop)), "n_included": int(len(inc)),
                  "or_asd_inclusion": round(float(np.exp(lm.params["asd"])), 3), "p_asd": float(lm.pvalues["asd"]),
                  "weight_range_after_trunc": [round(float(inc.w.min()), 3), round(float(inc.w.max()), 3)], "results": {}}
    for unit, coh in [("ABIDE_I", "abide1"), ("ABIDE_II", "abide2")]:
        for g, lab in [(2, "TC"), (1, "ASD")]:
            sel = ((meta.cohort == coh) & (meta.dx == g)).values & np.isfinite(ww)
            out["ipw"]["results"][f"{unit}_{lab}"] = summarize(w1.values[sel], w2.values[sel], 2000, ww[sel])
            print("ipw", unit, lab, out["ipw"]["results"][f"{unit}_{lab}"]["median_icc"], flush=True)

    # S6（偏离预设）
    z = pd.read_csv(NB / "B13_zrc_strength_full.csv.gz")
    z = z[z.site_ok & (z.dx == 1)]
    wz = z.pivot_table(index=["cohort", "sub_n", "site"], columns="feature", values="zrc").dropna()
    s = wz.index.to_frame(index=False)
    s["n_ext"] = (wz.abs() > 2).sum(1).values
    s["mean_abs_z"] = wz.abs().mean(1).values
    s = s.merge(qc[["cohort", "sub_n", "age", "sex", "meanfd_retained_fd20"]], on=["cohort", "sub_n"]) \
         .merge(ph[["cohort", "sub_n", "ados_css", "srs_raw_total"]], on=["cohort", "sub_n"], how="left")
    out["S6"] = {"deviation": "P1 无 ICC ≥ 0.5 的分区；改为两个全脑指标 + 逐分区探索（BH-FDR）", "n_asd": int(len(s)), "results": {}}
    for sym in ["ados_css", "srs_raw_total"]:
        d = s.dropna(subset=[sym]).copy()
        res = {"n": int(len(d)), "descriptive_only_lt100": bool(len(d) < 100)}
        for ind in ["n_ext", "mean_abs_z"]:
            dd = d.copy()
            for c in [ind, sym]:
                dd[c] = (dd[c] - dd[c].mean()) / dd[c].std()
            fit = smf.ols(f"{ind} ~ {sym} + age + C(sex) + meanfd_retained_fd20 + C(site)", data=dd).fit()
            res[ind] = {"std_beta": round(float(fit.params[sym]), 4), "ci95": [round(float(x), 4) for x in fit.conf_int().loc[sym]],
                        "p": float(fit.pvalues[sym])}
        key = pd.MultiIndex.from_frame(d[["cohort", "sub_n", "site"]])
        pv = np.array([spearmanr(wz.loc[key, f].values, d[sym].values)[1] for f in wz.columns])
        rho = np.array([spearmanr(wz.loc[key, f].values, d[sym].values)[0] for f in wz.columns])
        q = multipletests(pv, method="fdr_bh")[1]
        res["parcelwise"] = {"n_q_lt_0.05": int((q < 0.05).sum()), "min_q": round(float(q.min()), 4),
                             "max_abs_rho": round(float(np.abs(rho).max()), 4)}
        out["S6"]["results"][sym] = res
        print("S6", sym, res, flush=True)

    (ROOT / "results/B17_sens_males_ipw_S6.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
