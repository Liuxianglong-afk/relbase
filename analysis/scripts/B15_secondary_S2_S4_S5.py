"""课题 B1 次要终点 S2、S4、S5（protocol_B1 v1.0 第 5 节；S4 按第 12 节登记的偏离预设执行）。

输入: derivatives/norms_B1/B13_zrc_{strength,thickness}_{full,half1,half2}.csv.gz（只用 site_ok）、B14_icc_per_feature.csv、
      results/B14_reliability.json、derivatives/features_B/B11_meta.csv、data/interim/abide_rest_qc_*.csv
S2 信度与时长:
  被试层面: 前后两半 strength z 图（379 分区）的空间 Pearson r ~ 每半保留秒数。GAM（statsmodels GLMGam，B 样条 df=6）
            加队列与诊断；另报 Spearman ρ 与按秒数五分位的中位数。
  Spearman–Brown: 以 P1 中位 ICC（及其 95% CI 两端）为半长 L（每队列×组的半长中位秒数）的信度，
            求达到 0.5、0.75 所需倍数 k = t(1-ρ) / (ρ(1-t))，换算为秒。全长数据（2L）的预测信度 2ρ/(1+ρ)。
S4 结构–功能对应（偏离预设：P1 无分区 ICC ≥ 0.5，改用全部 360 个皮层分区）:
  每人厚度 z 与全长 FC 强度 z 跨分区的 Spearman ρ；组间比较 OLS ρ ~ 诊断 + 年龄 + 性别 + 平均 FD + 站点（每队列）。
  衰减校正: 每人 FC 偏离图的全长信度 = 两半空间 r 的 Spearman–Brown 2r/(1+r)；厚度信度取 1；
            ρ_corr = ρ / sqrt(rel)，rel < 0.1 的被试不做校正（另报人数）。报告未校正与校正后的中位数。
S5 校正后的异质性: 极端偏离单元 = (分区, 符号)，|z| > 2。
  (a) 全长数据的全部极端偏离；(b) 两半都出现、且同号的极端偏离（可重复极端）。
  组内两两 Jaccard（并集为空的配对不计），报告 ASD 与 TC 的中位数；ASD − TC 用队列内按站点分层置换诊断 10,000 次（seed 42）。
  事后补充（2026-10-31，已登记方案第 12 节）：预设中位数在各组均为 0（多数配对无共享极端），另报平均数及其置换 P。
输出: results/B15_secondary.json；derivatives/norms_B1/B15_subject_level.csv（不含原始编号以外的标识）
用法: .venv_B/Scripts/python.exe -I scripts/B15_secondary_S2_S4_S5.py [--nperm 10000]
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.gam.api import BSplines, GLMGam

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derivatives/norms_B1"


def wide(st, cond):
    z = pd.read_csv(NB / f"B13_zrc_{st}_{cond}.csv.gz")
    z = z[z.site_ok]
    return z.pivot_table(index=["cohort", "sub_n", "site", "dx"], columns="feature", values="zrc")


def sb_k(rho, t):
    return t * (1 - rho) / (rho * (1 - t))


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))


def jaccard_matrix(x):
    x = x.astype(np.float64)
    inter = x @ x.T
    s = x.sum(1)
    union = s[:, None] + s[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def group_median(j, ix):
    sub = j[np.ix_(ix, ix)]
    v = sub[np.triu_indices(len(ix), 1)]
    return float(np.nanmedian(v)), int(np.isfinite(v).sum()), float(np.nanmean(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=10000)
    args = ap.parse_args()
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B15_secondary_S2_S4_S5.py"}

    full, h1, h2 = wide("strength", "full"), wide("strength", "half1"), wide("strength", "half2")
    idx = full.index.intersection(h1.index).intersection(h2.index)
    cols = full.columns
    full, h1, h2 = full.loc[idx, cols], h1.loc[idx, cols], h2.loc[idx, cols]
    ok = full.notna().all(1) & h1.notna().all(1) & h2.notna().all(1)
    full, h1, h2 = full[ok], h1[ok], h2[ok]
    meta = full.index.to_frame(index=False)
    qc = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_rest_qc_*.csv"))[-1])
    m11 = pd.read_csv(ROOT / "derivatives/features_B/B11_meta.csv")
    meta = meta.merge(qc[["cohort", "sub_n", "age", "sex", "meanfd_retained_fd20"]], on=["cohort", "sub_n"], how="left") \
               .merge(m11[["cohort", "sub_n", "sec_half"]], on=["cohort", "sub_n"], how="left")
    meta["group"] = np.where(meta.dx == 1, "ASD", "TC")

    # ---------- S2 ----------
    meta["r_half"] = rowcorr(h1.values, h2.values)
    meta["rel_full_sb"] = 2 * meta.r_half / (1 + meta.r_half)
    s2 = {"n": int(len(meta)), "r_half_median_by_cohort_group": {}, "spearman_r_vs_sec": {}, "quintiles": {}, "gam": {}, "spearman_brown": {}}
    for (c, g), d in meta.groupby(["cohort", "group"]):
        s2["r_half_median_by_cohort_group"][f"{c}_{g}"] = round(float(d.r_half.median()), 4)
    rho, p = spearmanr(meta.sec_half, meta.r_half)
    s2["spearman_r_vs_sec"] = {"rho": round(float(rho), 4), "p": float(p)}
    meta["sec_q"] = pd.qcut(meta.sec_half, 5)
    s2["quintiles"] = {str(k): {"median_sec": round(float(v.sec_half.median()), 1), "median_r_half": round(float(v.r_half.median()), 4), "n": int(len(v))}
                       for k, v in meta.groupby("sec_q", observed=True)}
    bs = BSplines(meta[["sec_half"]], df=[6], degree=[3])
    gam = GLMGam.from_formula("r_half ~ C(cohort) + C(group)", data=meta, smoother=bs).fit()
    grid = np.linspace(meta.sec_half.quantile(0.02), meta.sec_half.quantile(0.98), 9)
    pred = pd.DataFrame({"sec_half": grid, "cohort": "abide1", "group": "TC"})
    s2["gam"] = {"pred_grid_sec": [round(float(x), 1) for x in grid],
                 "pred_r_half_abide1_TC": [round(float(x), 4) for x in gam.predict(pred, exog_smooth=pred[["sec_half"]])],
                 "coef_group_TC_vs_ASD": round(float(gam.params.get("C(group)[T.TC]", np.nan)), 4),
                 "p_group": float(gam.pvalues.get("C(group)[T.TC]", np.nan))}
    b14 = json.loads((ROOT / "results/B14_reliability.json").read_text(encoding="utf-8"))["results"]["P1P2_strength_half"]
    for unit, coh in [("ABIDE_I", "abide1"), ("ABIDE_II", "abide2")]:
        for g in ["TC", "ASD"]:
            r = b14[unit][g]
            L = float(meta[(meta.cohort == coh) & (meta.group == g)].sec_half.median())
            row = {"half_sec_median": round(L, 1), "icc_half": r["median_icc"],
                   "pred_icc_full_2L": round(2 * r["median_icc"] / (1 + r["median_icc"]), 4)}
            for t in [0.5, 0.75]:
                row[f"sec_needed_icc_{t}"] = round(sb_k(r["median_icc"], t) * L, 0)
                lo, hi = r["ci95_mean_over_seeds"]
                row[f"sec_needed_icc_{t}_range_from_ci"] = [round(sb_k(hi, t) * L, 0), round(sb_k(lo, t) * L, 0)]
            s2["spearman_brown"][f"{unit}_{g}"] = row
    out["S2"] = s2

    # ---------- S4（偏离预设） ----------
    th = wide("thickness", "full")
    ctx = [c for c in th.columns if c in cols]
    common = meta.set_index(["cohort", "sub_n", "site", "dx"]).index.intersection(th.index)
    fz = full.loc[common, ctx].values
    tz = th.loc[common, ctx].values
    good = np.isfinite(tz).all(1)
    rhos = np.array([spearmanr(fz[i], tz[i])[0] if good[i] else np.nan for i in range(len(common))])
    s4df = common.to_frame(index=False)
    s4df["rho_sf"] = rhos
    s4df = s4df.merge(meta[["cohort", "sub_n", "age", "sex", "meanfd_retained_fd20", "group", "rel_full_sb"]], on=["cohort", "sub_n"])
    s4df["rho_corr"] = np.where(s4df.rel_full_sb >= 0.1, s4df.rho_sf / np.sqrt(s4df.rel_full_sb.clip(lower=0.1)), np.nan)
    s4 = {"deviation": "P1 无分区 ICC ≥ 0.5，改为全部皮层分区 + 被试层面衰减校正（protocol_B1 第 12 节）",
          "n_parcels": len(ctx), "n_subjects": int(np.isfinite(rhos).sum()),
          "n_not_corrected_rel_lt_0.1": int((s4df.rel_full_sb < 0.1).sum()), "by_cohort": {}}
    for coh, d in s4df.dropna(subset=["rho_sf"]).groupby("cohort"):
        res = {g: {"n": int(len(x)), "median_rho": round(float(x.rho_sf.median()), 4),
                   "iqr_rho": [round(float(v), 4) for v in x.rho_sf.quantile([0.25, 0.75])],
                   "median_rho_corrected": round(float(x.rho_corr.median()), 4),
                   "median_rel_full_sb": round(float(x.rel_full_sb.median()), 4)} for g, x in d.groupby("group")}
        fit = smf.ols("rho_sf ~ C(group, Treatment('TC')) + age + C(sex) + meanfd_retained_fd20 + C(site)", data=d).fit()
        term = "C(group, Treatment('TC'))[T.ASD]"
        res["ols_ASD_minus_TC"] = {"beta": round(float(fit.params[term]), 4),
                                   "ci95": [round(float(x), 4) for x in fit.conf_int().loc[term]], "p": float(fit.pvalues[term])}
        s4["by_cohort"][coh] = res
    out["S4"] = s4

    # ---------- S5 ----------
    def ext(w):
        v = w.values
        return np.hstack([v > 2, v < -2])

    ef, e1, e2 = ext(full), ext(h1), ext(h2)
    sets = {"a_full_all_extremes": ef, "b_reproducible_both_halves": e1 & e2}
    s5 = {}
    for coh in ["abide1", "abide2"]:
        sel = np.where(meta.cohort.values == coh)[0]
        g = meta.dx.values[sel]
        site = meta.site.values[sel]
        site_ix = [np.where(site == s)[0] for s in np.unique(site)]
        s5[coh] = {}
        for name, x in sets.items():
            xs = x[sel]
            j = jaccard_matrix(xs)
            ia, it = np.where(g == 1)[0], np.where(g == 2)[0]
            ma, na, mna = group_median(j, ia)
            mt, nt, mnt = group_median(j, it)
            obs, obs_mean = ma - mt, mna - mnt
            rng = np.random.default_rng(42)
            null, null_mean = np.empty(args.nperm), np.empty(args.nperm)
            for k in range(args.nperm):
                gp = g.copy()
                for ix in site_ix:
                    gp[ix] = rng.permutation(gp[ix])
                ra, rt = group_median(j, np.where(gp == 1)[0]), group_median(j, np.where(gp == 2)[0])
                null[k], null_mean[k] = ra[0] - rt[0], ra[2] - rt[2]
            s5[coh][name] = {"ASD": {"n": int(len(ia)), "median_jaccard": round(ma, 4), "n_valid_pairs": na,
                                     "share_subjects_no_extreme": round(float((xs[ia].sum(1) == 0).mean()), 4),
                                     "median_n_extremes": float(np.median(xs[ia].sum(1)))},
                             "TC": {"n": int(len(it)), "median_jaccard": round(mt, 4), "n_valid_pairs": nt,
                                    "share_subjects_no_extreme": round(float((xs[it].sum(1) == 0).mean()), 4),
                                    "median_n_extremes": float(np.median(xs[it].sum(1)))},
                             "ASD_minus_TC": round(obs, 4),
                             "perm_p_two_sided": round(float((np.sum(np.abs(null) >= abs(obs)) + 1) / (args.nperm + 1)), 5),
                             # 事后（预设中位数在两组均为 0，触底无信息）：两两 Jaccard 平均数及同一置换下的检验
                             "posthoc_mean_jaccard": {"ASD": round(mna, 4), "TC": round(mnt, 4), "ASD_minus_TC": round(obs_mean, 4),
                                                      "perm_p_two_sided": round(float((np.sum(np.abs(null_mean) >= abs(obs_mean)) + 1) / (args.nperm + 1)), 5)}}
            print("S5", coh, name, s5[coh][name], flush=True)
    out["S5"] = s5

    meta.drop(columns=["sec_q"]).merge(s4df[["cohort", "sub_n", "rho_sf", "rho_corr"]], on=["cohort", "sub_n"], how="left") \
        .to_csv(NB / "B15_subject_level.csv", index=False)
    (ROOT / "results/B15_secondary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["S2", "S4"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
