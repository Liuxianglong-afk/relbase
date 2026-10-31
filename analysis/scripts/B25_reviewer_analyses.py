"""课题 B1 事后补充分析（2026-10-31，回应内审意见 manuscript_B1/review_internal_v0.1_20261031.md；登记于方案第 12 节）。

a. Spearman–Brown 外推的外部验证：在跨 run 子样本中，用同一批被试的分半 ICC 按 SB 预测 run 长度下的信度，与实测跨 run ICC 比较
   （逐分区预测后取中位数；被试层面 bootstrap 2,000 次 × 种子 42/7/123/2024，报告观测 − 预测之差的 95% CI）。
b. P2 的机会水平：站点内打乱第二半的被试对应关系（保留分区与站点结构），1,000 次，得正向一致率的零分布。
c. 极端阈值敏感性：|z| > 1.5 / 2.0 / 2.5 / 3.0 下的 P2（及其机会水平，200 次置换）、relbase 基线（s = 0 与 s = 1）、
   跨站点 Jaccard 实测值及相对 s = 0 的单侧检验（500 次）。
d. 年龄与信度：被试层面两半图相关 r_half ~ 年龄 + 每半时长 + 平均 FD + 组别 + 队列（OLS），另加年龄 × 组别；
   按全样本年龄三分位分层的中位 ICC（bootstrap 2,000 次 × 4 种子）。
e. 可检出的最小共享比例：在全长信度下（逐分区），对 ABIDE I、II 的 ASD 组（各自 n 与 z 标准差），
   s = 0.02–0.20 各 200 次模拟，计算超过 s = 0 零分布 95% 分位数的比例（检验效能），取效能 ≥ 0.8 的最小 s。
输出: results/B25_reviewer_analyses.json
用法: .venv_B/Scripts/python.exe -I scripts/B25_reviewer_analyses.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "relbase"))
from relbase import BaselineModel, estimate_inputs, icc21, mean_jaccard, spearman_brown  # noqa: E402

NB = ROOT / "derivatives/norms_B1"
SEEDS = [42, 7, 123, 2024]
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


def boot_ci(fn, n, nboot=2000):
    """fn(idx) -> 统计量；按种子分别取 2.5/97.5 分位，报告跨种子平均的界。"""
    lo, hi = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        v = np.array([fn(rng.integers(0, n, n)) for _ in range(nboot)])
        lo.append(np.quantile(v, 0.025))
        hi.append(np.quantile(v, 0.975))
    return [round(float(np.mean(lo)), 4), round(float(np.mean(hi)), 4)]


def site_perm(site, rng):
    perm = np.arange(len(site))
    for s in np.unique(site):
        ix = np.where(site == s)[0]
        perm[ix] = rng.permutation(ix)
    return perm


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B25_reviewer_analyses.py"}
    full, h1, h2 = wide("full"), wide("half1"), wide("half2")
    idx = full.index.intersection(h1.index).intersection(h2.index)
    full, h1, h2 = full.loc[idx], h1.loc[idx], h2.loc[idx]
    dx, coh, site = (idx.get_level_values(k).to_numpy() for k in ["dx", "cohort", "site"])
    groups = {f"{g}_{c}": (dx == code) & (coh == c) for g, code in [("ASD", 1), ("TC", 2)] for c in ["abide1", "abide2"]}
    meta = pd.read_csv(ROOT / "derivatives/features_B/B11_meta.csv").set_index(["cohort", "sub_n"])

    # 自检：P2 与 B14 一致
    b14 = json.loads((ROOT / "results/B14_reliability.json").read_text(encoding="utf-8"))["results"]["P1P2_strength_half"]
    chk = {}
    for k, m in groups.items():
        g, c = k.split("_")
        unit = {"abide1": "ABIDE_I", "abide2": "ABIDE_II"}[c]
        chk[k] = [round(pos_agreement(h1.values[m], h2.values[m]), 4), b14[unit][g]["extreme"]["pos_agreement_mean_dir"]]
    out["selfcheck_p2_vs_B14"] = chk
    print("selfcheck", chk, flush=True)

    # a. SB 外推验证（跨 run 子样本）
    r1, r2 = wide("run1"), wide("run2")
    ridx = r1.index.intersection(r2.index).intersection(idx)
    a_res = {}
    for g, code in [("TC", 2), ("ASD", 1), ("all", None)]:
        sel = ridx if code is None else ridx[ridx.get_level_values("dx") == code]
        H1, H2, R1, R2 = h1.loc[sel].values, h2.loc[sel].values, r1.loc[sel].values, r2.loc[sel].values
        keys = list(zip(sel.get_level_values("cohort"), sel.get_level_values("sub_n")))
        L_half = meta.loc[keys, "sec_half"].to_numpy(float)
        L_run = meta.loc[keys, ["sec_run1", "sec_run2"]].min(axis=1).to_numpy(float)

        def stat(ix):
            f = np.median(L_run[ix]) / np.median(L_half[ix])
            pred = np.median(spearman_brown(np.clip(icc21(H1[ix], H2[ix]), 0, 1), f))
            return float(np.median(icc21(R1[ix], R2[ix])) - pred)
        allix = np.arange(len(sel))
        f = float(np.median(L_run) / np.median(L_half))
        a_res[g] = {"n": int(len(sel)), "median_sec_half": float(np.median(L_half)), "median_sec_run": float(np.median(L_run)),
                    "length_factor": round(f, 3),
                    "median_icc_half": round(float(np.median(icc21(H1, H2))), 4),
                    "predicted_median_icc_run": round(float(np.median(spearman_brown(np.clip(icc21(H1, H2), 0, 1), f))), 4),
                    "observed_median_icc_run": round(float(np.median(icc21(R1, R2))), 4),
                    "observed_minus_predicted": round(stat(allix), 4),
                    "ci95_diff": boot_ci(stat, len(sel))}
        print("a", g, a_res[g], flush=True)
    out["a_sb_validation_between_run"] = a_res

    # b/c. P2 机会水平与阈值敏感性
    est = estimate_inputs(full.values[dx == 2], h1.values[dx == 2], h2.values[dx == 2])
    rel = np.clip(est["reliability_full"], 0, 1)
    R_obs, R_noise = est["R_signal"], est["R_noise"]
    bc = {}
    for thr in [1.5, 2.0, 2.5, 3.0]:
        row = {}
        nperm = 1000 if thr == 2.0 else 200
        for k, m in groups.items():
            A, B, s = h1.values[m], h2.values[m], site[m]
            obs = pos_agreement(A, B, thr)
            rng = np.random.default_rng(42)
            null = np.array([pos_agreement(A, B[site_perm(s, rng)], thr) for _ in range(nperm)])
            row[k] = {"extreme_rate_half1": round(float((np.abs(A) > thr).mean()), 4), "p2_observed": round(obs, 4),
                      "p2_chance_mean": round(float(null.mean()), 4), "p2_chance_q975": round(float(np.quantile(null, 0.975)), 4),
                      "ratio_to_chance": round(obs / null.mean(), 2) if null.mean() > 0 else None,
                      "jaccard_between_site": round(mean_jaccard(full.values[m], thr, blocks=site[m]), 5)}
        model = BaselineModel(R_obs, R_noise, scale=float(full.values[dx == 1].std()), threshold=thr)
        curve = model.baseline_curve(201, rel, s_grid=(0.0, 1.0), reps=25)
        row["baseline_s0_s1"] = [round(curve[0]["mean"], 5), round(curve[1]["mean"], 5)]
        for k, m in groups.items():
            gm = BaselineModel(R_obs, R_noise, scale=float(full.values[m].std()), threshold=thr)
            t = gm.test_heterogeneous_null(row[k]["jaccard_between_site"], int(m.sum()), rel, reps=500, seed=42)
            row[k]["null_mean"] = round(t["null_mean"], 5)
            row[k]["p_one_sided_vs_s0"] = round(t["p_one_sided"], 4)
        bc[f"thr={thr}"] = row
        print("bc", thr, {k: (v["p2_observed"], v["p2_chance_mean"], v.get("p_one_sided_vs_s0")) for k, v in row.items() if k != "baseline_s0_s1"},
              row["baseline_s0_s1"], flush=True)
    out["bc_chance_and_threshold"] = bc

    # d. 年龄
    sl = pd.read_csv(NB / "B15_subject_level.csv")
    fit = smf.ols("r_half ~ age + sec_half + meanfd_retained_fd20 + C(group) + C(cohort)", data=sl).fit()
    fit2 = smf.ols("r_half ~ age * C(group) + sec_half + meanfd_retained_fd20 + C(cohort)", data=sl).fit()
    inter = [k for k in fit2.params.index if ":" in k][0]
    d_res = {"n": int(len(sl)), "age_range": [float(sl.age.min()), float(sl.age.max())],
             "beta_age_per_year": round(float(fit.params["age"]), 5), "ci95": [round(float(x), 5) for x in fit.conf_int().loc["age"]],
             "p_age": round(float(fit.pvalues["age"]), 4),
             "beta_age_x_group": round(float(fit2.params[inter]), 5), "p_age_x_group": round(float(fit2.pvalues[inter]), 4)}
    meta_idx = pd.DataFrame(index=idx).reset_index().merge(sl[["cohort", "sub_n", "age"]], on=["cohort", "sub_n"], how="left")
    age = meta_idx.age.to_numpy(float)
    cuts = np.nanquantile(age, [1 / 3, 2 / 3])
    tert = np.digitize(age, cuts)
    strata = {}
    for t, lab in enumerate(["lower", "middle", "upper"]):
        for g, code in [("TC", 2), ("ASD", 1)]:
            m = (tert == t) & (dx == code)
            A, B = h1.values[m], h2.values[m]
            strata[f"{lab}_{g}"] = {"n": int(m.sum()), "age_range": [round(float(age[m].min()), 1), round(float(age[m].max()), 1)],
                                   "median_icc": round(float(np.median(icc21(A, B))), 4),
                                   "ci95": boot_ci(lambda ix: float(np.median(icc21(A[ix], B[ix]))), int(m.sum()))}
    d_res["age_tertile_cuts"] = [round(float(c), 2) for c in cuts]
    d_res["tertile_median_icc"] = strata
    out["d_age"] = d_res
    print("d", json.dumps(d_res, ensure_ascii=False), flush=True)

    # e. 可检出的最小共享比例
    e_res = {}
    for k in ["ASD_abide1", "ASD_abide2"]:
        m = groups[k]
        n, sd = int(m.sum()), float(full.values[m].std())
        gm = BaselineModel(R_obs, R_noise, scale=sd)
        q95 = float(np.quantile(gm.overlap_distribution(n, rel, 0.0, reps=1000, seed=42), 0.95))
        power = {}
        for s in np.round(np.arange(0.02, 0.21, 0.02), 2):
            alt = gm.overlap_distribution(n, rel, float(s), reps=200, seed=1000 + int(s * 100))
            power[float(s)] = round(float((alt > q95).mean()), 3)
        ss, pw = np.array(list(power)), np.array(list(power.values()))
        s80 = float(np.interp(0.8, pw, ss)) if pw.max() >= 0.8 and np.all(np.diff(pw) >= -0.05) else None
        e_res[k] = {"n": n, "null_q95": round(q95, 5), "power_by_s": power, "min_detectable_s_power80": None if s80 is None else round(s80, 3)}
        print("e", k, e_res[k], flush=True)
    out["e_detectable_shared_fraction"] = e_res

    (ROOT / "results/B25_reviewer_analyses.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
