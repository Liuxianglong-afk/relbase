"""课题 B1 敏感性分析 4：特征层面 ComBat-GAM（neuroHarmonize 2.4.5）代替按站点重新居中。

样本: 主分析样本（pass_fd20）中位于 site_ok 站点（本地 TC ≥ 10）者。
每个数据长度（full / half1 / half2）分别协调 379 个 FC 强度：
  只用 TC 学习（SITE + 年龄平滑项 + 性别 + 平均 FD，eb=True，seed 42），再用学到的模型应用到 ASD，避免诊断信息进入协调模型。
之后 B12（变体 combat）重新拟合常模，B13 不再重新居中。
输出: derivatives/features_B/combat/B11_strength_{full,half1,half2}.csv；results/B11c_combat.json
用法: .venv_B/Scripts/python.exe -I scripts/B11c_combat_gam.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn

ROOT = Path(__file__).resolve().parents[1]
FB = ROOT / "derivatives/features_B"
MIN_TC = 10


def main():
    qc = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_rest_qc_*.csv"))[-1])
    cov = qc.loc[qc.pass_fd20, ["cohort", "sub_n", "site", "dx_group", "age", "sex", "meanfd_retained_fd20"]]
    ntc = cov[cov.dx_group == 2].groupby("site").size()
    cov = cov[cov.site.isin(ntc[ntc >= MIN_TC].index)]
    out = FB / "combat"
    out.mkdir(parents=True, exist_ok=True)
    summ = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B11c_combat_gam.py", "conditions": {}}
    for c in ["full", "half1", "half2"]:
        f = pd.read_csv(FB / f"B11_strength_{c}.csv")
        d = cov.merge(f, on=["cohort", "sub_n"])
        feats = [x for x in f.columns if x not in ("cohort", "sub_n")]
        cv = pd.DataFrame({"SITE": d.site.values, "AGE": d.age.values, "SEX": (d.sex == 2).astype(float).values,
                           "FD": d.meanfd_retained_fd20.values})
        tc = (d.dx_group == 2).values
        model, adj_tc = harmonizationLearn(d.loc[tc, feats].values, cv[tc].reset_index(drop=True), smooth_terms=["AGE"],
                                            smooth_term_bounds=(float(cv.AGE.min()) - 0.01, float(cv.AGE.max()) + 0.01), seed=42)
        adj_asd = harmonizationApply(d.loc[~tc, feats].values, cv[~tc].reset_index(drop=True), model)
        res = d[["cohort", "sub_n"]].copy()
        arr = np.empty((len(d), len(feats)))
        arr[tc], arr[~tc] = adj_tc, adj_asd
        res = pd.concat([res.reset_index(drop=True), pd.DataFrame(arr, columns=feats)], axis=1)
        res.to_csv(out / f"B11_strength_{c}.csv", index=False)
        # 协调效果：TC 中各特征站点均值的标准差（中位数），协调前后对比
        before = d.loc[tc].groupby("site")[feats].mean().std().median()
        after = pd.DataFrame(adj_tc, columns=feats).assign(site=d.site.values[tc]).groupby("site")[feats].mean().std().median()
        summ["conditions"][c] = {"n": int(len(d)), "n_tc": int(tc.sum()), "median_between_site_sd_tc_before": round(float(before), 5),
                                 "after": round(float(after), 5)}
        print(c, summ["conditions"][c], flush=True)
    (ROOT / "results/B11c_combat.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
