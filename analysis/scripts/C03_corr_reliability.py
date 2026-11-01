"""CoRR：同一批被试中，会话内（分半）与跨会话（重测）个体偏离信度的比较（课题 B1 事后外部分析，2026-11-01 登记）。

输入: derivatives/corr/C02_strength_{full,half1,half2}.csv、C02_meta.csv；data/raw/corr/corr_pheno_clean.csv
常模: 参考样本 = 全部通过质控者的第一次会话（ses-1）；线性常模（年龄 B 样条 df = 3 + 性别 + 保留帧平均 FD + 站点哑变量），
      全长、前半、后半分别建模；五折交叉拟合（按站点分层，种子 42），ses-1 与 ses-2 均由不含本人的那一折模型打分。
      CoRR 多数站点人数少，不能做留一站点居中，站点改作协变量（与 ABIDE 主分析不同，论文中注明）。
比较（≥ 2 个会话且两次均通过质控的被试；377 个分区，L/R_H_ROI 缺失；其余分区有缺失的会话整体排除）:
  a. 会话内分半 ICC(2,1) 中位数（ses-1 前半 vs 后半；ses-2 同）
  b. 跨会话、等长：ses-1 前半 vs ses-2 前半（与 a 时长相同）
  c. 跨会话、全长：ses-1 全长 vs ses-2 全长
  d. b − a 的配对 bootstrap 95% CI（被试层面 1,000 次 × 种子 42/7/123/2024，跨种子平均界）
  e. 极端偏离（|z| > 2）正向一致率：会话内 vs 跨会话（等长）
  f. 原始 FC 强度（未建模）的同样比较
  g. 按重测间隔分层（≤ 60 天 vs > 60 天）
输出: results/C03_corr_reliability.json
用法: .venv_B/Scripts/python.exe -I scripts/C03_corr_reliability.py [--variant nogsr]
  --variant nogsr: 读 derivatives/corr/nogsr/（不回归全脑信号），输出 results/C03_corr_reliability_nogsr.json
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "relbase"))
from relbase import icc21  # noqa: E402

D = ROOT / "derivatives/corr"
SEEDS = [42, 7, 123, 2024]
sys.stdout.reconfigure(encoding="utf-8")


def pos_agreement(a, b, thr=2.0):
    ca, cb = np.sign(a) * (np.abs(a) > thr), np.sign(b) * (np.abs(b) > thr)
    both = ((ca != 0) & (ca == cb)).sum()
    return float((both / max((ca != 0).sum(), 1) + both / max((cb != 0).sum(), 1)) / 2)


def med_icc(a, b):
    return float(np.median(icc21(a, b)))


def boot_diff(fn_between, fn_within, n, nboot=1000):
    lo, hi = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        v = []
        for _ in range(nboot):
            ix = rng.integers(0, n, n)
            v.append(fn_between(ix) - fn_within(ix))
        lo.append(np.quantile(v, 0.025))
        hi.append(np.quantile(v, 0.975))
    return [round(float(np.mean(lo)), 4), round(float(np.mean(hi)), 4)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    variant = ap.parse_args().variant
    D = ROOT / "derivatives/corr" if variant == "main" else ROOT / "derivatives/corr" / variant
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/C03_corr_reliability.py", "variant": variant}
    meta = pd.read_csv(D / "C02_meta.csv")
    feat = {c: pd.read_csv(D / f"C02_strength_{c}.csv") for c in ["full", "half1", "half2"]}
    cols = [c for c in feat["full"].columns[2:] if c not in ("L_H_ROI", "R_H_ROI")]
    # 少数会话因眶额/颞极信号丢失有分区无顶点数据：排除这些会话，保留全部 377 个分区
    bad = set()
    for v in feat.values():
        bad |= set(map(tuple, v.loc[v[cols].isna().any(axis=1), ["sub", "ses"]].to_numpy()))
    out["sessions_processed"], out["sessions_pass_qc_before_exclusion"] = int(len(meta)), int(meta.pass_qc.sum())
    meta = meta[[(s, e) not in bad for s, e in zip(meta["sub"], meta.ses)]]
    out["sessions_excluded_missing_parcels"] = len(bad)
    ph = pd.read_csv(ROOT / "data/raw/corr/corr_pheno_clean.csv")
    ph["age"] = pd.to_numeric(ph.AGE_AT_SCAN_1, errors="coerce")
    base = ph.drop_duplicates("SUBID").set_index("SUBID")
    dur = ph[ph.SESSION != "Baseline"].groupby("SUBID").agg(dur=("RETEST_DURATION", "first"), unit=("RETEST_UNITS", "first"))

    def days(r):
        try:
            v = float(r.dur)
        except (TypeError, ValueError):
            return np.nan
        return v * {"d": 1, "m": 30.4, "y": 365.25, "h": 1 / 24, "w": 7}.get(str(r.unit).strip().lower()[:1], np.nan)
    dur["days"] = dur.apply(days, axis=1)

    m = meta[meta.pass_qc].copy()
    m["SUBID"] = m["sub"].str[4:].astype(int)
    m = m[m.SUBID.isin(base.index)]
    m["age"] = m.SUBID.map(base.age)
    m["sex"] = m.SUBID.map(base.SEX).astype(str)
    m["site"] = m.SUBID.map(base.SITE)
    ses1 = m[m.ses == "ses-1"].reset_index(drop=True)
    retest = m[m.ses.isin(["ses-1", "ses-2"])].groupby("SUBID").ses.nunique()
    retest = retest[retest == 2].index
    out["n"] = {"reference_ses1": int(len(ses1)), "retest_both_pass": int(len(retest)), "sites_reference": int(ses1.site.nunique()),
                "parcels": len(cols), "age_range_retest": [float(m[m.SUBID.isin(retest)].age.min()), float(m[m.SUBID.isin(retest)].age.max())],
                "retest_days_median_iqr": [round(float(x), 1) for x in dur.loc[retest, "days"].quantile([0.5, 0.25, 0.75])],
                "sessions_pass_qc": int(len(m)), "age_median_retest": float(base.loc[retest, "age"].median()),
                "sites_retest": {k: int(v) for k, v in base.loc[retest, "SITE"].value_counts().items()}}
    mr = m[m.SUBID.isin(retest) & m.ses.isin(["ses-1", "ses-2"])]
    out["n"]["sec_full_median_iqr"] = [int(round(float(x))) for x in mr.sec_full.quantile([0.5, 0.25, 0.75])]
    out["n"]["sec_half_median"] = round(float(mr.sec_half.median()), 1)
    print("n", out["n"], flush=True)

    # 五折（按站点分层）
    rng = np.random.default_rng(42)
    fold = pd.Series(0, index=ses1.SUBID)
    for s, g in ses1.groupby("site"):
        fold[g.SUBID.values] = rng.permutation(np.arange(len(g)) % 5)

    def design(df, sites):
        d = df.assign(site=pd.Categorical(df.site, categories=sites))
        f = "bs(age, df=3, lower_bound=%f, upper_bound=%f) + C(sex) + meanfd_retained + C(site)" % (ses1.age.min() - 1, ses1.age.max() + 1)
        return np.asarray(dmatrix(f, d, return_type="dataframe"))
    sites = sorted(ses1.site.unique())
    Z = {}
    for c in ["full", "half1", "half2"]:
        F = feat[c].merge(m[["sub", "ses", "SUBID", "age", "sex", "site", "meanfd_retained"]], on=["sub", "ses"])
        Xall = design(F, sites)
        Y = F[cols].to_numpy(float)
        z = np.full_like(Y, np.nan)
        is1 = (F.ses == "ses-1").to_numpy()
        fF = F.SUBID.map(fold).to_numpy()
        for f in range(5):
            tr = is1 & (fF != f)
            B, *_ = np.linalg.lstsq(Xall[tr], Y[tr], rcond=None)
            sd = (Y[tr] - Xall[tr] @ B).std(0, ddof=Xall.shape[1])
            te = fF == f  # 无 ses-1 参考数据者不打分（不在重测样本中）
            z[te] = (Y[te] - Xall[te] @ B) / sd
        Z[c] = F[["SUBID", "ses"]].assign(**{"_row": range(len(F))}).set_index(["SUBID", "ses"]), z, Y
    def get(c, ses, ids, raw=False):
        idx, z, Y = Z[c]
        rows = idx.loc[[(i, ses) for i in ids], "_row"].to_numpy()
        return (Y if raw else z)[rows]

    def compare(ids, raw=False):
        a1, b1 = get("half1", "ses-1", ids, raw), get("half2", "ses-1", ids, raw)
        a2, b2 = get("half1", "ses-2", ids, raw), get("half2", "ses-2", ids, raw)
        f1, f2 = get("full", "ses-1", ids, raw), get("full", "ses-2", ids, raw)
        within = lambda ix: (med_icc(a1[ix], b1[ix]) + med_icc(a2[ix], b2[ix])) / 2
        between = lambda ix: med_icc(a1[ix], a2[ix])
        allix = np.arange(len(ids))
        r = {"n": len(ids), "within_ses1": round(med_icc(a1, b1), 4), "within_ses2": round(med_icc(a2, b2), 4),
             "within_mean": round(within(allix), 4), "between_equal_length": round(between(allix), 4),
             "between_full_length": round(med_icc(f1, f2), 4),
             "between_minus_within": round(between(allix) - within(allix), 4),
             "ci95_between_minus_within": boot_diff(between, within, len(ids)),
             "ratio_between_to_within": round(between(allix) / within(allix), 3)}
        if not raw:
            r["p2_within"] = round((pos_agreement(a1, b1) + pos_agreement(a2, b2)) / 2, 4)
            r["p2_between_equal_length"] = round(pos_agreement(a1, a2), 4)
        return r
    ids = list(retest)
    out["deviation_z"] = compare(ids)
    print("z", out["deviation_z"], flush=True)
    out["raw_strength"] = compare(ids, raw=True)
    print("raw", out["raw_strength"], flush=True)
    d = dur.loc[ids, "days"]
    for lab, sel in [("interval_le_60d", d <= 60), ("interval_gt_60d", d > 60)]:
        sub = list(d[sel].index)
        if len(sub) >= 20:
            out[lab] = compare(sub)
            print(lab, out[lab], flush=True)
    (ROOT / ("results/C03_corr_reliability.json" if variant == "main" else f"results/C03_corr_reliability_{variant}.json")).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
