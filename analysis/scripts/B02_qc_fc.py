"""DCAN ptseries 的质控与 FC 计算（protocol_B 第 5、6、9 节）。

剔帧: DCAN motion_mask.mat 中 FD_threshold = 0.20（主分析）与 0.30（敏感性）两项；frame_removal == 1 为剔除，
      开头 skip 帧已计入。ptseries 帧数须等于 total_frame_count，否则记为排除。
剩余时长门槛（不看诊断，预先写定）: 全部 DCAN 齐全者中，FD 0.20 下剩余 ≥ 240 s 的比例 ≥ 60% 则用 240 s，否则用 180 s。
FC: 保留帧上的 Pearson r → Fisher z（r 截断于 ±0.9999），存上三角 float32。剩余 < 120 s 者不计算 FC。
FC 强度: 每个分区与其余分区正值 z 的均值。
预检（不涉及诊断）: TC 组平均 FC 上三角在 ABIDE I 与 II 之间的相关（门槛 r > 0.8）。

输出:
  data/interim/abide_rest_qc_<YYYYMMDD>.csv
  derivatives/fc/<atlas>_fd<thr>_fcz_upper.npy + _subjects.csv + _parcels.txt
  derivatives/fc/<atlas>_fd<thr>_strength.csv
  results/B02_qc_fc.json

用法: python -I scripts/B02_qc_fc.py [--limit N]
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
ATLASES = {"glasser": "HCP2016FreeSurferSubcortical", "gordon": "Gordon2014FreeSurferSubcortical"}
FD_THRS = (0.20, 0.30)
MIN_SEC_FC = 120
TODAY = dt.date.today().strftime("%Y%m%d")


def latest(pattern):
    files = sorted((ROOT / "data/interim").glob(pattern))
    if not files:
        raise SystemExit(f"未找到 {pattern}，先运行 scripts/B01_download_dcan.py")
    return files[-1]


def subj_dir(r):
    return ROOT / "data/raw/abide_dcan" / r.cohort / f"sub-{r.sub_n}" / r.ses


def load_mask(path):
    md = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["motion_data"]
    out = {}
    for thr in FD_THRS:
        e = min(md, key=lambda e: abs(float(e.FD_threshold) - thr))
        keep = np.asarray(e.frame_removal) == 0
        out[thr] = {"keep": keep, "tr": float(e.epi_TR), "total": int(e.total_frame_count),
                    "remain": int(e.remaining_frame_count), "sec": float(e.remaining_seconds),
                    "mean_fd": float(e.remaining_frame_mean_FD)}
    return out


def fcz(ts):
    r = np.corrcoef(ts.T)
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    man = pd.read_csv(latest("abide_dcan_manifest_*.csv"))
    man = man[man.complete].reset_index(drop=True)
    if a.limit:
        man = man.head(a.limit)

    qc_rows, fc = [], {(at, t): [] for at in ATLASES for t in FD_THRS}
    fc_subs = {k: [] for k in fc}
    parcels = {}
    for i, r in enumerate(man.itertuples(), 1):
        d = subj_dir(r)
        row = {"cohort": r.cohort, "site": r.site, "sub_n": r.sub_n, "dx_group": r.dx_group,
               "age": r.age, "sex": r.sex, "exclude_reason": ""}
        try:
            m = load_mask(next((d / "func").glob("*motion_mask.mat")))
        except Exception as e:
            row["exclude_reason"] = f"mask_error: {e}"
            qc_rows.append(row)
            continue
        row["tr"] = m[0.20]["tr"]
        row["total_frames"] = m[0.20]["total"]
        for t in FD_THRS:
            tag = f"fd{int(t * 100):02d}"
            row[f"remain_frames_{tag}"] = m[t]["remain"]
            row[f"remain_sec_{tag}"] = m[t]["sec"]
            row[f"meanfd_retained_{tag}"] = m[t]["mean_fd"]
        for at, an in ATLASES.items():
            img = nib.load(next((d / "func").glob(f"*atlas-{an}_desc-filtered_timeseries.ptseries.nii")))
            ts = img.get_fdata(dtype=np.float32)
            names = list(img.header.get_axis(1).name)
            if at not in parcels:
                parcels[at] = names
            if names != parcels[at]:
                row["exclude_reason"] += f"{at}_parcel_order;"
                continue
            if ts.shape[0] != m[0.20]["total"]:
                row["exclude_reason"] += f"{at}_frames_{ts.shape[0]}!={m[0.20]['total']};"
                continue
            bad = np.isnan(ts).any(0) | (ts.std(0) == 0)
            row[f"{at}_bad_parcels"] = int(bad.sum())
            for t in FD_THRS:
                if m[t]["sec"] < MIN_SEC_FC:
                    continue
                z = fcz(ts[m[t]["keep"]])
                iu = np.triu_indices(z.shape[0], 1)
                fc[(at, t)].append(z[iu].astype(np.float32))
                fc_subs[(at, t)].append((r.cohort, r.sub_n))
        qc_rows.append(row)
        if i % 100 == 0:
            print(f"  {i}/{len(man)}", flush=True)

    qc = pd.DataFrame(qc_rows)

    # 剩余时长门槛：只看总体合格率，不看诊断
    # 两个图谱都无 NaN/零方差分区才算合格，主分析与图谱敏感性用同一批人
    bad = (qc.get("glasser_bad_parcels", 0) != 0) | (qc.get("gordon_bad_parcels", 0) != 0)
    qc.loc[(qc.exclude_reason == "") & bad, "exclude_reason"] = "bad_parcels"
    ok_base = qc.exclude_reason == ""
    rate240 = float((ok_base & (qc.remain_sec_fd20 >= 240)).sum() / len(qc))
    min_sec = 240 if rate240 >= 0.60 else 180
    for t in FD_THRS:
        tag = f"fd{int(t * 100):02d}"
        qc[f"pass_{tag}"] = ok_base & (qc[f"remain_sec_{tag}"] >= min_sec)
    qc.loc[ok_base & ~qc.pass_fd20, "exclude_reason"] = f"remain_sec_fd20<{min_sec}"
    qc.to_csv(ROOT / f"data/interim/abide_rest_qc_{TODAY}.csv", index=False)

    out_dir = ROOT / "derivatives/fc"
    out_dir.mkdir(parents=True, exist_ok=True)
    strength_summary, precheck = {}, {}
    for (at, t), mats in fc.items():
        tag = f"{at}_fd{int(t * 100):02d}"
        X = np.vstack(mats)
        np.save(out_dir / f"{tag}_fcz_upper.npy", X)
        subs = pd.DataFrame(fc_subs[(at, t)], columns=["cohort", "sub_n"])
        subs.to_csv(out_dir / f"{tag}_subjects.csv", index=False)
        (out_dir / f"{tag}_parcels.txt").write_text("\n".join(parcels[at]), encoding="utf-8")
        n = len(parcels[at])
        iu = np.triu_indices(n, 1)
        S = np.zeros((len(X), n), dtype=np.float32)
        for k, v in enumerate(X):
            full = np.zeros((n, n), dtype=np.float32)
            full[iu] = v
            full = full + full.T
            S[k] = np.where(full > 0, full, 0).sum(1) / (n - 1)
        pd.concat([subs, pd.DataFrame(S, columns=parcels[at])], axis=1).to_csv(out_dir / f"{tag}_strength.csv", index=False)
        strength_summary[tag] = {"n_subjects": int(len(X)), "n_parcels": n}

        # 预检：TC 组（dx_group=2）平均 FC 在两个队列间的相关，只用通过质控者
        key = qc[[f"pass_fd{int(t * 100):02d}", "cohort", "sub_n", "dx_group"]]
        s2 = subs.merge(key, on=["cohort", "sub_n"], how="left")
        sel = s2[f"pass_fd{int(t * 100):02d}"].fillna(False).astype(bool) & (s2.dx_group == 2)
        m1 = X[(sel & (s2.cohort == "abide1")).values].mean(0)
        m2 = X[(sel & (s2.cohort == "abide2")).values].mean(0)
        r12 = float(np.corrcoef(m1, m2)[0, 1]) if np.isfinite(m1).all() and np.isfinite(m2).all() else None
        precheck[tag] = {"r_TC_meanFC_abide1_vs_abide2": None if r12 is None else round(r12, 4),
                         "n_TC_abide1": int((sel & (s2.cohort == "abide1")).sum()),
                         "n_TC_abide2": int((sel & (s2.cohort == "abide2")).sum())}

    p = qc[qc.pass_fd20]
    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/B02_qc_fc.py",
        "n_complete_dcan": int(len(qc)),
        "rate_remain240_fd20_all": round(rate240, 4),
        "min_sec_rule_result": min_sec,
        "n_exclude_by_reason": qc.exclude_reason.replace("", "none").str.split(":").str[0].value_counts().to_dict(),
        "pass_fd20_by_cohort_dx": {f"{c}_dx{d}": int(n) for (c, d), n in p.groupby(["cohort", "dx_group"]).size().items()},
        "pass_fd30_n": int(qc.pass_fd30.sum()),
        "tr_values": qc.tr.round(3).value_counts().to_dict() if "tr" in qc else {},
        "meanfd_retained_fd20_by_dx_passed": {str(d): [round(float(g.mean()), 4), round(float(g.std()), 4)]
                                               for d, g in p.groupby("dx_group").meanfd_retained_fd20},
        "remain_sec_fd20_by_dx_passed": {str(d): [round(float(g.mean()), 1), round(float(g.std()), 1)]
                                          for d, g in p.groupby("dx_group").remain_sec_fd20},
        "fc_outputs": strength_summary,
        "precheck_TC_meanFC_cross_cohort": precheck,
        "limit": a.limit,
    }
    (ROOT / "results/B02_qc_fc.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
