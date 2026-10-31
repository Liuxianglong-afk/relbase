"""课题 B1 敏感性分析 1–3 的特征（protocol_B1 v1.0 第 7 节）：只算 FC 强度，数据长度 full / half1 / half2。

变体:
  fd30     FD 0.30 mm 剔帧掩膜；样本 = pass_fd30
  oddeven  FD 0.20；保留帧按奇偶交替分半（时间上不独立，作为信度上限参照）；样本 = pass_fd20
  gordon   FD 0.20；Gordon333 + 19 皮层下 ptseries；样本 = pass_fd20
强度定义与 B11 相同（每个分区与其余分区正值 Fisher z 的均值）。
输出: derivatives/features_B/<变体>/B11_strength_{full,half1,half2}.csv；results/B11s_<变体>.json
用法: .venv_B/Scripts/python.exe -I scripts/B11s_features_sensitivity.py <fd30|oddeven|gordon>
"""
import datetime as dt
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"fd30": ("pass_fd30", 0.30, "HCP2016", "time"), "oddeven": ("pass_fd20", 0.20, "HCP2016", "oddeven"),
            "gordon": ("pass_fd20", 0.20, "Gordon2014", "time")}


def strength(ts):
    z = np.arctanh(np.clip(np.corrcoef(ts.T), -0.9999, 0.9999))
    np.fill_diagonal(z, 0)
    return np.where(z > 0, z, 0).sum(1) / (z.shape[0] - 1)


def main():
    v = sys.argv[1]
    pass_col, fd, atlas, split = VARIANTS[v]
    qc = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_rest_qc_*.csv"))[-1])
    man = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_dcan_manifest_*.csv"))[-1])[["cohort", "sub_n", "ses"]]
    s = qc[qc[pass_col]].merge(man, on=["cohort", "sub_n"]).reset_index(drop=True)
    rows = {c: [] for c in ["full", "half1", "half2"]}
    names = None
    secs = []
    for i, r in enumerate(s.itertuples(), 1):
        d = ROOT / "data/raw/abide_dcan" / r.cohort / f"sub-{r.sub_n}" / r.ses / "func"
        img = nib.load(next(d.glob(f"*atlas-{atlas}FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii")))
        nm = [str(x) for x in img.header.get_axis(1).name]
        if names is None:
            names = nm
        assert nm == names
        ts = img.get_fdata(dtype=np.float64)
        md = sio.loadmat(next(d.glob("*motion_mask.mat")), squeeze_me=True, struct_as_record=False)["motion_data"]
        e = [x for x in md if round(float(x.FD_threshold), 2) == fd][0]
        idx = np.where(np.asarray(e.frame_removal) == 0)[0]
        h = len(idx) // 2
        if split == "time":
            segs = {"full": idx, "half1": idx[:h], "half2": idx[h:2 * h]}
        else:
            segs = {"full": idx, "half1": idx[0::2][:h], "half2": idx[1::2][:h]}
        for c, ix in segs.items():
            rows[c].append([r.cohort, r.sub_n, *strength(ts[ix])])
        secs.append(h * float(e.epi_TR))
        if i % 200 == 0:
            print(f"  {i}/{len(s)}", flush=True)
    out = ROOT / "derivatives/features_B" / v
    out.mkdir(parents=True, exist_ok=True)
    for c, rr in rows.items():
        pd.DataFrame(rr, columns=["cohort", "sub_n", *names]).to_csv(out / f"B11_strength_{c}.csv", index=False)
    summ = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B11s_features_sensitivity.py", "variant": v,
            "pass_column": pass_col, "fd": fd, "atlas": atlas, "split": split, "n_subjects": int(len(s)), "n_parcels": len(names),
            "sec_half_median_iqr": [round(float(x), 1) for x in np.percentile(secs, [50, 25, 75])]}
    (ROOT / f"results/B11s_{v}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
