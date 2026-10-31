"""B02 的独立复核（Verifier）：随机抽被试（seed 42），用不同实现重算 FC，与 derivatives/fc 中保存的结果比较。

不同之处: 掩膜按 FD_threshold 数值精确匹配（B02 取最接近项）；剔帧向量另行用 remaining_frame_count 交叉核对；
FC 用 pandas DataFrame.corr 计算（B02 用 numpy.corrcoef）；上三角用 pandas 索引重建。
通过标准: 每人 Fisher z 最大绝对差 < 1e-4，且保留帧数与 QC 表一致。
用法: .venv_B/Scripts/python.exe -I scripts/B02v_verify_fc.py [--n 5]
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    qc = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_rest_qc_*.csv"))[-1])
    man = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_dcan_manifest_*.csv"))[-1])
    subs = pd.read_csv(ROOT / "derivatives/fc/glasser_fd20_subjects.csv")
    X = np.load(ROOT / "derivatives/fc/glasser_fd20_fcz_upper.npy", mmap_mode="r")
    pick = qc[qc.pass_fd20].sample(a.n, random_state=42)
    res = []
    for r in pick.itertuples():
        ses = man.loc[(man.cohort == r.cohort) & (man.sub_n == r.sub_n), "ses"].iloc[0]
        d = ROOT / "data/raw/abide_dcan" / r.cohort / f"sub-{r.sub_n}" / ses / "func"
        md = sio.loadmat(next(d.glob("*motion_mask.mat")), squeeze_me=True, struct_as_record=False)["motion_data"]
        e = [x for x in md if round(float(x.FD_threshold), 2) == 0.20]
        assert len(e) == 1, "FD 0.20 项不唯一"
        keep = np.asarray(e[0].frame_removal) == 0
        assert keep.sum() == int(e[0].remaining_frame_count)
        ts = nib.load(next(d.glob("*HCP2016FreeSurferSubcortical*ptseries.nii"))).get_fdata()
        df = pd.DataFrame(ts[keep])
        z = np.arctanh(df.corr().clip(-0.9999, 0.9999).to_numpy())
        n = z.shape[0]
        upper = np.concatenate([z[i, i + 1:] for i in range(n - 1)])
        k = subs.index[(subs.cohort == r.cohort) & (subs.sub_n == r.sub_n)][0]
        diff = float(np.abs(upper - X[k]).max())
        res.append({"cohort": r.cohort, "kept_frames": int(keep.sum()),
                    "qc_remain_frames": int(r.remain_frames_fd20), "max_abs_diff": diff,
                    "pass": diff < 1e-4 and int(keep.sum()) == int(r.remain_frames_fd20)})
    out = {"script": "scripts/B02v_verify_fc.py", "seed": 42, "n": a.n,
           "all_pass": all(x["pass"] for x in res), "detail": res}
    (ROOT / "results/B02v_verify_fc.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
