"""CoRR：由 fmriprep CIFTI dtseries 得到与 ABIDE 主分析一致的分区 FC 强度（全长、前半、后半）。

去噪（尽量对齐 DCAN desc-filtered）：回归 24 个头动参数（6 参数、一阶导数、平方、导数平方）+ 白质 + 脑脊液 + 全脑信号
  （fmriprep confounds；导数首行 NaN 置 0；去除 non_steady_state_outlier 帧）→ 0.008–0.09 Hz 二阶 Butterworth 零相位带通
  → FD > 0.2 mm 剔帧，并剔除保留帧中不足 5 帧的连续片段（Power 2014）。与 DCAN 的差异：DCAN 在回归时排除高运动帧，
  本脚本对全部帧回归后再剔帧（论文中注明为近似）。
分区: CoRR 的 fmriprep CIFTI 皮层为 fsaverage5 网格（每半球 10,242 顶点，去内侧壁），皮层下为 19 个体积结构。
  皮层用 HCP-MMP1.0 的 fsaverage 注释（Mills 2016，figshare doi 10.6084/m9.figshare.3498446.v2，MD5 已核对）取前 10,242 个顶点；
  皮层下按 CIFTI 结构名对应 ABIDE 的 19 个结构；按 ABIDE ptseries 的分区名与顺序重排（360 + 19 = 379）。
特征: 分区 FC 强度 = 与其余分区正值 Fisher z 的均值（同 B11）；保留帧按时间顺序等分前后两半。
质控: 剔帧后保留 ≥ 240 s（同 ABIDE）。
缺失: L_H_ROI、R_H_ROI 位于 fmriprep fsaverage5 CIFTI 的内侧壁掩膜内，无顶点；记为缺失，FC 强度在其余 377 个分区上计算。
输出: derivatives/corr/C02_strength_{full,half1,half2}.csv、C02_meta.csv（含被试编号，仅本地）
用法: .venv_B/Scripts/python.exe -I scripts/C02_corr_features.py [--check] [--no-gsr]
  --no-gsr: 不回归全脑信号（敏感性分析，2026-11-02），输出到 derivatives/corr/nogsr/
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fsio
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/corr/fmriprep"
OUT = ROOT / "derivatives/corr"
sys.stdout.reconfigure(encoding="utf-8")
ANNOT = ROOT / "data/raw/atlas/hcpmmp_fsaverage"


def load_annots():
    out = {}
    for h, struct in [("lh", "CIFTI_STRUCTURE_CORTEX_LEFT"), ("rh", "CIFTI_STRUCTURE_CORTEX_RIGHT")]:
        lab, _, nm = fsio.read_annot(str(ANNOT / f"{h}.HCP-MMP1.annot"))
        nm = [n.decode() for n in nm]
        out[struct] = np.array([nm[k] if k > 0 else "" for k in lab.astype(np.int64)[:10242]], dtype=object)
    return out


def grayordinate_names(ax, annots):
    """每个灰质坐标的 ABIDE 分区名（内侧壁等未标记为空串）。"""
    g = np.empty(ax.size, dtype=object)
    for name, slc, bm in ax.iter_structures():
        if name in annots:
            g[slc] = annots[name][bm.vertex]
        else:
            g[slc] = name.replace("CIFTI_STRUCTURE_", "")
    return g


def confound_matrix(cf, gsr=True):
    """24 参数头动模型（6 参数、一阶差分、二者的平方；此版 fmriprep 只提供 6 参数，差分与平方在此计算）+ WM + CSF + GS。"""
    M = cf[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]].fillna(0).to_numpy(float)
    D = np.vstack([np.zeros((1, 6)), np.diff(M, axis=0)])
    S = cf[["white_matter", "csf"] + (["global_signal"] if gsr else [])].fillna(0).to_numpy(float)
    return np.column_stack([np.ones(len(M)), M, D, M ** 2, D ** 2, S])


def censor_mask(fd, keep0, thr=0.2, min_run=5):
    keep = keep0 & ~(np.nan_to_num(fd, nan=0.0) > thr)
    out = keep.copy()
    i = 0
    while i < len(keep):
        if keep[i]:
            j = i
            while j < len(keep) and keep[j]:
                j += 1
            if j - i < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def strength(ts):
    fc = np.arctanh(np.clip(np.corrcoef(ts.T), -0.9999, 0.9999))
    np.fill_diagonal(fc, np.nan)
    return np.nanmean(np.where(fc > 0, fc, np.nan), axis=1)


def process(dts, cfp, names, annots, gsr=True):
    img = nib.load(dts)
    ax = img.header.get_axis(1)
    g = grayordinate_names(ax, annots)
    present = [n for n in names if np.any(g == n)]
    assert set(names) - set(present) <= {"L_H_ROI", "R_H_ROI"}, set(names) - set(present)  # 海马区位于 fsaverage5 内侧壁掩膜内
    tr = float(img.header.get_axis(0).step)
    Y = img.get_fdata(dtype=np.float32)
    P = np.vstack([Y[:, g == n].mean(1) for n in present]).T.astype(np.float64)  # T × 377
    cf = pd.read_csv(cfp, sep="\t")
    nss = cf.filter(like="non_steady_state_outlier").sum(1).to_numpy() > 0
    valid = ~nss
    X = confound_matrix(cf, gsr)
    B, *_ = np.linalg.lstsq(X[valid], P[valid], rcond=None)
    R = P - X @ B
    b, a = butter(2, [0.008, 0.09], btype="band", fs=1 / tr)
    R = filtfilt(b, a, R, axis=0)
    keep = censor_mask(cf["framewise_displacement"].to_numpy(float), valid)
    idx = np.where(keep)[0]
    h = len(idx) // 2
    fd = cf["framewise_displacement"].to_numpy(float)
    meta = {"tr": tr, "n_frames": len(P), "sec_full": len(idx) * tr, "sec_half": h * tr,
            "meanfd_retained": float(np.nanmean(fd[idx])) if len(idx) else np.nan}
    feats = {}
    if len(idx) * tr >= 240:
        for c, ix in {"full": idx, "half1": idx[:h], "half2": idx[h:2 * h]}.items():
            s = dict(zip(present, strength(R[ix])))
            feats[c] = [s.get(n, np.nan) for n in names]
    return meta, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只处理第一个会话并打印核对信息")
    ap.add_argument("--no-gsr", action="store_true", help="不回归全脑信号（敏感性分析）")
    a = ap.parse_args()
    out_dir = OUT / "nogsr" if a.no_gsr else OUT
    names = pd.read_csv(ROOT / "docs_B/glasser_parcel_table.csv").name.tolist()
    annots = load_annots()
    files = sorted(SRC.glob("sub-*/ses-*/func/*run-1_bold.dtseries.nii"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {c: [] for c in ["full", "half1", "half2"]}
    metas = []
    for i, f in enumerate(files, 1):
        cfp = Path(str(f).replace("_bold.dtseries.nii", "_desc-confounds_regressors.tsv"))
        if not cfp.exists():
            continue
        sub, ses = f.parts[-4], f.parts[-3]
        meta, feats = process(f, cfp, names, annots, gsr=not a.no_gsr)
        metas.append({"sub": sub, "ses": ses, **meta, "pass_qc": bool(feats)})
        for c, v in feats.items():
            rows[c].append([sub, ses, *v])
        if a.check:
            print(meta, "pass" if feats else "fail", "strength range", None if not feats else (float(np.nanmin(feats["full"])), float(np.nanmax(feats["full"]))), "n nan", None if not feats else int(np.isnan(feats["full"]).sum()))
            return
        if i % 25 == 0:
            print(i, "/", len(files), flush=True)
    for c, r in rows.items():
        pd.DataFrame(r, columns=["sub", "ses", *names]).to_csv(out_dir / f"C02_strength_{c}.csv", index=False)
    pd.DataFrame(metas).to_csv(out_dir / "C02_meta.csv", index=False)
    print("sessions", len(metas), "pass", sum(m["pass_qc"] for m in metas))


if __name__ == "__main__":
    main()
