"""课题 B1 特征（protocol_B1 v0.2 第 3、6 节）：各数据长度的 FC 强度、CAB-NP 网络 FC，以及 Glasser 分区皮层厚度。

样本: 共用质控通过者（abide_rest_qc_*.csv 中 pass_fd20），FD 0.20 剔帧掩膜。
数据长度:
  full   全部保留帧
  half1  保留帧按时间顺序的前一半；half2 后一半（两半帧数相同，奇数时舍去最后一帧）
  run1/run2  多 run 站点（OHSU、ABIDEII-OHSU_1、ABIDEII-IP_1、ABIDEII-ONRC_2）：拼接的 desc-filtered ptseries 按各 run 帧数
             （HTTP Range 读取 run 级 dtseries 的 NIfTI-2 头）切开；取前两个保留帧 ≥ 60 的 run。帧数之和与 total_frame_count
             不一致的被试不做跨 run。
特征:
  strength  每个分区与其余分区正值 Fisher z 的均值（379 项，列名为 ptseries 分区名）
  netfc     CAB-NP 12 网络 + subcortical 共 13 组，组内（上三角）与组间 Fisher z 均值（91 项）
  thickness DCAN fsLR32k 厚度按 MMP 分区的顶点均值（360 项）
输出: derivatives/features_B/B11_{strength,netfc}_{cond}.csv、B11_thickness.csv、B11_meta.csv；results/B11_features.json
用法: .venv_B/Scripts/python.exe -I scripts/B11_features_reliability.py [--limit N]
"""
import argparse
import datetime as dt
import json
import re
import struct
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from itertools import combinations_with_replacement
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://fcp-indi.s3.us-east-1.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
PREFIX = {"abide1": "data/Projects/ABIDE/Derivatives/DCAN/", "abide2": "data/Projects/ABIDE2/Derivatives/DCAN/"}
MULTIRUN_SITES = {"OHSU", "ABIDEII-OHSU_1", "ABIDEII-IP_1", "ABIDEII-ONRC_2"}
MIN_RUN_FRAMES = 60
OUT = ROOT / "derivatives/features_B"


def latest(pattern):
    return sorted((ROOT / "data/interim").glob(pattern))[-1]


def http_get(req, retries=5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def run_frames(cohort, sub_n, ses):
    sid = f"sub-{sub_n:07d}" if cohort == "abide1" else f"sub-{sub_n}"
    q = urllib.parse.urlencode({"list-type": "2", "prefix": f"{PREFIX[cohort]}{sid}/{ses}/func/"})
    root = ET.fromstring(http_get(f"{BUCKET}/?{q}"))
    keys = sorted(k for k in (c.find("s3:Key", NS).text for c in root.findall("s3:Contents", NS))
                  if re.search(r"_run-\d+_bold_timeseries\.dtseries\.nii$", k))
    keys.sort(key=lambda k: int(re.search(r"_run-(\d+)_", k).group(1)))
    out = []
    for k in keys:
        req = urllib.request.Request(f"{BUCKET}/{urllib.parse.quote(k)}", headers={"Range": "bytes=0-539"})
        h = http_get(req)
        out.append(int(struct.unpack("<8q", h[16:80])[5]))
    return out


def fcz(ts):
    return np.arctanh(np.clip(np.corrcoef(ts.T), -0.9999, 0.9999))


def features(ts, groups, pairs):
    z = fcz(ts)
    n = z.shape[0]
    zz = z.copy()
    np.fill_diagonal(zz, 0)
    strength = np.where(zz > 0, zz, 0).sum(1) / (n - 1)
    net = []
    for a, b in pairs:
        ia, ib = groups[a], groups[b]
        if a == b:
            sub = z[np.ix_(ia, ia)]
            net.append(sub[np.triu_indices(len(ia), 1)].mean())
        else:
            net.append(z[np.ix_(ia, ib)].mean())
    return strength, np.array(net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    qc = pd.read_csv(latest("abide_rest_qc_*.csv"))
    man = pd.read_csv(latest("abide_dcan_manifest_*.csv"))[["cohort", "sub_n", "ses"]]
    s = qc[qc.pass_fd20].merge(man, on=["cohort", "sub_n"]).reset_index(drop=True)
    if a.limit:
        s = s.head(a.limit)
    pt = pd.read_csv(ROOT / "docs_B/glasser_parcel_table.csv")
    names = pt.name.tolist()
    nets = sorted(pt.network.unique())
    groups = {g: np.where(pt.network.values == g)[0] for g in nets}
    pairs = list(combinations_with_replacement(nets, 2))
    pair_names = [f"{x}__{y}" for x, y in pairs]
    vlab = np.load(ROOT / "docs_B/glasser_cortex_vertex_labels.npy")
    ctx = pt[pt.mmp_id > 0]

    rows = {c: {"strength": [], "netfc": []} for c in ["full", "half1", "half2", "run1", "run2"]}
    thick, meta = [], []
    for i, r in enumerate(s.itertuples(), 1):
        d = ROOT / "data/raw/abide_dcan" / r.cohort / f"sub-{r.sub_n}" / r.ses
        img = nib.load(next((d / "func").glob("*HCP2016FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii")))
        assert list(img.header.get_axis(1).name) == names
        ts = img.get_fdata(dtype=np.float64)
        md = sio.loadmat(next((d / "func").glob("*motion_mask.mat")), squeeze_me=True, struct_as_record=False)["motion_data"]
        e = [x for x in md if round(float(x.FD_threshold), 2) == 0.20][0]
        keep = np.asarray(e.frame_removal) == 0
        tr = float(e.epi_TR)
        idx = np.where(keep)[0]
        h = len(idx) // 2
        segs = {"full": idx, "half1": idx[:h], "half2": idx[h:2 * h]}
        m = {"cohort": r.cohort, "sub_n": r.sub_n, "site": r.site, "tr": tr, "sec_full": len(idx) * tr, "sec_half": h * tr,
             "n_runs": np.nan, "run_check": "", "sec_run1": np.nan, "sec_run2": np.nan}
        if r.site in MULTIRUN_SITES:
            try:
                fr = run_frames(r.cohort, r.sub_n, r.ses)
                m["n_runs"] = len(fr)
                if sum(fr) != ts.shape[0]:
                    m["run_check"] = f"sum{sum(fr)}!=total{ts.shape[0]}"
                else:
                    bounds = np.cumsum([0] + fr)
                    ok_runs = [np.where(keep[bounds[j]:bounds[j + 1]])[0] + bounds[j] for j in range(len(fr))]
                    ok_runs = [x for x in ok_runs if len(x) >= MIN_RUN_FRAMES][:2]
                    m["run_check"] = f"qualifying_runs={len(ok_runs)}"
                    if len(ok_runs) == 2:
                        # 两个 run 截到相同保留帧数（各取前 n 帧），避免长度差异混入信度
                        n = min(len(ok_runs[0]), len(ok_runs[1]))
                        ok_runs = [ok_runs[0][:n], ok_runs[1][:n]]
                        segs["run1"], segs["run2"] = ok_runs
                        m["sec_run1"], m["sec_run2"] = len(ok_runs[0]) * tr, len(ok_runs[1]) * tr
            except Exception as ex:
                m["run_check"] = f"error: {ex}"
        for c, ix in segs.items():
            st, nf = features(ts[ix], groups, pairs)
            rows[c]["strength"].append([r.cohort, r.sub_n, *st])
            rows[c]["netfc"].append([r.cohort, r.sub_n, *nf])
        th = nib.load(next((d / "anat").glob("*thickness.dscalar.nii"))).get_fdata().ravel()
        assert th.size == vlab.size
        thick.append([r.cohort, r.sub_n, *[th[(vlab == mid) & (th > 0)].mean() for mid in ctx.mmp_id]])
        meta.append(m)
        if i % 100 == 0:
            print(f"  {i}/{len(s)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B11_features_reliability.py",
               "n_subjects": int(len(s)), "n_by_condition": {}}
    for c, dd in rows.items():
        pd.DataFrame(dd["strength"], columns=["cohort", "sub_n", *names]).to_csv(OUT / f"B11_strength_{c}.csv", index=False)
        pd.DataFrame(dd["netfc"], columns=["cohort", "sub_n", *pair_names]).to_csv(OUT / f"B11_netfc_{c}.csv", index=False)
        summary["n_by_condition"][c] = len(dd["strength"])
    pd.DataFrame(thick, columns=["cohort", "sub_n", *ctx.name]).to_csv(OUT / "B11_thickness.csv", index=False)
    meta = pd.DataFrame(meta)
    meta.to_csv(OUT / "B11_meta.csv", index=False)
    summary["sec_half_median_iqr"] = [float(x) for x in meta.sec_half.quantile([0.5, 0.25, 0.75]).round(1)]
    summary["run_check_counts"] = meta.run_check.replace("", "single_run_site").str.split(":").str[0].value_counts().to_dict()
    mr = meta.dropna(subset=["sec_run1"])
    summary["run_pairs_by_site"] = mr.site.value_counts().to_dict()
    summary["sec_run_median"] = [float(mr.sec_run1.median()) if len(mr) else None, float(mr.sec_run2.median()) if len(mr) else None]
    summary["thickness_nan_cells"] = int(pd.DataFrame(thick).iloc[:, 2:].isna().sum().sum())
    (ROOT / "results/B11_features.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
