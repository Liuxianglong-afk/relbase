"""下载 ABIDE I/II 的 DCAN（ABCD-HCP 管线）衍生数据：分区时间序列、剔帧掩膜、皮层厚度。可断点续传。

纳入（与课题 A 相同，protocol_B 第 3 节）: 5–21 岁、dx_group ∈ {1,2}；排除 ABIDEII-UCLA_Long/UPSM_Long。
每人只取一个 session（按名称排序的第一个；纵向站点已排除，正常情况下每人只有一个）。
每人下载 4 个文件:
  func/*atlas-HCP2016FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii  主分析（Glasser 360 + 19 皮层下）
  func/*atlas-Gordon2014FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii  图谱敏感性
  func/*desc-filtered_motion_mask.mat  FD 0–0.5 mm 各阈值的剔帧向量
  anat/*space-fsLR32k_thickness.dscalar.nii  皮层厚度（课题 B1 结构偏离）
完整性: 文件大小与 S3 清单一致。
输出:
  data/raw/abide_dcan/<cohort>/sub-<id>/<ses>/{func,anat}/<原文件名>
  data/interim/abide_dcan_manifest_<YYYYMMDD>.csv（每人一行，含 4 个文件是否齐全）
  data/interim/abide_dcan_download_log_<YYYYMMDD>.csv
  results/abide_dcan_download.json

用法: python -I scripts/B01_download_dcan.py [--limit N] [--workers 8]
"""
import argparse
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://fcp-indi.s3.us-east-1.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
PREFIXES = {"abide1": "data/Projects/ABIDE/Derivatives/DCAN/", "abide2": "data/Projects/ABIDE2/Derivatives/DCAN/"}
EXCLUDE_SITES = {"ABIDEII-UCLA_Long", "ABIDEII-UPSM_Long"}
FILE_TYPES = {
    "glasser": re.compile(r"func/[^/]*atlas-HCP2016FreeSurferSubcortical_desc-filtered_timeseries\.ptseries\.nii$"),
    "gordon": re.compile(r"func/[^/]*atlas-Gordon2014FreeSurferSubcortical_desc-filtered_timeseries\.ptseries\.nii$"),
    "motion": re.compile(r"func/[^/]*desc-filtered_motion_mask\.mat$"),
    "thickness": re.compile(r"anat/[^/]*space-fsLR32k_thickness\.dscalar\.nii$"),
}
KEY_RE = re.compile(r"^sub-(?P<sub>\d+)/(?P<ses>ses-[^/]+)/(?P<rest>.+)$")
TODAY = dt.date.today().strftime("%Y%m%d")


def http_get(url, retries=5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  retry {i + 1}: {e}")
            time.sleep(2 ** i)


def list_keys(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(http_get(f"{BUCKET}/?{urllib.parse.urlencode(q)}"))
        for c in root.findall("s3:Contents", NS):
            out.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def inventory():
    rows = []
    for cohort, prefix in PREFIXES.items():
        keys = list_keys(prefix)
        print(f"{cohort}: {len(keys)} keys")
        for key, size in keys:
            m = KEY_RE.match(key[len(prefix):])
            if not m:
                continue
            for ftype, pat in FILE_TYPES.items():
                if pat.search(m["rest"]):
                    rows.append({"cohort": cohort, "sub_n": int(m["sub"]), "ses": m["ses"], "ftype": ftype,
                                 "key": key, "bytes": size})
    return pd.DataFrame(rows)


def eligible():
    p = pd.read_csv(ROOT / "data/interim/abide_pheno_harmonized_20261014.csv", dtype=str)
    p = p[(p.in_age_5_21 == "True") & p.dx_group.isin(["1", "2"]) & ~p.site.isin(EXCLUDE_SITES)].copy()
    p["sub_n"] = p["sub"].astype(int)
    return p[["cohort", "site", "sub_n", "dx_group", "age", "sex"]].drop_duplicates(["cohort", "sub_n"])


def dest_path(r):
    return ROOT / "data/raw/abide_dcan" / r.cohort / f"sub-{r.sub_n}" / r.ses / r.key.split(f"/{r.ses}/", 1)[1]


def fetch(r):
    path = dest_path(r)
    if path.exists() and path.stat().st_size == r.bytes:
        return r.key, "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_bytes(http_get(f"{BUCKET}/{urllib.parse.quote(r.key)}"))
        if tmp.stat().st_size != r.bytes:
            return r.key, "size_mismatch"
        tmp.replace(path)
        return r.key, "downloaded"
    except Exception as e:
        return r.key, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    inv = inventory()
    first_ses = inv.groupby(["cohort", "sub_n"]).ses.min().rename("first_ses").reset_index()
    inv = inv.merge(first_ses, on=["cohort", "sub_n"])
    n_multi_ses = int((inv.groupby(["cohort", "sub_n"]).ses.nunique() > 1).sum())
    inv = inv[inv.ses == inv.first_ses].drop(columns="first_ses")

    el = eligible()
    targets = inv.merge(el[["cohort", "sub_n"]], on=["cohort", "sub_n"])
    subs = el[["cohort", "sub_n"]].drop_duplicates()
    if a.limit:
        subs = subs.head(a.limit)
        targets = targets.merge(subs, on=["cohort", "sub_n"])
    print(f"eligible {len(el)}; target files {len(targets)}")

    log = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, r) for r in targets.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            key, status = f.result()
            log.append({"key": key, "status": status})
            if i % 200 == 0:
                print(f"  {i}/{len(futs)}", flush=True)
    log = pd.DataFrame(log)
    log.to_csv(ROOT / f"data/interim/abide_dcan_download_log_{TODAY}.csv", index=False)

    ok_keys = set(log.loc[log.status.isin(["downloaded", "exists"]), "key"])
    targets["ok"] = targets.key.isin(ok_keys)
    wide = targets.pivot_table(index=["cohort", "sub_n", "ses"], columns="ftype", values="ok", aggfunc="any").reset_index()
    man = el.merge(wide, on=["cohort", "sub_n"], how="left")
    for ft in FILE_TYPES:
        if ft not in man:
            man[ft] = False
        man[ft] = man[ft].astype("boolean").fillna(False).astype(bool)
    man["complete"] = man[list(FILE_TYPES)].all(axis=1)
    man.to_csv(ROOT / f"data/interim/abide_dcan_manifest_{TODAY}.csv", index=False)

    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/B01_download_dcan.py",
        "n_eligible": int(len(man)),
        "n_subjects_with_multiple_ses_in_dcan": n_multi_ses,
        "status_counts": log.status.str.split(":").str[0].value_counts().to_dict(),
        "GB_downloaded_or_present": round(targets.loc[targets.ok, "bytes"].sum() / 1e9, 2),
        "files_present_by_type": {ft: int(man[ft].sum()) for ft in FILE_TYPES},
        "complete_by_cohort_dx": {f"{c}_dx{d}": int(n) for (c, d), n in
                                  man.groupby(["cohort", "dx_group"]).complete.sum().items()},
        "n_complete": int(man.complete.sum()),
        "limit": a.limit,
    }
    (ROOT / "results/abide_dcan_download.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
