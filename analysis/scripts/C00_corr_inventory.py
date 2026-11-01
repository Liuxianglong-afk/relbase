"""CoRR 重测数据可用性清点（FCP-INDI 公开 S3，fmriprep 输出）。

列出 data/Projects/CORR/Outputs/fmriprep/fmriprep/ 下全部键，统计每名被试各会话、各 run 的
CIFTI dtseries（bold.dtseries.nii）与 confounds（desc-confounds_regressors.tsv）是否齐全及文件大小。
输出: data/interim/corr_fmriprep_inventory_<日期>.csv（含被试编号，仅本地）；results/C00_corr_inventory.json（汇总，不含编号）
用法: .venv_B/Scripts/python.exe -I scripts/C00_corr_inventory.py
"""
import datetime as dt
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://fcp-indi.s3.us-east-1.amazonaws.com"
PREFIX = "data/Projects/CORR/Outputs/fmriprep/fmriprep/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
KEY_RE = re.compile(r"(sub-[^/]+)/(ses-[^/]+)/func/.*_run-(\d+)_(bold\.dtseries\.nii|desc-confounds_regressors\.tsv)$")
sys.stdout.reconfigure(encoding="utf-8")


def list_keys(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=120) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            out.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def main():
    keys = list_keys(PREFIX)
    rows = []
    for k, size in keys:
        m = KEY_RE.search(k[len(PREFIX):])
        if m:
            rows.append({"sub": m.group(1), "ses": m.group(2), "run": int(m.group(3)), "kind": "dtseries" if "dtseries" in m.group(4) else "confounds",
                         "key": k, "bytes": size})
    df = pd.DataFrame(rows)
    stamp = dt.date.today().strftime("%Y%m%d")
    df.to_csv(ROOT / f"data/interim/corr_fmriprep_inventory_{stamp}.csv", index=False)
    piv = df.pivot_table(index=["sub", "ses", "run"], columns="kind", values="bytes", aggfunc="first")
    ok = piv.dropna()
    run1 = ok.reset_index().sort_values("run").groupby(["sub", "ses"]).first().reset_index()
    ses_per_sub = run1.groupby("sub").ses.nunique()
    multi = ses_per_sub[ses_per_sub >= 2].index
    sel = run1[run1["sub"].isin(multi)].sort_values(["sub", "ses"]).groupby("sub").head(2)
    summary = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "n_keys_total": len(keys),
               "n_dtseries": int((df.kind == "dtseries").sum()), "n_subjects_any": int(df["sub"].nunique()),
               "n_subject_sessions_complete": int(len(run1)), "n_subjects_ge2_sessions": int(len(multi)),
               "sessions_per_subject": {str(k): int(v) for k, v in ses_per_sub.value_counts().sort_index().items()},
               "dtseries_bytes_median": float(ok.dtseries.median()),
               "download_gb_first_run_two_sessions": round(float(sel[["dtseries", "confounds"]].sum().sum()) / 1e9, 1)}
    (ROOT / "results/C00_corr_inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
