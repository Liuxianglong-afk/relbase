"""下载 CoRR fmriprep 输出（FCP-INDI 公开 S3）：有年龄与性别的被试，每个会话的 run-1 CIFTI dtseries 与 confounds。

被试: data/interim/corr_fmriprep_inventory_*.csv 中 dtseries 与 confounds 齐全、且在 CoRR_AggregatedPhenotypicData 中有
      数值年龄与性别（1/2）者（2026-11-01：229 人，其中 104 人 ≥ 2 个会话；约 27 GB）。
断点续传: 本地文件存在且大小与 S3 一致则跳过；先写 .part 再改名。
输出: data/raw/corr/fmriprep/<sub>/<ses>/func/...（含被试编号，仅本地）；logs 由调用方重定向
用法: .venv_B/Scripts/python.exe -I scripts/C01_corr_download.py
"""
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://fcp-indi.s3.us-east-1.amazonaws.com"
PREFIX = "data/Projects/CORR/Outputs/fmriprep/fmriprep/"
DEST = ROOT / "data/raw/corr/fmriprep"
sys.stdout.reconfigure(encoding="utf-8")


def selection():
    inv = pd.read_csv(sorted((ROOT / "data/interim").glob("corr_fmriprep_inventory_*.csv"))[-1])
    ok = inv.pivot_table(index=["sub", "ses", "run"], columns="kind", values="bytes", aggfunc="first").dropna().reset_index()
    r1 = ok.sort_values("run").groupby(["sub", "ses"]).first().reset_index()[["sub", "ses", "run"]]
    r1["SUBID"] = r1["sub"].str[4:].astype(int)
    ph = pd.read_csv(ROOT / "data/raw/corr/corr_pheno_clean.csv")
    ph["age"] = pd.to_numeric(ph.AGE_AT_SCAN_1, errors="coerce")
    good = set(ph[ph.age.notna() & ph.SEX.astype(str).isin(["1", "2"])].SUBID)
    r1 = r1[r1.SUBID.isin(good)]
    return inv.merge(r1[["sub", "ses", "run"]], on=["sub", "ses", "run"])


def fetch(key, size, path, retries=5):
    if path.exists() and path.stat().st_size == size:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for k in range(retries):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=300) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if part.stat().st_size == size:
                part.replace(path)
                return "ok"
        except Exception as e:  # 网络错误重试
            print("retry", k + 1, type(e).__name__, flush=True)
            time.sleep(5 * (k + 1))
    return "fail"


def main():
    sel = selection()
    total = sel.bytes.sum() / 1e9
    print(f"files {len(sel)}, {total:.1f} GB", flush=True)
    done, stats = 0.0, {"ok": 0, "skip": 0, "fail": 0}
    for i, r in enumerate(sel.itertuples(), 1):
        st = fetch(r.key, int(r.bytes), DEST / r.key[len(PREFIX):])
        stats[st] += 1
        done += r.bytes / 1e9
        if i % 20 == 0 or st == "fail":
            print(f"{i}/{len(sel)} {done:.1f}/{total:.1f} GB {stats}", flush=True)
    print("finished", stats, flush=True)


if __name__ == "__main__":
    main()
