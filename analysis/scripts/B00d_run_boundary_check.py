"""核实 DCAN desc-filtered ptseries 是否为各 run 按顺序拼接：用 HTTP Range 读取每个 run 级 dtseries 的 NIfTI-2 头（前 540 字节，
dim[] 为 offset 16 起的 8 个 int64；CIFTI 时间维在 dim[5]），比较 run 帧数之和与 motion_mask 的 total_frame_count。
用法: python -I scripts/B00d_run_boundary_check.py <qc_csv> <manifest_csv> <输出json> [每个队列抽查人数]
"""
import json, re, struct, sys, urllib.parse, urllib.request, xml.etree.ElementTree as ET
import pandas as pd
B = "https://fcp-indi.s3.us-east-1.amazonaws.com"; NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PRE = {"abide1": "data/Projects/ABIDE/Derivatives/DCAN/", "abide2": "data/Projects/ABIDE2/Derivatives/DCAN/"}

def ls(prefix):
    r = ET.fromstring(urllib.request.urlopen(f"{B}/?{urllib.parse.urlencode({'list-type': '2', 'prefix': prefix})}", timeout=60).read())
    return [c.find("s:Key", NS).text for c in r.findall("s:Contents", NS)]

def nframes(key):
    req = urllib.request.Request(f"{B}/{urllib.parse.quote(key)}", headers={"Range": "bytes=0-539"})
    h = urllib.request.urlopen(req, timeout=60).read()
    assert struct.unpack("<i", h[:4])[0] == 540, "不是 NIfTI-2 头"
    dims = struct.unpack("<8q", h[16:80])
    return int(dims[5]), dims

qc = pd.read_csv(sys.argv[1]); man = pd.read_csv(sys.argv[2]); k = int(sys.argv[4]) if len(sys.argv) > 4 else 6
res = []
for coh, pre in PRE.items():
    cand = qc[(qc.cohort == coh) & qc.site.isin(["OHSU", "ABIDEII-OHSU_1", "ABIDEII-IP_1", "ABIDEII-ONRC_2"])].merge(man[["cohort", "sub_n", "ses"]], on=["cohort", "sub_n"])
    for r in cand.head(k).itertuples():
        sid = f"sub-{r.sub_n:07d}" if coh == "abide1" else f"sub-{r.sub_n}"
        keys = ls(f"{pre}{sid}/{r.ses}/func/")
        runs = sorted([x for x in keys if re.search(r"_run-\d+_bold_timeseries\.dtseries\.nii$", x)])
        fr = [nframes(x)[0] for x in runs]
        res.append({"cohort": coh, "site": r.site, "n_runs": len(runs), "run_frames": fr, "sum": sum(fr),
                    "total_frame_count_mask": int(r.total_frames), "match": sum(fr) == int(r.total_frames)})
        print(res[-1], flush=True)
json.dump(res, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
