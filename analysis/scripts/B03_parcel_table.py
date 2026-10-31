"""Glasser（HCP-MMP1.0）分区表：按 DCAN ptseries 的分区顺序，对应 hcp_utils 的 MMP 编号与 Cole-Anticevic 网络。

DCAN ptseries 顺序为 R 半球 180 个 → L 半球 180 个 → 19 个皮层下结构；hcp_utils.mmp 为 L → R，因此按名称对应
（ptseries 名称去掉 "_ROI" 后缀 == mmp 标签）。
网络归属：每个 MMP 分区内 ca_network 灰质坐标标签的众数；同时记录纯度（众数占比）。皮层下结构的网络记为 subcortical。
厚度映射：hcp_utils.mmp.map_all 的前 59,412 个灰质坐标即 DCAN fsLR32k 厚度 dscalar 的皮层顶点（左 29,696 + 右 29,716）。

输出: docs_B/glasser_parcel_table.csv、docs_B/glasser_cortex_vertex_labels.npy（59,412 个顶点的 MMP 编号）
用法: .venv_B/Scripts/python.exe -I scripts/B03_parcel_table.py
"""
from pathlib import Path

import hcp_utils as hcp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_CTX = 59412


def main():
    names = (ROOT / "derivatives/fc/glasser_fd20_parcels.txt").read_text(encoding="utf-8").split("\n")
    mmp_id = {str(v): int(k) for k, v in hcp.mmp.labels.items() if v}
    vert_mmp = np.asarray(hcp.mmp.map_all[:N_CTX])
    vert_ca = np.asarray(hcp.ca_network.map_all[:N_CTX])
    ca_lab = {int(k): str(v) for k, v in hcp.ca_network.labels.items()}

    rows = []
    for i, nm in enumerate(names):
        row = {"idx": i, "name": nm}
        if nm.endswith("_ROI"):
            key = nm[:-4]
            mid = mmp_id[key]
            ca = vert_ca[vert_mmp == mid]
            vals, cnt = np.unique(ca[ca > 0], return_counts=True)
            row.update(hemi=key[0], mmp_id=mid, n_vertices=int((vert_mmp == mid).sum()),
                       network=ca_lab[int(vals[cnt.argmax()])], network_purity=round(float(cnt.max() / cnt.sum()), 3))
        else:
            row.update(hemi="L" if "LEFT" in nm else ("R" if "RIGHT" in nm else "M"), mmp_id=-1, n_vertices=0,
                       network="subcortical", network_purity=1.0)
        rows.append(row)
    t = pd.DataFrame(rows)
    out = ROOT / "docs_B"
    out.mkdir(exist_ok=True)
    t.to_csv(out / "glasser_parcel_table.csv", index=False)
    np.save(out / "glasser_cortex_vertex_labels.npy", vert_mmp.astype(np.int16))

    ctx = t[t.mmp_id > 0]
    assert len(ctx) == 360 and ctx.mmp_id.nunique() == 360, "皮层分区未一一对应"
    assert (ctx.n_vertices > 0).all(), "存在没有顶点的分区"
    print("cortical parcels matched:", len(ctx), "| subcortical:", int((t.mmp_id < 0).sum()))
    print("network counts:", t.network.value_counts().to_dict())
    print("purity <0.8:", int((ctx.network_purity < 0.8).sum()), "min", ctx.network_purity.min())


if __name__ == "__main__":
    main()
