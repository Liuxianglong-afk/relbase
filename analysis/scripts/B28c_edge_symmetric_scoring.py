"""课题 B1 事后分析（2026-11-01）：检验常模打分不对称是否造成连接水平 ASD > TC 的跨站点重叠。

背景: B28 中 TC 由五折交叉拟合模型打分（各折模型误差不同），ASD 全部由同一个全 TC 模型打分（共享同一份模型误差），
      这会在不同站点的 ASD 之间诱导正相关。B28b 固定每人极端数后，连接水平仍有很小的 ASD > TC（P = .0001）。
做法: ASD 按站点分层随机分到五折，用不含该折 TC 的模型打分（与 TC 对称）；其余同 B28（线性常模、站点居中）；
      再按 B28b 的每人固定极端数计算跨站点 Jaccard 与站点内置换 P。
输出: results/B28c_edge_symmetric_scoring.json
用法: .venv_B/Scripts/python.exe -I scripts/B28c_edge_symmetric_scoring.py
"""
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B28 = load("B28_edge_level")
B28b = load("B28b_edge_fixed_count")


def symmetric_z(Y, X, tc, site):
    n = len(Y)
    z = np.empty_like(Y)
    rng = np.random.default_rng(42)
    fold = np.empty(n, dtype=int)
    for s in np.unique(site):
        for grp in (True, False):
            ix = np.where((site == s) & (tc == grp))[0]
            fold[ix] = rng.permutation(np.arange(len(ix)) % 5)
    for f in range(5):
        train = tc & (fold != f)
        test = fold == f
        B, *_ = np.linalg.lstsq(X[train], Y[train], rcond=None)
        sd = (Y[train] - X[train] @ B).std(axis=0, ddof=X.shape[1])
        z[test] = (Y[test] - X[test] @ B) / sd
    return z


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B28c_edge_symmetric_scoring.py"}
    sl = pd.read_csv(ROOT / "derivatives/norms_B1/B15_subject_level.csv")
    E, pt, iu = B28.load_edges(sl)
    Y = E["full"]
    del E
    tc = (sl.dx == 2).to_numpy()
    site = sl.site.to_numpy()
    X = np.asarray(dmatrix("bs(age, df=4) + C(sex) + meanfd_retained_fd20", sl, return_type="dataframe"))
    z = B28.recentre(symmetric_z(Y, X, tc, site), tc, site).astype(np.float32)
    rate = float((np.abs(z[tc]) > 2).mean())
    k = int(round(rate * z.shape[1]))
    out["k"] = k
    out["sd_tc"], out["sd_asd"] = round(float(z[tc].std()), 4), round(float(z[~tc].std()), 4)
    for c in ["abide1", "abide2"]:
        m = (sl.cohort == c).to_numpy()
        out[c] = B28b.compare(z[m], np.where(tc[m], 2, 1), site[m], k)
        print(c, out[c], flush=True)
    (ROOT / "results/B28c_edge_symmetric_scoring.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
