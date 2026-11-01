"""课题 B1 事后分析（2026-11-01，登记于方案第 12 节）：连接（edge）水平的个体偏离信度与重叠。

动机: 主分析用分区 FC 强度；Ilioska 2026 等异质性研究用连接水平偏离。审稿人可能要求在同一水平比较。
数据: 主样本 833 人（site_ok 站点），沿用 B11 的剔帧（FD 0.2 mm）与前后分半；379 分区的全部 71,631 条连接（Fisher z）。
常模: 连接数过多，不能逐条拟合 GAMLSS；改用线性常模（年龄 B 样条 df = 4 + 性别 + 保留帧平均 FD，同方差），
      全长、前半、后半分别建模；TC 用按站点分层的五折交叉拟合（种子 42），ASD 用全部 TC 拟合的模型；
      站点居中与主分析相同（TC 用同站点留一 TC 的均值与 SD，ASD 用同站点全部 TC）。
指标（各队列 × 组）:
  a. 连接分半 ICC(2,1) 中位数（bootstrap 200 次 × 种子 42/7/123/2024，跨种子平均界）；ICC ≥ 0.5 的比例
  b. P2 极端（|z| > 2）正向一致率，及站点内打乱两半对应关系的机会水平（200 次）
  c. 同一连接同向极端者的最大比例（全长；Ilioska 的报告方式），及其前后两半各自计算时的值
  d. 区域水平与网络水平汇总：每人每分区（379）/每网络（13）的极端连接数，分半 ICC 中位数
  e. 跨站点两两 Jaccard（全长极端连接集合，含符号），ASD − TC 差异的站点内置换 P（10,000 次）
输出: results/B28_edge_level.json；derivatives/norms_B1/edges/（连接 z，仅本地）
用法: .venv_B/Scripts/python.exe -I scripts/B28_edge_level.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
from patsy import dmatrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "relbase"))
from relbase import icc21  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SEEDS = [42, 7, 123, 2024]
OUT = ROOT / "derivatives/norms_B1/edges"
CONDS = ["full", "half1", "half2"]


def latest(pattern):
    return sorted((ROOT / "data/interim").glob(pattern))[-1]


def load_edges(sl):
    man = pd.read_csv(latest("abide_dcan_manifest_*.csv"))[["cohort", "sub_n", "ses"]]
    s = sl.merge(man, on=["cohort", "sub_n"], how="left")
    pt = pd.read_csv(ROOT / "docs_B/glasser_parcel_table.csv")
    names = pt.name.tolist()
    P = len(names)
    iu = np.triu_indices(P, 1)
    E = {c: np.empty((len(s), len(iu[0])), dtype=np.float32) for c in CONDS}
    for i, r in enumerate(s.itertuples()):
        d = ROOT / "data/raw/abide_dcan" / r.cohort / f"sub-{r.sub_n}" / r.ses / "func"
        img = nib.load(next(d.glob("*HCP2016FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii")))
        assert list(img.header.get_axis(1).name) == names
        ts = img.get_fdata(dtype=np.float64)
        md = sio.loadmat(next(d.glob("*motion_mask.mat")), squeeze_me=True, struct_as_record=False)["motion_data"]
        e = [x for x in md if round(float(x.FD_threshold), 2) == 0.20][0]
        idx = np.where(np.asarray(e.frame_removal) == 0)[0]
        h = len(idx) // 2
        for c, ix in {"full": idx, "half1": idx[:h], "half2": idx[h:2 * h]}.items():
            fc = np.arctanh(np.clip(np.corrcoef(ts[ix].T), -0.9999, 0.9999))
            E[c][i] = fc[iu]
        if (i + 1) % 100 == 0:
            print("edges", i + 1, flush=True)
    return E, pt, iu


def normative_z(Y, X, tc, site):
    """线性常模：TC 五折交叉拟合（按站点分层），ASD 用全部 TC；返回未居中的 z。"""
    n = len(Y)
    z = np.empty_like(Y)
    rng = np.random.default_rng(42)
    tci = np.where(tc)[0]
    fold = np.empty(n, dtype=int)
    for s in np.unique(site[tc]):
        ix = tci[site[tc] == s]
        fold[ix] = rng.permutation(np.arange(len(ix)) % 5)

    def fit_apply(train, test):
        B, *_ = np.linalg.lstsq(X[train], Y[train], rcond=None)
        res = Y[train] - X[train] @ B
        sd = res.std(axis=0, ddof=X.shape[1])
        z[test] = (Y[test] - X[test] @ B) / sd
    for f in range(5):
        fit_apply(tci[fold[tci] != f], tci[fold[tci] == f])
    fit_apply(tci, np.where(~tc)[0])
    return z


def recentre(z, tc, site):
    """站点居中：TC 用同站点留一 TC 的均值与 SD；ASD 用同站点全部 TC。"""
    out = np.empty_like(z)
    for s in np.unique(site):
        ix_tc = np.where((site == s) & tc)[0]
        ix_asd = np.where((site == s) & ~tc)[0]
        Zt = z[ix_tc].astype(np.float64)
        k = len(ix_tc)
        S, SS = Zt.sum(0), (Zt ** 2).sum(0)
        for j, i in enumerate(ix_tc):
            m = (S - Zt[j]) / (k - 1)
            v = (SS - Zt[j] ** 2 - (k - 1) * m ** 2) / (k - 2)
            out[i] = (z[i] - m) / np.sqrt(v)
        if len(ix_asd):
            out[ix_asd] = (z[ix_asd] - Zt.mean(0)) / Zt.std(0, ddof=1)
    return out


def pos_agreement(a, b, thr=2.0):
    ca, cb = np.sign(a) * (np.abs(a) > thr), np.sign(b) * (np.abs(b) > thr)
    both = ((ca != 0) & (ca == cb)).sum()
    return float((both / max((ca != 0).sum(), 1) + both / max((cb != 0).sum(), 1)) / 2)


def max_prev(z, thr=2.0):
    return float(max((z > thr).mean(0).max(), (z < -thr).mean(0).max()))


def site_perm(site, rng):
    perm = np.arange(len(site))
    for s in np.unique(site):
        ix = np.where(site == s)[0]
        perm[ix] = rng.permutation(ix)
    return perm


def boot_median_icc(A, B, nboot=200):
    lo, hi = [], []
    n = len(A)
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        v = np.array([np.median(icc21(A[ix], B[ix])) for ix in (rng.integers(0, n, n) for _ in range(nboot))])
        lo.append(np.quantile(v, 0.025))
        hi.append(np.quantile(v, 0.975))
    return [round(float(np.mean(lo)), 4), round(float(np.mean(hi)), 4)]


def counts_by(zz, iu, labels, thr=2.0):
    """每人每个区域（或网络）的极端连接数：连接计入其两个端点所属区域。"""
    ext = (np.abs(zz) > thr).astype(np.float32)
    L = np.unique(labels)
    M = np.zeros((zz.shape[1], len(L)), dtype=np.float32)
    li = {l: k for k, l in enumerate(L)}
    for e, (a, b) in enumerate(zip(*iu)):
        M[e, li[labels[a]]] += 1
        M[e, li[labels[b]]] += 1
    return ext @ M


def jaccard_matrix(zz, thr=2.0):
    x = np.hstack([zz > thr, zz < -thr]).astype(np.float32)
    inter = x @ x.T
    s = x.sum(1)
    union = s[:, None] + s[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B28_edge_level.py"}
    sl = pd.read_csv(ROOT / "derivatives/norms_B1/B15_subject_level.csv")
    E, pt, iu = load_edges(sl)
    tc = (sl.dx == 2).to_numpy()
    site = sl.site.to_numpy()
    coh = sl.cohort.to_numpy()
    X = np.asarray(dmatrix("bs(age, df=4) + C(sex) + meanfd_retained_fd20", sl, return_type="dataframe"))
    OUT.mkdir(parents=True, exist_ok=True)
    Z = {}
    for c in CONDS:
        Z[c] = recentre(normative_z(E[c], X, tc, site), tc, site).astype(np.float32)
        np.save(OUT / f"edge_zrc_{c}.npy", Z[c])
        print("z", c, "tc mean/sd", round(float(Z[c][tc].mean()), 4), round(float(Z[c][tc].std()), 4), flush=True)
    del E
    out["n_edges"] = int(Z["full"].shape[1])
    out["calibration_tc_full"] = {"mean": round(float(Z["full"][tc].mean()), 4), "sd": round(float(Z["full"][tc].std()), 4),
                                  "share_abs_gt2": round(float((np.abs(Z["full"][tc]) > 2).mean()), 4)}
    region = pt.name.to_numpy()
    network = pt.network.to_numpy()
    res = {}
    for c in ["abide1", "abide2"]:
        for g, code in [("TC", True), ("ASD", False)]:
            m = (coh == c) & (tc == code)
            A, B, F, s = Z["half1"][m], Z["half2"][m], Z["full"][m], site[m]
            icc = icc21(A, B)
            rng = np.random.default_rng(42)
            null = [pos_agreement(A, B[site_perm(s, rng)]) for _ in range(200)]
            rA, rB = counts_by(A, iu, region), counts_by(B, iu, region)
            nA, nB = counts_by(A, iu, network), counts_by(B, iu, network)
            res[f"{g}_{c}"] = {
                "n": int(m.sum()),
                "a_median_icc": round(float(np.median(icc)), 4), "a_ci95": boot_median_icc(A, B),
                "a_share_icc_ge_0.5": round(float((icc >= 0.5).mean()), 5),
                "b_p2": round(pos_agreement(A, B), 4), "b_p2_chance": round(float(np.mean(null)), 4),
                "c_max_prevalence_full": round(max_prev(F), 4),
                "c_max_prevalence_half1": round(max_prev(A), 4), "c_max_prevalence_half2": round(max_prev(B), 4),
                "d_region_count_median_icc": round(float(np.median(icc21(rA, rB))), 4),
                "d_network_count_median_icc": round(float(np.median(icc21(nA, nB))), 4),
                "extreme_rate_full": round(float((np.abs(F) > 2).mean()), 4)}
            print("res", g, c, res[f"{g}_{c}"], flush=True)
    out["by_group"] = res

    jac = {}
    for c in ["abide1", "abide2"]:
        m = coh == c
        J = jaccard_matrix(Z["full"][m])
        st, lab = site[m], np.where(tc[m], 2, 1)
        iu2 = np.triu_indices(m.sum(), 1)
        same = (st[:, None] == st[None, :])[iu2]
        v = J[iu2]

        def gm(lb, mask):
            ok = (lb[:, None] == lb[None, :])[iu2]
            a = np.nanmean(v[ok & mask & (lb[iu2[0]] == 1)])
            t = np.nanmean(v[ok & mask & (lb[iu2[0]] == 2)])
            return a, t
        a_all, t_all = gm(lab, np.ones_like(same))
        a_bs, t_bs = gm(lab, ~same)
        rng = np.random.default_rng(42)
        null = []
        for _ in range(10000):
            lp = lab[site_perm(st, rng)]
            a, t = gm(lp, ~same)
            null.append(a - t)
        p = (1 + np.sum(np.abs(null) >= abs(a_bs - t_bs))) / 10001
        jac[c] = {"all_pairs": [round(float(a_all), 5), round(float(t_all), 5)],
                  "between_site": [round(float(a_bs), 5), round(float(t_bs), 5)], "between_site_perm_p": round(float(p), 4)}
        print("jaccard", c, jac[c], flush=True)
    out["e_between_site_jaccard_ASD_TC"] = jac
    (ROOT / "results/B28_edge_level.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
