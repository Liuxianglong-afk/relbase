"""课题 B1 事后分析（2026-10-31，登记于方案第 12 节）：测量噪声基线模拟。

问题: 在实测信度下，"个体间极端偏离重叠"能否区分 ASD 的异质（真实偏离各不相同）与同质（真实偏离相同）？
生成模型（每人 379 个分区）:
  真实偏离 T_i = sqrt(s)·C + sqrt(1-s)·U_i       C 为全组共享模式（每次重复抽一次），U_i 为个人特有模式
  观察偏离 z_i = k·[sqrt(r)·T_i + sqrt(1-r)·E_i]   r = 分区信度（ICC），E_i 为测量噪声，k = 实测 ASD z 的标准差
  空间相关: C、U ~ N(0, R_obs)，R_obs = TC 全长 z 的分区间相关；E ~ N(0, R_noise)，R_noise = TC 两半差值的分区间相关。
验证: s = 0、r = 实测半长 ICC 时，同一被试两次独立测量的极端偏离正向一致率，应接近实测 P2（B14）。
主分析: r ∈ {0.1, …, 0.9} ∪ {实测全长预测 ICC} × s ∈ {0, 0.25, 0.5, 0.75, 1}；n = ABIDE I ASD 人数；
  指标 = 极端偏离（|z| > 2，分区 × 符号）的两两 Jaccard 平均数（与 B15 S5 事后指标相同）。
  另报：在实测 r 下，s = 0 与 s = 1 的 Jaccard 之差（指标的可分辨范围），以及与实测 ASD、TC 平均 Jaccard 匹配的 s。
种子: 42 / 7 / 123 / 2024，每个种子 25 次重复；报告跨种子均值 ± SD。
输出: results/B20_noise_sim.json；figs_B1/F7_noise_baseline.png/.pdf
用法: .venv_B/Scripts/python.exe -I scripts/B20_noise_baseline_sim.py [--reps 25]
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derivatives/norms_B1"
SEEDS = [42, 7, 123, 2024]
S_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
R_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
S_COL = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#0d366b"]  # 有序色阶（dataviz 蓝色 ramp 150→700）
OBS_COL = {"TC": "#2a78d6", "ASD": "#eb6834"}


def wide(cond):
    z = pd.read_csv(NB / f"B13_zrc_strength_{cond}.csv.gz")
    z = z[z.site_ok]
    return z.pivot_table(index=["cohort", "sub_n", "dx"], columns="feature", values="zrc").dropna()


def mean_jaccard(z, thr=2.0):
    x = np.hstack([z > thr, z < -thr]).astype(np.float64)
    inter = x @ x.T
    s = x.sum(1)
    union = s[:, None] + s[None, :] - inter
    iu = np.triu_indices(len(x), 1)
    u, i = union[iu], inter[iu]
    ok = u > 0
    return float((i[ok] / u[ok]).mean())


def pos_agreement(a, b, thr=2.0):
    ca, cb = np.sign(a) * (np.abs(a) > thr), np.sign(b) * (np.abs(b) > thr)
    both = ((ca != 0) & (ca == cb)).sum()
    return float((both / max((ca != 0).sum(), 1) + both / max((cb != 0).sum(), 1)) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B20_noise_baseline_sim.py", "reps_per_seed": args.reps}

    full, h1, h2 = wide("full"), wide("half1"), wide("half2")
    idx = full.index.intersection(h1.index).intersection(h2.index)
    full, h1, h2 = full.loc[idx], h1.loc[idx], h2.loc[idx]
    dx = idx.get_level_values("dx")
    coh = idx.get_level_values("cohort")
    tc = dx == 2
    R_obs = np.corrcoef(full.values[tc].T)
    R_noise = np.corrcoef(((h1.values[tc] - h2.values[tc]) / np.sqrt(2)).T)
    eps = 1e-6 * np.eye(R_obs.shape[0])
    L_obs, L_noise = np.linalg.cholesky(R_obs + eps), np.linalg.cholesky(R_noise + eps)
    k_asd = float(full.values[~tc].std())
    b14 = json.loads((ROOT / "results/B14_reliability.json").read_text(encoding="utf-8"))["results"]["P1P2_strength_half"]
    r_half = float(np.mean([b14[u][g]["median_icc"] for u in ["ABIDE_I", "ABIDE_II"] for g in ["TC", "ASD"]]))
    r_full = 2 * r_half / (1 + r_half)
    p2_obs = float(np.mean([b14[u][g]["extreme"]["pos_agreement_mean_dir"] for u in ["ABIDE_I", "ABIDE_II"] for g in ["TC", "ASD"]]))
    n = int(((coh == "abide1") & ~tc).sum())
    obs = {}
    for g, code in [("TC", 2), ("ASD", 1)]:
        obs[g] = {c: mean_jaccard(full.values[(dx == code) & (coh == c)]) for c in ["abide1", "abide2"]}
    out["inputs"] = {"n_sim": n, "k_asd_sd": round(k_asd, 4), "r_half_obs": round(r_half, 4), "r_full_pred": round(r_full, 4),
                     "p2_obs": round(p2_obs, 4), "obs_mean_jaccard": {g: {c: round(v, 4) for c, v in d.items()} for g, d in obs.items()},
                     "R_obs_mean_offdiag": round(float(R_obs[np.triu_indices_from(R_obs, 1)].mean()), 4),
                     "R_noise_mean_offdiag": round(float(R_noise[np.triu_indices_from(R_noise, 1)].mean()), 4)}
    P = R_obs.shape[0]

    def draw(rng, L, m):
        return rng.standard_normal((m, P)) @ L.T

    # 验证：两次独立测量的正向一致率
    val = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        v = []
        for _ in range(args.reps):
            T = draw(rng, L_obs, n)
            a = k_asd * (np.sqrt(r_half) * T + np.sqrt(1 - r_half) * draw(rng, L_noise, n))
            b = k_asd * (np.sqrt(r_half) * T + np.sqrt(1 - r_half) * draw(rng, L_noise, n))
            v.append(pos_agreement(a, b))
        val.append(np.mean(v))
    out["validation_p2"] = {"sim_mean": round(float(np.mean(val)), 4), "sim_sd_over_seeds": round(float(np.std(val)), 4), "observed": round(p2_obs, 4)}
    print("validation", out["validation_p2"], flush=True)

    # 主网格
    rs = sorted(set(R_GRID + [round(r_full, 3)]))
    grid = {}
    for s in S_GRID:
        for r in rs:
            per_seed = []
            for sd in SEEDS:
                rng = np.random.default_rng(sd + int(1000 * s) + int(1e5 * r))
                vals = []
                for _ in range(args.reps):
                    C = draw(rng, L_obs, 1)
                    T = np.sqrt(s) * C + np.sqrt(1 - s) * draw(rng, L_obs, n)
                    z = k_asd * (np.sqrt(r) * T + np.sqrt(1 - r) * draw(rng, L_noise, n))
                    vals.append(mean_jaccard(z))
                per_seed.append(np.mean(vals))
            grid[f"s={s}|r={r}"] = {"s": s, "r": r, "mean": round(float(np.mean(per_seed)), 5), "sd_over_seeds": round(float(np.std(per_seed)), 5)}
        print("s", s, [grid[f"s={s}|r={r}"]["mean"] for r in rs], flush=True)
    out["grid"] = grid
    at = {s: grid[f"s={s}|r={round(r_full, 3)}"]["mean"] for s in S_GRID}
    at9 = {s: grid[f"s={s}|r=0.9"]["mean"] for s in S_GRID}
    out["at_observed_reliability"] = {"r": round(r_full, 3), "jaccard_by_s": at, "range_s0_to_s1": round(at[1.0] - at[0.0], 5)}
    out["at_r_0.9"] = {"jaccard_by_s": at9, "range_s0_to_s1": round(at9[1.0] - at9[0.0], 5)}
    xs = np.array(S_GRID)
    ys = np.array([at[s] for s in S_GRID])
    out["implied_shared_fraction_at_observed_r"] = {g: {c: round(float(np.interp(v, ys, xs)), 3) for c, v in d.items()} for g, d in obs.items()}
    (ROOT / "results/B20_noise_sim.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图由 scripts/B20f_plot_noise_baseline.py 从本 JSON 绘制
    print(json.dumps({k: out[k] for k in ["inputs", "validation_p2", "at_observed_reliability", "at_r_0.9", "implied_shared_fraction_at_observed_r"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
