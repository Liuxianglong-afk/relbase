"""课题 B1 F7：噪声基线模拟图（读 results/B20_noise_sim.json，不重跑模拟）。

(a) 平均两两 Jaccard ~ 分区信度，每条线一个共享比例 s（有序蓝色色阶，dataviz 蓝 ramp 250→700）；虚线 = ABIDE 全长预测信度。
(b) 在 ABIDE 全长预测信度处：各 s 的模拟值（柱，误差线 = 跨种子 SD）与实测 ASD、TC（两个队列）水平线对比。
    2026-11-02 改：实测值改用跨站点配对（results/B22_s5_site_pairs.json），与正文一致；全部配对含站点居中伪影。
输出: figs_B1/F7_noise_baseline.png/.pdf
用法: .venv_B/Scripts/python.exe -I scripts/B20f_plot_noise_baseline.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
S_COL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
OBS_COL = {"TC": "#2a78d6", "ASD": "#eb6834"}


def main():
    d = json.loads((ROOT / "results/B20_noise_sim.json").read_text(encoding="utf-8"))
    g = d["grid"]
    S = sorted({v["s"] for v in g.values()})
    R = sorted({v["r"] for v in g.values()})
    r_obs = d["at_observed_reliability"]["r"]
    b22 = json.loads((ROOT / "results/B22_s5_site_pairs.json").read_text(encoding="utf-8"))["results"]
    obs = {g: {c: b22[c]["between_site_pairs"][g] for c in ["abide1", "abide2"]} for g in ["ASD", "TC"]}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
    for s, col in zip(S, S_COL):
        a.plot(R, [g[f"s={s}|r={r}"]["mean"] for r in R], color=col, lw=2, marker="o", ms=3, label=f"s = {s:g}")
    a.axvline(r_obs, color=MUTED, lw=0.8, ls="--")
    a.text(r_obs + 0.015, 0.30, f"ABIDE\nfull-length\nICC ≈ {r_obs:.2f}", fontsize=6.5, color=MUTED, va="top")
    a.set_xlabel("Parcel reliability (ICC)"); a.set_ylabel("Mean pairwise Jaccard of extremes")
    a.set_title("a  Simulated overlap", loc="left", fontsize=9)
    a.legend(title="Shared fraction of\ntrue deviation", title_fontsize=6.5, frameon=False, fontsize=6.5, loc="upper left")
    x = list(range(len(S)))
    vals = [g[f"s={s}|r={r_obs}"]["mean"] for s in S]
    sds = [g[f"s={s}|r={r_obs}"]["sd_over_seeds"] for s in S]
    b.bar(x, vals, yerr=sds, color=S_COL, width=0.62, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    for i, v in zip(x, vals):
        b.text(i, v + 0.0025, f"{v:.3f}", ha="center", fontsize=6.5, color=INK)
    for grp in ["ASD", "TC"]:
        for c, lab, ls in [("abide1", "I", "-"), ("abide2", "II", (0, (4, 2)))]:
            b.axhline(obs[grp][c], color=OBS_COL[grp], lw=1.4, ls=ls, label=f"{grp}, ABIDE {lab}: {obs[grp][c]:.3f}")
    b.legend(title="Observed, between-site pairs", title_fontsize=6.3, frameon=False, fontsize=6.3, loc="upper left")
    b.grid(axis="x", visible=False)
    b.set_axisbelow(True)
    b.set_xticks(x); b.set_xticklabels([f"{s:g}" for s in S])
    b.set_xlabel("Shared fraction s"); b.set_ylabel("Mean pairwise Jaccard")
    b.set_title(f"b  At ICC ≈ {r_obs:.2f}", loc="left", fontsize=9)
    b.set_ylim(0, max(vals) * 1.45)
    fig.tight_layout()
    fig.savefig(ROOT / "figs_B1/F7_noise_baseline.png", dpi=300)
    fig.savefig(ROOT / "figs_B1/F7_noise_baseline.pdf")


if __name__ == "__main__":
    main()
