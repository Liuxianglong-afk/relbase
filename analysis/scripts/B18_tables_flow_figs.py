"""课题 B1 表与非脑图：T1 人口学、F1 被试流程、F2 常模校准、F6 次要终点（S2 时长、S4 结构–功能、S5 可重复极端）。

T1: 主分析样本（pass_fd20 且 site_ok），按队列 × 诊断：人数、年龄、男性比例、FIQ、平均 FD、剔帧后时长、ADOS CSS、SRS 原始总分；
    组间比较 Welch t / χ²。另报全部质控合格者（985）的同一组表（补充材料）。
F1: 表型表人数 → DCAN 衍生数据齐全 → 质控排除（按原因）→ 质控合格 → 本地 TC < 10 站点（敏感性）→ 主分析；另注跨 run 子样本。
F2: (a) TC 全长 z' 的直方图与 N(0,1) 及留一理论分布；(b) 各站点 z' 标准差：观察值 vs 留一理论值。
F6: (a) 两半偏离图空间 r ~ 每半秒数（散点 + GAM 预测）；(b) 结构–功能 ρ 的分布（TC、ASD）；(c) 每人可重复极端数的分布。
配色: TC #2a78d6、ASD #eb6834（dataviz 默认分类色 1–2，validate_palette.js 浅色模式全部通过）。
输出: tables_B1/T1_demographics.csv、T1s_demographics_all985.csv；figs_B1/F1_flow、F2_calibration、F5_secondary（.png 300 dpi + .pdf）；
      results/B18_tables_flow.json
用法: .venv_B/Scripts/python.exe -I scripts/B18_tables_flow_figs.py
"""
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derivatives/norms_B1"
FIG, TAB = ROOT / "figs_B1", ROOT / "tables_B1"
COL = {"TC": "#2a78d6", "ASD": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def fmt(x, d=1):
    x = x.dropna()
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}" if len(x) else "—"


def t1(df):
    rows = []
    for coh, d in df.groupby("cohort"):
        a, t = d[d.dx == 1], d[d.dx == 2]
        def p_t(c):
            x, y = a[c].dropna(), t[c].dropna()
            return stats.ttest_ind(x, y, equal_var=False).pvalue if len(x) > 1 and len(y) > 1 else np.nan
        chi = stats.chi2_contingency(pd.crosstab(d.dx, d.sex))[1]
        for lab, c, dd, p in [("N", None, 0, np.nan), ("Age, y", "age", 1, p_t("age")), ("Male, n (%)", "sex", 0, chi),
                              ("FIQ", "fiq", 1, p_t("fiq")), ("Mean FD of retained frames, mm", "meanfd_retained_fd20", 3, p_t("meanfd_retained_fd20")),
                              ("Retained time, s", "remain_sec_fd20", 0, p_t("remain_sec_fd20")),
                              ("ADOS CSS (n)", "ados_css", 1, np.nan), ("SRS raw total (n)", "srs_raw_total", 1, np.nan)]:
            if c is None:
                va, vt = str(len(a)), str(len(t))
            elif c == "sex":
                va, vt = f"{(a.sex == 1).sum()} ({100 * (a.sex == 1).mean():.0f})", f"{(t.sex == 1).sum()} ({100 * (t.sex == 1).mean():.0f})"
            elif c in ("ados_css", "srs_raw_total"):
                va, vt = f"{fmt(a[c], dd)} ({a[c].notna().sum()})", f"{fmt(t[c], dd)} ({t[c].notna().sum()})"
            else:
                va, vt = fmt(a[c], dd), fmt(t[c], dd)
            rows.append({"cohort": "ABIDE I" if coh == "abide1" else "ABIDE II", "variable": lab, "ASD": va, "TC": vt,
                         "P": "" if np.isnan(p) else ("<.001" if p < 0.001 else f"{p:.3f}")})
    return pd.DataFrame(rows)


def box(ax, x, y, w, h, text, fc="#f4f3ef"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=MUTED, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))


def save(fig, stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B18_tables_flow_figs.py"}
    qc = pd.read_csv(sorted((ROOT / "data/interim").glob("abide_rest_qc_*.csv"))[-1])
    ph = pd.read_csv(ROOT / "data/interim/abide_pheno_harmonized_20261014.csv")
    cal = json.loads((ROOT / "results/B13_calibration.json").read_text(encoding="utf-8"))
    ok_sites = set(cal["sites_ok"])
    qc = qc.rename(columns={"dx_group": "dx"}).merge(ph[["cohort", "sub_n", "fiq", "ados_css", "srs_raw_total"]], on=["cohort", "sub_n"], how="left")
    ana = qc[qc.pass_fd20 & qc.site.isin(ok_sites)]
    TAB.mkdir(exist_ok=True)
    t1(ana).to_csv(TAB / "T1_demographics.csv", index=False, encoding="utf-8")
    t1(qc[qc.pass_fd20]).to_csv(TAB / "T1s_demographics_all985.csv", index=False, encoding="utf-8")

    # F1 流程
    m11 = pd.read_csv(ROOT / "derivatives/features_B/B11_meta.csv")
    n_run = int(m11.merge(ana[["cohort", "sub_n"]]).sec_run1.notna().sum())
    flow = {"pheno": {c: int((ph.cohort == c).sum()) for c in ["abide1", "abide2"]},
            "dcan_complete": {c: int((qc.cohort == c).sum()) for c in ["abide1", "abide2"]},
            "excluded_by_reason": qc[~qc.pass_fd20].exclude_reason.fillna("other").value_counts().to_dict(),
            "qc_pass": {c: int((qc.pass_fd20 & (qc.cohort == c)).sum()) for c in ["abide1", "abide2"]},
            "small_site_excluded": int((qc.pass_fd20 & ~qc.site.isin(ok_sites)).sum()),
            "analysis": {f"{c}_{'ASD' if g == 1 else 'TC'}": int(((ana.cohort == c) & (ana.dx == g)).sum()) for c in ["abide1", "abide2"] for g in [1, 2]},
            "run_pairs_in_analysis": n_run}
    flow["no_complete_dcan"] = sum(flow["pheno"].values()) - sum(flow["dcan_complete"].values())
    out["flow"] = flow
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ph_n, dc = flow["pheno"], flow["dcan_complete"]
    # 两栏：主流程 x = 0.05–0.55，排除框 x = 0.63–0.97；框外 pad 0.01，栏间留白 ≥ 0.06，横向箭头止于排除框左缘
    XM, WM, XR, WR, PAD = 0.30, 0.50, 0.80, 0.34, 0.01
    main = {"pheno": (0.93, 0.075), "dcan": (0.78, 0.075), "qc": (0.54, 0.075), "ana": (0.22, 0.12), "run": (0.05, 0.075)}

    def mbox(k, text, **kw):
        box(ax, XM, main[k][0], WM, main[k][1], text, **kw)

    mbox("pheno", f"ABIDE I and II phenotypic records\nABIDE I n = {ph_n['abide1']}; ABIDE II n = {ph_n['abide2']}")
    box(ax, XR, 0.855, WR, 0.07, f"No complete DCAN derivatives\n(n = {flow['no_complete_dcan']})")
    mbox("dcan", f"DCAN (ABCD-HCP) derivatives complete\nABIDE I n = {dc['abide1']}; ABIDE II n = {dc['abide2']}")
    ex = flow["excluded_by_reason"]
    lab = {"remain_sec_fd20<240": "< 240 s retained after\ncensoring (FD > 0.2 mm)", "bad_parcels": "Parcels without signal"}
    ex_txt = "\n".join(f"{lab.get(k, k)}: {v}" for k, v in ex.items())
    box(ax, XR, 0.66, WR, 0.15, f"Excluded (n = {sum(ex.values())})\n{ex_txt}")
    qp = flow["qc_pass"]
    mbox("qc", f"Passed quality control (≥ 240 s retained)\nABIDE I n = {qp['abide1']}; ABIDE II n = {qp['abide2']}")
    box(ax, XR, 0.40, WR, 0.1, f"Sites with < 10 TC\n(n = {flow['small_site_excluded']})\nsensitivity analyses only")
    an = flow["analysis"]
    mbox("ana", f"Primary analysis (19 sites)\nABIDE I: ASD {an['abide1_ASD']}, TC {an['abide1_TC']}\n"
        f"ABIDE II: ASD {an['abide2_ASD']}, TC {an['abide2_TC']}", fc="#e9f1fb")
    mbox("run", f"Between-run subsample\ntwo runs of equal length, n = {n_run}")
    order = ["pheno", "dcan", "qc", "ana", "run"]
    for a, b in zip(order, order[1:]):
        arrow(ax, XM, main[a][0] - main[a][1] / 2 - PAD, XM, main[b][0] + main[b][1] / 2 + PAD)
    for y in (0.855, 0.66, 0.40):
        arrow(ax, XM, y, XR - WR / 2 - PAD, y)
    save(fig, "F1_flow")

    # F2 校准
    z = pd.read_csv(NB / "B13_zrc_strength_full.csv.gz")
    tc = z[(z.dx == 2) & z.site_ok]
    th = json.loads((ROOT / "results/B13v_calibration_theory.json").read_text(encoding="utf-8"))["sets"]["strength"]["per_site"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
    xs = np.linspace(-5, 5, 400)
    axs[0].hist(tc.zrc.clip(-5, 5), bins=80, density=True, color=COL["TC"], alpha=0.55, edgecolor="none", label="TC, observed")
    axs[0].plot(xs, stats.norm.pdf(xs), color=INK, lw=1.2, label="N(0, 1)")
    k = tc.groupby("site").sub_n.nunique()
    sc = float(np.sqrt(((1 + 1 / (k - 1)) * (k - 2) / (k - 4) * tc.groupby("site").size() / len(tc)).sum()))
    axs[0].plot(xs, stats.norm.pdf(xs, scale=sc), color=INK, lw=1.2, ls="--", label=f"Leave-one-out expectation (SD {sc:.3f})")
    axs[0].set_xlabel("Deviation z' (TC, full-length data, all parcels)"); axs[0].set_ylabel("Density")
    axs[0].legend(frameon=False, fontsize=6.5, loc="upper left"); axs[0].set_ylim(0, 0.52); axs[0].set_title("a  Calibration in TC", loc="left", fontsize=9)
    e = np.array([v["exp_sd"] for v in th.values()]); o = np.array([v["obs_sd"] for v in th.values()]); kk = np.array([v["k"] for v in th.values()])
    axs[1].plot([1.0, 1.3], [1.0, 1.3], color=MUTED, lw=0.8, ls="--")
    axs[1].scatter(e, o, s=np.clip(kk, 10, 90) * 0.9, color=COL["TC"], edgecolor="white", lw=0.6)
    axs[1].set_xlabel("Expected SD (leave-one-out, by site size)"); axs[1].set_ylabel("Observed SD")
    axs[1].set_title("b  Per-site SD (marker size = TC per site)", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "F2_calibration")

    # F6 次要终点
    sl = pd.read_csv(NB / "B15_subject_level.csv")
    b15 = json.loads((ROOT / "results/B15_secondary.json").read_text(encoding="utf-8"))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for g in ["TC", "ASD"]:
        d = sl[sl.group == g]
        axs[0].scatter(d.sec_half, d.r_half, s=8, color=COL[g], alpha=0.45, edgecolor="none", label=g)
    gg = b15["S2"]["gam"]
    axs[0].plot(gg["pred_grid_sec"], gg["pred_r_half_abide1_TC"], color=INK, lw=1.5, label="GAM fit (ABIDE I TC)")
    axs[0].set_xlabel("Retained time per half, s"); axs[0].set_ylabel("Split-half map correlation r")
    axs[0].set_title("a  Reliability vs duration", loc="left", fontsize=9); axs[0].legend(frameon=False, fontsize=7, markerscale=1.5)
    bins = np.linspace(-0.3, 0.3, 31)
    for g in ["TC", "ASD"]:
        axs[1].hist(sl[sl.group == g].rho_sf.dropna(), bins=bins, density=True, histtype="step", color=COL[g], lw=1.6, label=g)
    axs[1].axvline(0, color=MUTED, lw=0.8)
    axs[1].set_xlabel("Structure–function ρ (thickness z vs FC z)"); axs[1].set_ylabel("Density")
    axs[1].set_title("b  Structure–function", loc="left", fontsize=9); axs[1].legend(frameon=False, fontsize=7)
    ext = []
    w = {c: pd.read_csv(NB / f"B13_zrc_strength_{c}.csv.gz") for c in ["full", "half1", "half2"]}
    for c in w:
        w[c] = w[c][w[c].site_ok].pivot_table(index=["cohort", "sub_n", "dx"], columns="feature", values="zrc").dropna()
    idx = w["full"].index.intersection(w["half1"].index).intersection(w["half2"].index)
    f, a, b = (w[c].loc[idx].values for c in ["full", "half1", "half2"])
    n_full = (np.abs(f) > 2).sum(1)
    n_rep = (((a > 2) & (b > 2)) | ((a < -2) & (b < -2))).sum(1)
    dxv = idx.get_level_values("dx")
    maxn = 80
    width = 0.4
    for j, (g, code) in enumerate([("TC", 2), ("ASD", 1)]):
        for kind, vals, alpha, off in [("full data", n_full, 0.35, -width / 2), ("reproduced in both halves", n_rep, 0.9, width / 2)]:
            v = np.clip(vals[dxv == code], 0, maxn)
            med = np.median(vals[dxv == code])
            axs[2].boxplot(v, positions=[j + off], widths=0.32, patch_artist=True, showfliers=False, whis=(5, 95),
                           boxprops=dict(facecolor=COL[g], alpha=alpha, edgecolor=MUTED), medianprops=dict(color=INK), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
            axs[2].text(j + off, -3.2, f"{med:.0f}", ha="center", fontsize=7, color=INK)
    axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["TC", "ASD"]); axs[2].set_ylim(-6, 75)
    axs[2].set_ylabel("Extremes per person (|z| > 2)")
    axs[2].set_title("c  Full data vs reproducible", loc="left", fontsize=9)
    axs[2].text(0.5, 0.97, "light = full data · dark = both halves\nwhiskers 5–95th pct · numbers = medians", transform=axs[2].transAxes,
                ha="center", va="top", fontsize=6.2, color=MUTED)
    fig.tight_layout()
    save(fig, "F5_secondary")
    out["F6"] = {"median_n_extreme_full": {g: float(np.median(n_full[dxv == c])) for g, c in [("TC", 2), ("ASD", 1)]},
                 "median_n_reproduced": {g: float(np.median(n_rep[dxv == c])) for g, c in [("TC", 2), ("ASD", 1)]},
                 "share_extremes_reproduced_overall": round(float(n_rep.sum() / n_full.sum()), 4)}
    (ROOT / "results/B18_tables_flow.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
