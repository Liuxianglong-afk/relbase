"""课题 B1 影像图（fsLR32k inflated 表面，Glasser 360 皮层分区；皮层下 19 个结构不上表面图）。

F3 分区信度脑图: B14 的 P1 分半 ICC(2,1)，4 行（ABIDE I TC、I ASD、II TC、II ASD）× 4 视角（左外侧、左内侧、右外侧、右内侧）。
   单色相顺序色阶，0–0.5 固定范围。
F4 个体示例: 预设选择规则 = ABIDE I 中两半偏离图空间 r 最接近本组中位数的 ASD 与 TC 各 1 人（B15_subject_level.csv）；
   每人 2 行（前半、后半）× 4 视角，z 截断于 ±3，双色发散色阶、中点灰色；标题注明各半极端偏离数与两半同号重复数。
   被选被试的编号只写入 logs/B16_selected_examples_LOCAL_ONLY.json，不出现在图上。
F5 达到 ICC 0.5 所需时长: 按分区，用两个队列 TC 的 ICC 均值与 TC 半长中位秒数做 Spearman–Brown 外推（分钟，截断于 60）。
输出: figs_B1/F3_icc_brainmap.png/.pdf、F4_individual_example.png/.pdf、F6_minutes_needed.png/.pdf；results/B16_brain_figures.json
用法: .venv_B/Scripts/python.exe -I scripts/B16_brain_figures.py
"""
import datetime as dt
import json
from pathlib import Path

import hcp_utils as hcp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from nilearn import plotting  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "derivatives/norms_B1"
FIG = ROOT / "figs_B1"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef3fa", "#9ecae1", "#4292c6", "#08519c", "#08306b"])
DIV = LinearSegmentedColormap.from_list("div_bgr", ["#2166ac", "#92c5de", "#d9d9d9", "#f4a582", "#b2182b"])
VIEWS = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]
NL = hcp.vertex_info.num_meshl

pt = pd.read_csv(ROOT / "docs_B/glasser_parcel_table.csv")
name2mmp = dict(zip(pt.name, pt.mmp_id))
vlab = np.load(ROOT / "docs_B/glasser_cortex_vertex_labels.npy")


def to_surface(values_by_name):
    """分区值（名称 → 值）→ 64984 顶点数组，左右半球分开返回；内侧壁为 NaN。"""
    lut = np.full(361, np.nan)
    for n, v in values_by_name.items():
        if n in name2mmp and name2mmp[n] > 0:
            lut[name2mmp[n]] = v
    ctx = lut[vlab]
    full = np.full(2 * NL, np.nan)
    full[np.r_[hcp.vertex_info.grayl, NL + hcp.vertex_info.grayr]] = ctx
    return full[:NL], full[NL:]


def draw_row(fig, axes, vals, cmap, vmin, vmax):
    left, right = to_surface(vals)
    for ax, (hemi, view) in zip(axes, VIEWS):
        mesh = hcp.mesh.inflated_left if hemi == "left" else hcp.mesh.inflated_right
        bg = hcp.mesh.sulc_left if hemi == "left" else hcp.mesh.sulc_right
        data = left if hemi == "left" else right
        plotting.plot_surf(mesh, data, hemi=hemi, view=view, bg_map=bg, cmap=cmap, vmin=vmin, vmax=vmax,
                           bg_on_data=False, axes=ax, figure=fig, colorbar=False, engine="matplotlib")
        ax.set_box_aspect(None, zoom=1.45)


def colorbar(fig, rect, cmap, vmin, vmax, label, ticks=None):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def grid(nrows, row_labels, title):
    fig = plt.figure(figsize=(9.2, 2.0 * nrows + 1.35))
    axes = [[fig.add_subplot(nrows, 4, r * 4 + c + 1, projection="3d") for c in range(4)] for r in range(nrows)]
    plt.subplots_adjust(left=0.13, right=0.99, top=1 - 0.45 / (2.0 * nrows + 1.35), bottom=1.25 / (2.0 * nrows + 1.35), wspace=0.0, hspace=0.02)
    for r, lab in enumerate(row_labels):
        y = np.mean([axes[r][0].get_position().y0, axes[r][0].get_position().y1])
        fig.text(0.065, y, lab, ha="center", va="center", fontsize=9)
    for c, (h, v) in enumerate(VIEWS):
        x = np.mean([axes[0][c].get_position().x0, axes[0][c].get_position().x1])
        fig.text(x, axes[0][c].get_position().y1 + 0.004, f"{'Left' if h == 'left' else 'Right'} {v}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(title, fontsize=11, y=0.998, va="top")
    return fig, axes


def save(fig, stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300)
    fig.savefig(FIG / f"{stem}.pdf")
    plt.close(fig)


def main():
    left_ok = (vlab[: len(hcp.vertex_info.grayl)] <= 180).all() and (vlab[len(hcp.vertex_info.grayl):] > 180).all()
    assert left_ok, "顶点标签与半球顺序不符"
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B16_brain_figures.py"}

    icc = pd.read_csv(NB / "B14_icc_per_feature.csv")
    icc = icc[icc.pair == "P1P2_strength_half"]
    rows = [("ABIDE_I", "TC"), ("ABIDE_I", "ASD"), ("ABIDE_II", "TC"), ("ABIDE_II", "ASD")]
    labels = [f"{u.replace('_', ' ')}\n{g}" for u, g in rows]
    fig, axes = grid(4, labels, "Split-half reliability of FC-strength deviations, ICC(2,1)")
    for r, (u, g) in enumerate(rows):
        d = icc[(icc.unit == u) & (icc.group == g)]
        draw_row(fig, axes[r], dict(zip(d.feature, d.icc)), SEQ, 0, 0.5)
    colorbar(fig, [0.35, 0.075, 0.45, 0.016], SEQ, 0, 0.5, "ICC(2,1)", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    save(fig, "F3_icc_brainmap")
    ctx = icc[icc.feature.map(name2mmp).fillna(0) > 0]
    out["F3"] = {f"{u}_{g}": {"cortical_median": round(float(ctx[(ctx.unit == u) & (ctx.group == g)].icc.median()), 4),
                              "cortical_max": round(float(ctx[(ctx.unit == u) & (ctx.group == g)].icc.max()), 4)} for u, g in rows}
    tc = ctx[ctx.group == "TC"].groupby("feature").icc.mean()
    out["F3"]["top5_tc_mean"] = {k: round(float(v), 3) for k, v in tc.sort_values(ascending=False).head(5).items()}
    out["F3"]["bottom5_tc_mean"] = {k: round(float(v), 3) for k, v in tc.sort_values().head(5).items()}

    # F4 个体示例
    sl = pd.read_csv(NB / "B15_subject_level.csv")
    pick = {}
    for g in ["ASD", "TC"]:
        d = sl[(sl.cohort == "abide1") & (sl.group == g)].copy()
        d["dist"] = (d.r_half - d.r_half.median()).abs()
        pick[g] = d.sort_values(["dist", "sub_n"]).iloc[0]
    zz = {c: pd.read_csv(NB / f"B13_zrc_strength_{c}.csv.gz") for c in ["half1", "half2"]}
    labels, maps, stats = [], [], {}
    for g in ["ASD", "TC"]:
        p = pick[g]
        seg = {}
        for c in ["half1", "half2"]:
            z = zz[c]
            seg[c] = z[(z.cohort == p.cohort) & (z.sub_n == p.sub_n)].set_index("feature").zrc
        e1, e2 = seg["half1"].abs() > 2, seg["half2"].abs() > 2
        both = int((e1 & e2 & (np.sign(seg["half1"]) == np.sign(seg["half2"]))).sum())
        stats[g] = {"r_half": round(float(p.r_half), 3), "n_ext_half1": int(e1.sum()), "n_ext_half2": int(e2.sum()), "n_ext_both_same_sign": both}
        for c, nm in [("half1", "first half"), ("half2", "second half")]:
            labels.append(f"{g}, {nm}\n|z|>2: {int((seg[c].abs() > 2).sum())}")
            maps.append(seg[c].clip(-3, 3).to_dict())
    fig, axes = grid(4, labels, "Individual FC-strength deviation maps: two halves of one scan")
    for r, m in enumerate(maps):
        draw_row(fig, axes[r], m, DIV, -3, 3)
    colorbar(fig, [0.35, 0.075, 0.45, 0.016], DIV, -3, 3, "Deviation z (site-recentred)", [-3, -2, -1, 0, 1, 2, 3])
    # 两半同号重复数与选择规则写入图注（results/B16_brain_figures.json），不放在图内
    save(fig, "F4_individual_example")
    out["F4"] = {"selection_rule": "ABIDE I 中两半 z 图空间 r 最接近本组中位数者（并列取编号较小者）", "stats": stats}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/B16_selected_examples_LOCAL_ONLY.json").write_text(
        json.dumps({g: {"cohort": pick[g].cohort, "sub_n": int(pick[g].sub_n)} for g in pick}, indent=2), encoding="utf-8")

    # F5 所需时长
    b15 = json.loads((ROOT / "results/B15_secondary.json").read_text(encoding="utf-8"))["S2"]["spearman_brown"]
    L = np.mean([b15["ABIDE_I_TC"]["half_sec_median"], b15["ABIDE_II_TC"]["half_sec_median"]])
    rho = tc.clip(lower=0.01)
    minutes = (0.5 * (1 - rho) / (rho * 0.5) * L / 60).clip(upper=60)
    fig, axes = grid(1, ["TC\n(both cohorts)"], "Scan time needed for ICC = 0.5 (Spearman–Brown, minutes after censoring)")
    draw_row(fig, axes[0], minutes.to_dict(), SEQ.reversed(), 0, 30)
    colorbar(fig, [0.35, 0.22, 0.45, 0.045], SEQ.reversed(), 0, 30, "Minutes (capped at 30 for display)", [0, 5, 10, 15, 20, 25, 30])
    save(fig, "F6_minutes_needed")
    out["F5"] = {"half_sec_used": round(float(L), 1), "median_minutes": round(float(minutes.median()), 1),
                 "q10_q90_minutes": [round(float(x), 1) for x in minutes.quantile([0.1, 0.9])],
                 "share_parcels_le_10min": round(float((minutes <= 10).mean()), 3)}
    (ROOT / "results/B16_brain_figures.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
