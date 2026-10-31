"""课题 B1 汇总表。

T2 信度汇总: B14 主分析的 P1（中位 ICC、IQR、跨 seed 95% CI、ICC ≥ 0.5 比例）、P2（正向一致率、κ）、ASD − TC 差与置换 P；
    S1（跨 run）、S3（网络层面，分半与跨 run）同表。
T3 敏感性分析: P1 中位 ICC（95% CI）与 P2 正向一致率，按队列 × 组：主分析、FD 0.3、奇偶分半、Gordon、ComBat-GAM（B14 变体）、
    IPW、仅男性（B17）。缺失的变体结果文件记为"未运行"。
输出: tables_B1/T2_reliability.csv、T3_sensitivity.csv
用法: .venv_B/Scripts/python.exe -I scripts/B19_tables_T2_T3.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables_B1"
PAIR_LAB = {"P1P2_strength_half": "Parcel FC strength, split-half (P1/P2)", "S1_strength_run": "Parcel FC strength, between-run (S1)",
            "S3_netfc_half": "Network FC, split-half (S3)", "S3_netfc_run": "Network FC, between-run (S3)"}
UNIT_LAB = {"ABIDE_I": "ABIDE I", "ABIDE_II": "ABIDE II", "pooled": "Pooled"}


def ci(x):
    return f"{x['median_icc']:.3f} ({x['ci95_mean_over_seeds'][0]:.3f}–{x['ci95_mean_over_seeds'][1]:.3f})"


def p_fmt(p):
    return "<.001" if p < 0.001 else f"{p:.3f}"


def main():
    TAB.mkdir(exist_ok=True)
    b14 = json.loads((RES / "B14_reliability.json").read_text(encoding="utf-8"))["results"]
    rows = []
    for pair, units in b14.items():
        for u, r in units.items():
            for g in ["TC", "ASD"]:
                x = r[g]
                rows.append({"Analysis": PAIR_LAB[pair], "Cohort": UNIT_LAB[u], "Group": g, "n": x["n"],
                             "Median ICC (95% CI)": ci(x), "IQR": f"{x['iqr'][0]:.3f}–{x['iqr'][1]:.3f}",
                             "Features ICC ≥ 0.5, %": f"{100 * x['share_features_icc_ge_0.5']:.1f}",
                             "Extreme positive agreement, %": f"{100 * x['extreme']['pos_agreement_mean_dir']:.1f}",
                             "κ (3-category)": f"{x['extreme']['kappa_3cat']:.3f}",
                             "ASD − TC (P)": f"{r['ASD_minus_TC_median_icc']:+.3f} ({p_fmt(r['perm_p_two_sided'])})" if g == "ASD" else ""})
    pd.DataFrame(rows).to_csv(TAB / "T2_reliability.csv", index=False, encoding="utf-8")

    srcs = [("Primary analysis", RES / "B14_reliability.json", None), ("FD threshold 0.3 mm", RES / "B14_reliability_fd30.json", None),
            ("Odd/even frame split", RES / "B14_reliability_oddeven.json", None), ("Gordon333 atlas", RES / "B14_reliability_gordon.json", None),
            ("ComBat-GAM instead of site recentring", RES / "B14_reliability_combat.json", None),
            ("Inverse-probability weighting", RES / "B17_sens_males_ipw_S6.json", "ipw"), ("Males only", RES / "B17_sens_males_ipw_S6.json", "males_only")]
    rows = []
    for lab, f, key in srcs:
        if not f.exists():
            rows.append({"Analysis": lab, "Cohort": "", "Group": "", "n": "", "Median ICC (95% CI)": "未运行", "Extreme positive agreement, %": ""})
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for u in ["ABIDE_I", "ABIDE_II"]:
            for g in ["TC", "ASD"]:
                if key is None:
                    x = d["results"]["P1P2_strength_half"][u][g]
                    diff = d["results"]["P1P2_strength_half"][u]
                else:
                    x = (d["ipw"]["results"] if key == "ipw" else d["males_only"])[f"{u}_{g}"]
                    diff = None
                ext = x["extreme"]["pos_agreement_mean_dir"] if isinstance(x.get("extreme"), dict) else None
                rows.append({"Analysis": lab, "Cohort": UNIT_LAB[u], "Group": g, "n": x["n"], "Median ICC (95% CI)": ci(x),
                             "Extreme positive agreement, %": "" if ext is None else f"{100 * ext:.1f}",
                             "ASD − TC (P)": f"{diff['ASD_minus_TC_median_icc']:+.3f} ({p_fmt(diff['perm_p_two_sided'])})" if diff and g == "ASD" else ""})
    pd.DataFrame(rows).to_csv(TAB / "T3_sensitivity.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
