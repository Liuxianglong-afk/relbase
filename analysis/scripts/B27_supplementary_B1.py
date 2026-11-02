"""课题 B1 补充材料：由结果文件生成补充表（Markdown + CSV），并核对冻结方案的 SHA-256。

输入: results/B1_protocol_freeze_v1.0.json、B14v2、B22、B25、B26、B21 结果；tables_B1/T1s_demographics_all985.csv；
      docs_B/protocol_B1_v1.0_frozen_ca187995.md（冻结原文存档）
输出: tables_B1/S*.csv；manuscript_B1/B1_supplementary_tables_v0.1.md（表格部分，由本脚本生成，勿手改）
用法: .venv_B/Scripts/python.exe -I scripts/B27_supplementary_B1.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
T = ROOT / "tables_B1"
sys.stdout.reconfigure(encoding="utf-8")
COH = {"abide1": "ABIDE I", "abide2": "ABIDE II", "ABIDE_I": "ABIDE I", "ABIDE_II": "ABIDE II"}


def fmt_p(p):
    """P 值写法与主稿一致：< .001，否则三位小数、无前导 0。"""
    return "< .001" if p < 0.001 else "= " + f"{p:.3f}".lstrip("0")


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    cell = lambda v: "" if pd.isna(v) else str(v).replace("|", chr(92) + "|")  # 单元格内竖线转义
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def split(k):
    g, c = k.split("_", 1)
    return COH[c], g


def main():
    out = []
    # 冻结方案哈希核对
    fr = load("B1_protocol_freeze_v1.0.json")
    rec = fr["files"]["protocol_B1_常模偏离_多尺度解释.md"]["sha256"]
    arch = hashlib.sha256((ROOT / "docs_B/protocol_B1_v1.0_frozen_ca187995.md").read_bytes()).hexdigest()
    freeze = pd.DataFrame([{"Document": "B1 analysis plan v1.0", "Frozen at": fr["frozen_at"], "SHA-256 recorded at freezing": rec,
                            "SHA-256 of archived copy": arch, "Match": "yes" if rec == arch else "NO"},
                           {"Document": "Shared preprocessing plan", "Frozen at": fr["frozen_at"],
                            "SHA-256 recorded at freezing": fr["files"]["protocol_B_rsfMRI共用流程.md"]["sha256"], "SHA-256 of archived copy": "", "Match": ""}])
    print("freeze hash match:", rec == arch, flush=True)
    tabs = {"S0_freeze": ("Table S1. Hashes of the frozen analysis plans. ‖ ASD result files present at freezing: "
                          + (", ".join(fr["asd_result_files_present"]) or "none") + ".", freeze)}

    t1s = pd.read_csv(T / "T1s_demographics_all985.csv")
    tabs["S1_demographics_all"] = ("Table S2. Characteristics of all 985 participants passing quality control, including sites excluded from the primary analysis.", t1s)

    raw = load("B14v2_raw_vs_z_icc.json")["results"]
    tabs["S2_raw_vs_z"] = ("Table S3. Median split-half ICC of parcel FC strength before and after normative modelling.", pd.DataFrame(
        [{"Cohort": split(k.replace("_", "_", 1).replace("abide1_", "abide1|").replace("abide2_", "abide2|").replace("|", "_"))[0] if False else COH[k.split("_")[0]],
          "Group": k.split("_")[1], "n": v["n"], "Raw strength": v["raw"], "Site means removed": v["rawc_site_centered"], "Deviation z": v["zrc"]}
         for k, v in raw.items()]))

    b22 = load("B22_s5_site_pairs.json")["results"]
    rows = []
    for c, d in b22.items():
        for lab, nm in [("all_pairs", "All pairs"), ("same_site_pairs", "Same-site pairs"), ("between_site_pairs", "Between-site pairs")]:
            v = d[lab]
            rows.append({"Cohort": COH[c], "Pairs": nm, "ASD": v["ASD"], "TC": v["TC"], "ASD − TC": v["ASD_minus_TC"], "P (within-site permutation)": v["perm_p_two_sided"]})
    tabs["S3_site_pairs"] = ("Table S4. Mean pairwise Jaccard similarity of extreme deviations by site pairing. ‖ Full-length data, |z| > 2; P from 10,000 permutations of diagnosis within site.", pd.DataFrame(rows))

    b25, b26 = load("B25_reviewer_analyses.json"), load("B26_relbase_empirical_marginal.json")
    rows = []
    for thr, row in b25["bc_chance_and_threshold"].items():
        for k, v in row.items():
            if k == "baseline_s0_s1":
                continue
            c, g = split(k)
            e = b26["threshold_sensitivity"][thr][k]
            rows.append({"Threshold": thr.replace("thr=", "|z| > "), "Cohort": c, "Group": g, "Extreme rate": v["extreme_rate_half1"],
                         "P2 observed": v["p2_observed"], "P2 chance": v["p2_chance_mean"], "Ratio": v["ratio_to_chance"],
                         "Between-site Jaccard": v["jaccard_between_site"],
                         "Gaussian baseline (s = 0)": v["null_mean"], "P vs Gaussian": v["p_one_sided_vs_s0"],
                         "Empirical baseline (s = 0)": e["null_mean"], "P vs empirical": e["p_one_sided_vs_s0"]})
    tabs["S4_thresholds"] = ("Table S5. Sensitivity of positive agreement and between-site overlap to the extreme-deviation threshold. ‖ P2, positive agreement, with its chance level "
                             "from halves permuted among participants within site; between-site overlap compared with the fully individual baseline under the Gaussian "
                             "and the empirical-marginal model (one-sided Monte Carlo P).",
                             pd.DataFrame(rows))

    st = b25["d_age"]["tertile_median_icc"]
    tabs["S5_age"] = ("Table S6. Median split-half ICC by age tertile. ‖ Tertile cut points {} and {} years; 95% CI from 2,000 bootstrap resamples, mean of bounds over four seeds. "
                      "In a linear model of split-half map correlation, age β = {} per year (95% CI {}–{}), P {}; age × group interaction P {}.".format(
                          *b25["d_age"]["age_tertile_cuts"], b25["d_age"]["beta_age_per_year"], *b25["d_age"]["ci95"], fmt_p(b25["d_age"]["p_age"]),
                          fmt_p(b25["d_age"]["p_age_x_group"])),
                      pd.DataFrame([{"Tertile": k.split("_")[0], "Group": k.split("_")[1], "n": v["n"], "Age range, y": f"{v['age_range'][0]}–{v['age_range'][1]}",
                                     "Median ICC": v["median_icc"], "95% CI": f"{v['ci95'][0]}–{v['ci95'][1]}"} for k, v in st.items()]))

    a = b25["a_sb_validation_between_run"]
    tabs["S6_sb"] = ("Table S7. Check of the Spearman–Brown assumption in the between-run subsample. ‖ Parcel ICCs of halves of the full series projected to the run length.",
                     pd.DataFrame([{"Group": g, "n": v["n"], "Half length, s": v["median_sec_half"], "Run length, s": v["median_sec_run"],
                                    "Median ICC, halves": v["median_icc_half"], "Predicted, runs": v["predicted_median_icc_run"],
                                    "Observed, runs": v["observed_median_icc_run"], "Observed − predicted (95% CI)":
                                    f"{v['observed_minus_predicted']} ({v['ci95_diff'][0]} to {v['ci95_diff'][1]})"} for g, v in a.items()]))

    rows = []
    for k, v in b25["e_detectable_shared_fraction"].items():
        c, g = split(k)
        rows.append({"Model": "Gaussian", "Cohort": c, "n": v["n"], **{f"s = {s}": p for s, p in v["power_by_s"].items() if float(s) <= 0.12},
                     "s at 80% power": v["min_detectable_s_power80"]})
    for k, v in b26["primary_threshold"].items():
        c, g = split(k)
        rows.append({"Model": "Empirical marginal", "Cohort": c, "n": v["n"], **{f"s = {s}": p for s, p in v["power_by_s"].items() if float(s) <= 0.12},
                     "s at 80% power": v["min_detectable_s_power80"]})
    tabs["S7_power"] = ("Table S8. Power of the overlap test by true shared fraction s. ‖ ASD groups, |z| > 2, one-sided α = .05.", pd.DataFrame(rows))

    b21 = load("B21_relbase.json")
    cal = b26["calibration"]
    tabs["S8_relbase"] = ("Table S9. Validation of the relbase package.", pd.DataFrame([
        {"Check": "Reproduction of the original simulation (same inputs and seeds)", "Result": f"maximum absolute difference {b21['validation_reproduce_B20']['max_abs_diff']}"},
        {"Check": "ICC(2,1) against the primary analysis (379 parcels × 4 groups)", "Result": f"maximum absolute difference {max(v['max_abs_diff'] for v in b21['validation_icc_vs_B14'].values()):.1e}"},
        {"Check": "Empirical-marginal model: median split-half ICC, simulated vs observed (TC)", "Result": f"{cal['mapped_icc_calibrated']} vs {cal['observed_tc_half_median_icc']} (reliability scale factor {cal['reliability_scale_factor']})"},
        {"Check": "Empirical-marginal model: P2 positive agreement, simulated vs observed", "Result": f"{cal['p2_simulated']} vs {cal['p2_observed']}"},
        {"Check": "Unit tests", "Result": "22 tests passed (python -m unittest discover -s tests)"}]))

    b28, b28b, b28c = load("B28_edge_level.json"), load("B28b_edge_fixed_count.json"), load("B28c_edge_symmetric_scoring.json")
    rows = []
    for k, v in b28["by_group"].items():
        c, g = split(k)
        rows.append({"Cohort": c, "Group": g, "n": v["n"], "Median edge ICC (95% CI)": f"{v['a_median_icc']} ({v['a_ci95'][0]}–{v['a_ci95'][1]})",
                     "Edges ICC ≥ 0.5": v["a_share_icc_ge_0.5"], "P2": v["b_p2"], "P2 chance": v["b_p2_chance"],
                     "Max same-sign prevalence": v["c_max_prevalence_full"], "Extreme rate": v["extreme_rate_full"],
                     "ICC, counts by parcel": v["d_region_count_median_icc"], "ICC, counts by network": v["d_network_count_median_icc"]})
    tabs["S10_edges"] = ("Table S10. Reliability of edge-level deviations. ‖ 71,631 edges; linear normative model. 95% CI of the median ICC from 200 bootstrap resamples per seed, mean of bounds over four seeds.", pd.DataFrame(rows))
    rows = []
    for c in ["abide1", "abide2"]:
        j = b28["e_between_site_jaccard_ASD_TC"][c]
        rows.append({"Cohort": COH[c], "Comparison": "Edges, |z| > 2", "ASD": f"{j['between_site'][0]:.5f}", "TC": f"{j['between_site'][1]:.5f}",
                     "ASD − TC": f"{j['between_site'][0] - j['between_site'][1]:.5f}", "P": f"{j['between_site_perm_p']:.4f}"})
        for lab, src in [("Edges, fixed count", b28b[c]["edges"]), ("Edges, fixed count, symmetric scoring", b28c[c]), ("Parcels, fixed count", b28b[c]["parcels"])]:
            rows.append({"Cohort": COH[c], "Comparison": lab, "ASD": f"{src['between_site_ASD']:.5f}", "TC": f"{src['between_site_TC']:.5f}", "ASD − TC": f"{src['diff']:.5f}", "P": f"{src['perm_p']:.4f}"})
    tabs["S11_overlap_counts"] = ("Table S11. Between-site overlap of extreme deviations with a threshold and with a fixed number of extremes per participant. ‖ Mean Jaccard similarity; "
                                  f"fixed number, {b28b['abide1']['edges']['k']:,} edges or {b28b['abide1']['parcels']['k']} parcels per participant; P from 10,000 permutations of diagnosis within site.", pd.DataFrame(rows))

    c03 = load("C03_corr_reliability.json")
    rows = []
    for lab, key in [("All, deviation z", "deviation_z"), ("All, raw FC strength", "raw_strength"),
                     ("Interval ≤ 60 days, deviation z", "interval_le_60d"), ("Interval > 60 days, deviation z", "interval_gt_60d")]:
        v = c03[key]
        rows.append({"Sample": lab, "n": v["n"], "Within session (halves)": v["within_mean"],
                     "Between sessions, equal length": v["between_equal_length"], "Between sessions, full length": v["between_full_length"],
                     "Difference (95% CI)": f"{v['between_minus_within']} ({v['ci95_between_minus_within'][0]} to {v['ci95_between_minus_within'][1]})",
                     "P2 within": v.get("p2_within", "—"), "P2 between": v.get("p2_between_equal_length", "—")})
    ng = load("C03_corr_reliability_nogsr.json")["deviation_z"]
    rows.append({"Sample": "All, deviation z, no global signal regression", "n": ng["n"], "Within session (halves)": ng["within_mean"],
                 "Between sessions, equal length": ng["between_equal_length"], "Between sessions, full length": ng["between_full_length"],
                 "Difference (95% CI)": f"{ng['between_minus_within']} ({ng['ci95_between_minus_within'][0]} to {ng['ci95_between_minus_within'][1]})",
                 "P2 within": ng["p2_within"], "P2 between": ng["p2_between_equal_length"]})
    n = c03["n"]
    tabs["S12_corr_retest"] = ("Table S12. Within- and between-session reliability in the CoRR test–retest sample. ‖ Median ICC of 377 parcel FC strength measures within a session "
                                "(two halves, mean of both sessions) "
                               f"and between sessions. Reference sample {n['reference_ses1']} first sessions from {n['sites_reference']} sites; "
                               f"retest interval median {n['retest_days_median_iqr'][0]} days (IQR {n['retest_days_median_iqr'][1]}–{n['retest_days_median_iqr'][2]}). "
                               "Difference = between (equal length) − within; participant-level bootstrap, 1,000 resamples per seed, mean of bounds over four seeds. "
                               "Interval strata differ in site composition and are descriptive.", pd.DataFrame(rows))

    b29 = load("B29_icc31_fingerprint.json")["results"]
    rows = []
    for fs, flab in [("strength_half", "Parcel FC strength, halves"), ("strength_run", "Parcel FC strength, runs"), ("netfc_half", "Network FC, halves"), ("edges_half", "Edges, halves")]:
        for u, v in b29[fs].items():
            rows.append({"Features": flab, "Sample": u, "n": v["n"], "Median ICC(2,1)": v.get("median_icc21", "—"), "Median ICC(3,1)": v.get("median_icc31", "—"),
                         "Median map r, self": v["median_r_self"], "Median map r, others": v["median_r_other"],
                         "Identification": v["identification"], "Chance": v["chance"], "Discriminability": v["discriminability"]})
    tabs["S13_multivariate"] = ("Table S13. Consistency ICC(3,1) and multivariate reliability of deviation maps. ‖ Identification: share of participants whose first-segment map "
                                "correlated best with their own second-segment map among all participants in the sample (mean of both directions); "
                                "discriminability (chance 0.5) as defined by Bridgeford et al. [@bridgeford2021].", pd.DataFrame(rows))
    b30 = load("B30_relbase_identification_subtypes.json")
    rows = []
    for u, v in b30["identification"].items():
        for s_ in ["s=0.0", "s=0.5"]:
            rows.append({"Sample": u, "n": v["n"], "Simulation": s_.replace("s=", "s = "),
                         "Identification, simulated (95% range)": f"{v[s_]['identification_mean']:.4f} ({v[s_]['identification_q025_q975'][0]:.4f}–{v[s_]['identification_q025_q975'][1]:.4f})",
                         "Identification, observed": v["observed_identification"],
                         "Discriminability, simulated (95% range)": f"{v[s_]['discriminability_mean']:.4f} ({v[s_]['discriminability_q025_q975'][0]:.4f}–{v[s_]['discriminability_q025_q975'][1]:.4f})",
                         "Discriminability, observed": v["observed_discriminability"]})
    tabs["S14a_identification_sim"] = ("Table S14a. Identification expected from parcel reliability alone. ‖ Halves simulated with relbase at the observed TC split-half reliability of each parcel "
                                       "and the observed distribution of deviation scores (40 replicates).", pd.DataFrame(rows))
    rows = []
    for g, v in b30["subtypes"].items():
        for k in ["K=1", "K=2", "K=3", "K=5"]:
            r = {"Group": g.replace("ASD_abide1", "ABIDE I ASD").replace("ASD_abide2", "ABIDE II ASD"), "n": v["n"], "Subtypes": k.replace("K=", "")}
            for s_, w in v[k].items():
                if s_ != "min_s_power80":
                    r[f"Power, s = {s_}"] = w["power"]
            r["Smallest s with 80% power"] = v[k]["min_s_power80"] if v[k]["min_s_power80"] is not None else "> 1"
            rows.append(r)
    tabs["S14b_subtypes"] = ("Table S14b. Power of the between-person overlap test when the group consists of K equal-sized subtypes. ‖ One-sided test at α = .05 against fully "
                             "individual true deviations; each subtype has its own shared pattern, and s is the fraction of true variance shared within a subtype. Full-length reliability and the group's observed distribution of deviation scores; "
                             "200 replicates per cell, null from 1,000 replicates.", pd.DataFrame(rows))

    for k, (cap, df) in tabs.items():
        df.to_csv(T / f"{k}.csv", index=False, encoding="utf-8")
        title, _, note = cap.partition(" ‖ ")
        note = f"Note. {note} Other abbreviations as in the main article." if note else "Note. Abbreviations as in the main article."
        out.append(f"{title}\n\n{md(df)}\n\n{note}\n")
    head = "<!-- 由 scripts/B27_supplementary_B1.py 生成，勿手改；重跑脚本更新。 -->\n\n# Supplementary tables\n\n"
    (ROOT / "manuscript_B1/B1_supplementary_tables_v0.1.md").write_text(head + "\n".join(out), encoding="utf-8")
    print("tables:", len(tabs))


if __name__ == "__main__":
    main()
