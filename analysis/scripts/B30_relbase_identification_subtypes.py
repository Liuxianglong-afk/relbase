"""课题 B1 事后分析（2026-11-02，双专家视角内评补做 ③④）：relbase 模拟的被试识别率与亚型混合情景。

a. 识别率：在 TC 实测分区信度（B26 校准系数 1.0，即未缩放）与实测边缘分布下，模拟同一组被试的两半偏离图（s = 0），
   计算识别率与 discriminability，与 B29 实测比较——检验“单变量信度低”是否足以解释实测的多变量识别率。
b. 亚型：K 个等大亚型，各有自己的共享模式，亚型内共享比例 s；在 ASD 两队列的样本量、实测边缘分布与全长信度下，
   计算平均 Jaccard 与相对 s = 0 零分布（1,000 次，第 95 百分位）的检出能力（每格 200 次）；K = 1 即 B26 的原模型。
输入: derivatives/norms_B1/B13_zrc_strength_{full,half1,half2}.csv.gz；results/B26、B29
输出: results/B30_relbase_identification_subtypes.json
用法: .venv_B/Scripts/python.exe -I scripts/B30_relbase_identification_subtypes.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "relbase"))
from relbase import (BaselineModel, discriminability, estimate_inputs, identification_rate, marginal_quantiles,  # noqa: E402
                     spearman_brown)

NB = ROOT / "derivatives/norms_B1"
IDX = ["cohort", "sub_n", "site", "dx"]
sys.stdout.reconfigure(encoding="utf-8")


def wide(cond):
    z = pd.read_csv(NB / f"B13_zrc_strength_{cond}.csv.gz")
    z = z[z.site_ok]
    return z.pivot_table(index=IDX, columns="feature", values="zrc").dropna()


def main():
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/B30_relbase_identification_subtypes.py"}
    full, h1, h2 = wide("full"), wide("half1"), wide("half2")
    idx = full.index.intersection(h1.index).intersection(h2.index)
    full, h1, h2 = full.loc[idx], h1.loc[idx], h2.loc[idx]
    dx, coh = idx.get_level_values("dx").to_numpy(), idx.get_level_values("cohort").to_numpy()
    tc = dx == 2
    est = estimate_inputs(full.values[tc], h1.values[tc], h2.values[tc])
    factor = json.loads((ROOT / "results/B26_relbase_empirical_marginal.json").read_text(encoding="utf-8"))["calibration"]["reliability_scale_factor"]
    rel_half = np.clip(est["reliability_half"] * factor, 0, 1)
    rel_full = spearman_brown(rel_half, 2)
    b29 = json.loads((ROOT / "results/B29_icc31_fingerprint.json").read_text(encoding="utf-8"))["results"]["strength_half"]

    # a. 识别率
    m_half = BaselineModel(est["R_signal"], est["R_noise"], marginal=marginal_quantiles(np.vstack([h1.values, h2.values])))
    ident = {}
    for lab, n in [("ABIDE I TC", 278), ("ABIDE I ASD", 201), ("ABIDE II TC", 219), ("ABIDE II ASD", 135)]:
        row = {"n": n, "observed_identification": b29[lab]["identification"], "observed_discriminability": b29[lab]["discriminability"]}
        for s in [0.0, 0.5]:
            ids, dis = [], []
            for sd in [42, 7, 123, 2024]:
                rng = np.random.default_rng(sd)
                for _ in range(10):
                    a, b = m_half.simulate_pair(rng, n, rel_half, s)
                    ids.append(identification_rate(a, b))
                    dis.append(discriminability(a, b))
            row[f"s={s}"] = {"identification_mean": round(float(np.mean(ids)), 4), "identification_q025_q975": [round(float(x), 4) for x in np.quantile(ids, [0.025, 0.975])],
                             "discriminability_mean": round(float(np.mean(dis)), 4), "discriminability_q025_q975": [round(float(x), 4) for x in np.quantile(dis, [0.025, 0.975])]}
        ident[lab] = row
        print("ident", lab, row, flush=True)
    out["identification"] = ident

    # b. 亚型
    sub = {}
    for lab, c in [("ASD_abide1", "abide1"), ("ASD_abide2", "abide2")]:
        m = (dx == 1) & (coh == c)
        n = int(m.sum())
        gm = BaselineModel(est["R_signal"], est["R_noise"], marginal=marginal_quantiles(full.values[m]))
        null = gm.overlap_distribution(n, rel_full, 0.0, reps=1000, seed=42)
        q95 = float(np.quantile(null, 0.95))
        res = {"n": n, "null_mean": round(float(null.mean()), 5), "null_q95": round(q95, 5)}
        for k in [1, 2, 3, 5]:
            rk = {}
            for s in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
                alt = gm.overlap_distribution(n, rel_full, s, reps=200, seed=1000 + int(s * 100) + 10 * k, n_subtypes=k)
                rk[str(s)] = {"mean_jaccard": round(float(alt.mean()), 5), "power": round(float((alt > q95).mean()), 3)}
            ss = np.array([float(x) for x in rk])
            pw = np.array([v["power"] for v in rk.values()])
            rk["min_s_power80"] = round(float(np.interp(0.8, pw, ss)), 3) if pw.max() >= 0.8 and np.all(np.diff(pw) >= 0) else (
                None if pw.max() < 0.8 else round(float(ss[np.argmax(pw >= 0.8)]), 3))
            res[f"K={k}"] = rk
            print("subtypes", lab, k, {s: v["power"] for s, v in rk.items() if s != "min_s_power80"}, rk["min_s_power80"], flush=True)
        sub[lab] = res
    out["subtypes"] = sub
    (ROOT / "results/B30_relbase_identification_subtypes.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
