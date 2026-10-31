"""B14 icc21 的独立验证：(1) 与逐列朴素 ANOVA 实现比较；(2) 已知真值模拟。输出 results/B14v_verify_icc.json"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b14", ROOT / "scripts/B14_reliability.py"); b14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b14)

def naive(x):  # x: n×k 单列，Shrout & Fleiss ICC(2,1)
    n, k = x.shape; gm = x.mean()
    msr = k * sum((x[i].mean() - gm) ** 2 for i in range(n)) / (n - 1)
    msc = n * sum((x[:, j].mean() - gm) ** 2 for j in range(k)) / (k - 1)
    sse = sum((x[i, j] - x[i].mean() - x[:, j].mean() + gm) ** 2 for i in range(n) for j in range(k))
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

rng = np.random.default_rng(42)
a = rng.normal(size=(60, 7)); b = a * 0.6 + rng.normal(size=(60, 7)) + 0.3
v = b14.icc21(a, b); nv = np.array([naive(np.c_[a[:, j], b[:, j]]) for j in range(7)])
res = {"max_abs_diff_vs_naive": float(np.abs(v - nv).max())}
for err_var, truth in [(1.0, 0.5), (1 / 3, 0.75)]:
    t = rng.normal(size=(20000, 1)); e = np.sqrt(err_var)
    res[f"sim_true_{truth}"] = round(float(b14.icc21(t + rng.normal(0, e, (20000, 1)), t + rng.normal(0, e, (20000, 1)))[0]), 4)
ag = b14.extreme_agreement(np.array([[3., 0, -3]]), np.array([[2.5, 0, 3]]))
res["extreme_toy"] = ag
res["pass"] = res["max_abs_diff_vs_naive"] < 1e-10 and abs(res["sim_true_0.5"] - 0.5) < 0.02 and abs(res["sim_true_0.75"] - 0.75) < 0.02 and ag["n_both_same_sign"] == 1
(ROOT / "results/B14v_verify_icc.json").write_text(json.dumps(res, indent=2), encoding="utf-8"); print(json.dumps(res, indent=2))
