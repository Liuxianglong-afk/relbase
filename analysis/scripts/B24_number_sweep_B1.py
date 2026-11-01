"""课题 B1 手稿数字核查：手稿中每个数字是否能在结果文件（results/B*.json、C0*.json、tables_B1/*.csv）中按同等小数位找到来源。

规则: 小数按其小数位数四舍五入比对（百分数同时比对 /100 的值）；≥ 3 位整数直接比对；
      找不到来源的数字列出，供人工逐条判断（可能是推导值、文献值、方法参数或错误）。
输出: logs/B24_number_sweep_<日期>.txt（由调用方重定向）
用法: .venv_B/Scripts/python.exe -I scripts/B24_number_sweep_B1.py
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
DOCS = ["B1_draft_v0.1_abstract_legends.md", "B1_draft_v0.1_intro_discussion.md", "B1_draft_v0.1_methods_results.md", "B1_supplementary_methods_v0.1.md", "B1_supplementary_results_v0.1.md", "B1_cover_letter_v0.1.md"]


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from leaves(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)
    elif isinstance(o, str):
        for m in re.finditer(r"-?\d+\.?\d*", o):
            yield float(m.group())


def pool():
    vals = []
    for f in [*(ROOT / "results").glob("B*.json"), *(ROOT / "results").glob("C0*.json")]:
        vals += list(leaves(json.loads(f.read_text(encoding="utf-8"))))
    for f in (ROOT / "tables_B1").glob("*.csv"):
        for s in pd.read_csv(f).astype(str).values.ravel():
            vals += [float(m.group()) for m in re.finditer(r"-?\d+\.?\d*", s)]
    return vals


def rnd(x, d):
    return Decimal(str(abs(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def main():
    vals = pool()
    keys = {}
    for v in vals:
        for d in range(0, 5):
            keys.setdefault(d, set()).add(rnd(v, d))
            keys[d].add(rnd(v * 100, d))
            keys[d].add(rnd(v / 60, d))  # 秒→分钟
    for doc in DOCS:
        text = (ROOT / "manuscript_B1" / doc).read_text(encoding="utf-8")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        miss = []
        for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})(?![\w])", text):
            s = m.group().replace(",", "")
            d = len(s.split(".")[1]) if "." in s else 0
            if rnd(float(s), d) not in keys[d]:
                ctx = text[max(0, m.start() - 50): m.end() + 15].replace("\n", " ")
                miss.append(f"  {m.group():>8}  …{ctx}…")
        print(f"== {doc}: {len(miss)} numbers without a source match")
        print("\n".join(miss))


if __name__ == "__main__":
    main()
