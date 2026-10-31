"""Command-line interface.

    python -m relbase estimate --full z_full.csv --half1 z_h1.csv --half2 z_h2.csv --out inputs.npz
    python -m relbase curve    --inputs inputs.npz --n 201 [--reliability 0.35 | --per-feature] --out curve.csv
    python -m relbase infer    --inputs inputs.npz --observed-z z_group.csv [--reliability 0.35 | --per-feature]

CSV files hold one row per participant and one column per feature (a header row is expected;
non-numeric columns are dropped). Participant identifiers are never written to any output.
"""
import argparse
import csv
import json
import sys

import numpy as np

from .core import DEFAULT_S_GRID, METRICS, BaselineModel, estimate_inputs, implied_shared_fraction


def read_matrix(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    keep = []
    for j in range(len(header)):
        try:
            [float(r[j]) for r in body]
            keep.append(j)
        except ValueError:
            continue
    return np.array([[float(r[j]) for j in keep] for r in body]), [header[j] for j in keep]


def read_column(path, name):
    with open(path, newline="", encoding="utf-8") as f:
        return [row[name] for row in csv.DictReader(f)]


def load_model(args):
    d = np.load(args.inputs)
    rel = d["reliability_full"] if args.per_feature else args.reliability
    if rel is None:
        sys.exit("give --reliability or --per-feature")
    if np.ndim(rel):
        rel = np.clip(rel, 0, 1)
    marginal = None
    if args.empirical_marginal:
        if "marginal_quantiles" not in d.files:
            sys.exit("--empirical-marginal needs inputs written by this version of `relbase estimate`")
        marginal = d["marginal_quantiles"]
    model = BaselineModel(d["R_signal"], d["R_noise"], scale=args.scale if args.scale else float(d["scale"]),
                          threshold=args.threshold, marginal=marginal)
    return model, rel


def main(argv=None):
    ap = argparse.ArgumentParser(prog="relbase")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="estimate model inputs from a reference group")
    e.add_argument("--full", required=True)
    e.add_argument("--half1", required=True)
    e.add_argument("--half2", required=True)
    e.add_argument("--out", required=True)
    for name in ("curve", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", required=True, help="npz written by `relbase estimate`")
        p.add_argument("--reliability", type=float, help="one reliability for all features")
        p.add_argument("--per-feature", action="store_true", help="use per-feature full-length reliability from --inputs")
        p.add_argument("--metric", choices=sorted(METRICS), default="jaccard")
        p.add_argument("--threshold", type=float, default=2.0)
        p.add_argument("--scale", type=float, help="SD of observed z (default: from --inputs)")
        p.add_argument("--empirical-marginal", action="store_true",
                       help="map simulated values onto the reference marginal distribution (observed tails); --scale is then ignored")
        p.add_argument("--reps", type=int, default=25, help="replicates per seed")
        p.add_argument("--s-grid", type=float, nargs="+", default=list(DEFAULT_S_GRID))
    sub.choices["curve"].add_argument("--n", type=int, required=True)
    sub.choices["curve"].add_argument("--out")
    sub.choices["infer"].add_argument("--observed-z", required=True, help="CSV of the group whose overlap is interpreted")
    sub.choices["infer"].add_argument("--null-reps", type=int, default=1000)
    sub.choices["infer"].add_argument("--blocks-column", help="column of --observed-z (e.g. site): Jaccard uses only pairs from different blocks")
    args = ap.parse_args(argv)

    if args.cmd == "estimate":
        f, cols = read_matrix(args.full)
        h1, c1 = read_matrix(args.half1)
        h2, c2 = read_matrix(args.half2)
        if not (cols == c1 == c2) or not (len(f) == len(h1) == len(h2)):
            sys.exit("the three files must have the same participants (rows, same order) and features (columns)")
        d = estimate_inputs(f, h1, h2)
        np.savez_compressed(args.out, features=np.array(cols), **d)
        print(json.dumps({"n": len(f), "features": len(cols), "scale": round(d["scale"], 4),
                          "median_reliability_half": round(float(np.median(d["reliability_half"])), 4),
                          "median_reliability_full": round(float(np.median(d["reliability_full"])), 4)}, indent=1))
        return

    model, rel = load_model(args)
    if args.cmd == "curve":
        curve = model.baseline_curve(args.n, rel, args.s_grid, args.metric, args.reps)
        if args.out:
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(curve[0]))
                w.writeheader()
                w.writerows(curve)
        print(json.dumps(curve, indent=1))
        return

    z, _ = read_matrix(args.observed_z)
    if z.shape[1] != model.P:
        sys.exit(f"observed z has {z.shape[1]} features; model has {model.P}")
    if args.blocks_column:
        if args.metric != "jaccard":
            sys.exit("--blocks-column applies only to --metric jaccard")
        obs = METRICS[args.metric](z, args.threshold, blocks=read_column(args.observed_z, args.blocks_column))
    else:
        obs = METRICS[args.metric](z, args.threshold)
    curve = model.baseline_curve(len(z), rel, args.s_grid, args.metric, args.reps)
    print(json.dumps({"n": len(z), "metric": args.metric, "observed": obs, "curve": curve,
                      "implied_shared_fraction": implied_shared_fraction(obs, curve),
                      "test_vs_fully_individual": model.test_heterogeneous_null(obs, len(z), rel, args.metric, args.null_reps)}, indent=1))


if __name__ == "__main__":
    main()
