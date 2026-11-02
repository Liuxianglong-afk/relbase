# relbase

Reliability-matched baselines for the between-person overlap of extreme normative deviations.

Studies of heterogeneity often report that extreme deviations from a normative model rarely coincide
between individuals. How much overlap to expect depends on how reliable each person's deviation map is:
at low reliability, overlap stays low even when all individuals share the same true deviations.
`relbase` simulates deviation maps at a given reliability and returns the overlap expected for each
fraction of shared true deviation, so that an observed overlap can be read against that baseline.

## Model

For P features per person:

    true deviation      T_i = sqrt(s) * C + sqrt(1 - s) * U_i
    observed deviation  z_i = k * [sqrt(r) * T_i + sqrt(1 - r) * E_i]

C is a pattern shared by the group, U_i an individual pattern, E_i measurement noise, s the shared
fraction, r the reliability (one value or one per feature) and k the SD of observed z. C and U are drawn
with the spatial correlation of deviation maps (`R_signal`), E with the spatial correlation of the
difference between two measurements (`R_noise`); either may be omitted (independent features).

Deviation scores often have heavier tails than a Gaussian. With `marginal=` (observed deviation scores of
the group, or `marginal_quantiles` of them), the Gaussian values are mapped onto that empirical distribution
(a Gaussian copula), so simulated extremes occur at the observed rate at every threshold.

Subtypes: with `n_subtypes=k`, participants are split into k equal-sized subtypes, each with its own shared
pattern, and s is the fraction of true variance shared within a subtype (`simulate`, `overlap_distribution`,
`baseline_curve`; k = 1 is the model above).

Repeated measurements: `simulate_pair` returns two independent measurements of the same simulated group,
for example to compare `identification_rate` (fingerprinting) or `discriminability` of observed maps with
the values expected from univariate reliability alone.

Overlap metrics (extremes are |z| > threshold, with sign):

| metric | definition |
|---|---|
| `jaccard` | mean pairwise Jaccard similarity of extreme sets; `blocks=` keeps only pairs from different blocks (e.g. sites) |
| `max_prevalence` | largest proportion of participants with the same extreme in one feature |
| `mean_prevalence` | proportion of participants with an extreme, averaged over features |

## Use

Requires Python ≥ 3.9 and NumPy. Install the archived release or the current version from GitHub:

    pip install git+https://github.com/Liuxianglong-afk/relbase.git@v0.1.2
    pip install git+https://github.com/Liuxianglong-afk/relbase.git

or, from a local copy of this folder, `pip install .`.

```python
import numpy as np
from relbase import BaselineModel, estimate_inputs, implied_shared_fraction, mean_jaccard

# 1. inputs from a reference group with full-length and split-half deviation scores (participants x features)
inp = estimate_inputs(z_full_ref, z_half1_ref, z_half2_ref)

# 2. baseline for the group of interest
model = BaselineModel(inp["R_signal"], inp["R_noise"], scale=z_group.std())
curve = model.baseline_curve(n=len(z_group), reliability=np.clip(inp["reliability_full"], 0, 1))

# 3. read the observed overlap against it
obs = mean_jaccard(z_group, blocks=site_labels)
print(implied_shared_fraction(obs, curve))
print(model.test_heterogeneous_null(obs, len(z_group), np.clip(inp["reliability_full"], 0, 1)))
```

Command line (CSV: one row per participant, one column per feature, header row):

    python -m relbase estimate --full z_full.csv --half1 z_h1.csv --half2 z_h2.csv --out inputs.npz
    python -m relbase curve --inputs inputs.npz --n 200 --per-feature --out curve.csv
    python -m relbase infer --inputs inputs.npz --observed-z z_group.csv --per-feature --blocks-column site

`examples/abide_glasser379_tc_inputs.npz` holds inputs estimated from typically developing participants of
ABIDE I and II (FC strength of 360 cortical and 19 subcortical parcels, DCAN ABCD-HCP pipeline; group-level
correlation matrices and per-feature reliability only).

## Notes

- Reliability should refer to the data length used to define extremes (use `spearman_brown` to rescale
  split-half reliability to full length).
- When deviations were adjusted with site-level estimates, errors in those estimates are shared within a
  site and change within-site overlap; compare groups on between-site pairs (`blocks=`).
- With the default Gaussian marginal, heavy tails make the baseline underestimate coincidences of strict
  extremes; pass `marginal=` (CLI: `--empirical-marginal`) and check that simulated and observed extreme
  rates agree. Reliability then refers to the latent Gaussian scale; check that the median ICC of two
  simulated measurements matches the observed one and rescale reliability if it does not.
- `test_heterogeneous_null` asks whether overlap exceeds that expected from fully individual deviations;
  `implied_shared_fraction` inverts the baseline curve (interval from the 2.5–97.5% simulation bands).
- Identification of participants from whole maps can be high even when every feature is unreliable,
  because it pools many weak features; compare it with `simulate_pair` at the observed reliability.

## Analysis code of the accompanying study

`analysis/` holds the scripts, analysis plan and group-level results of the ABIDE study that introduced
relbase; see `analysis/README.md`.

## Tests

    python -m unittest discover -s tests

## Citation

If you use relbase, please cite the archived software release and the article that introduced it:

- Software: Liu X. relbase (version 0.1.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23891148
- Article: Liu X, et al. Low reliability of individual normative deviations in resting-state functional connectivity limits inferences about heterogeneity in autism. <Journal>, <year>. <DOI>

`CITATION.cff` holds the same information in machine-readable form (GitHub shows it as "Cite this repository").

## Licence

MIT; see `LICENSE`. The example inputs in `examples/` are derived from ABIDE I and ABIDE II
(International Neuroimaging Data-sharing Initiative); users of these data should acknowledge ABIDE and
follow its data usage terms.
