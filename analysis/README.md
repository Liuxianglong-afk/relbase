# Analysis code for the ABIDE reliability study

This folder holds the code, analysis plan and group-level results of the study that introduced relbase
("Low reliability of individual normative deviations in resting-state functional connectivity limits
inferences about heterogeneity in autism"). It is a snapshot of the project at the release that contains it;
`MANIFEST.tsv` lists every file with its SHA-256.

## Contents

| folder | content |
|---|---|
| `scripts/` | Python and R scripts, numbered in the order they are run |
| `protocol/` | shared preprocessing plan, the B1 analysis plan with its change log (section 12), and the frozen v1.0 text (SHA-256 ca187995…) |
| `results/` | group-level results (JSON) behind every number in the article |
| `tables/` | Tables 1–3 and supplementary Tables S1–S14 (CSV) |

No participant-level data are included. Participant identifiers that appeared as examples in a probe
script and in the shared plan were replaced with `<ID>` in this copy.

## Data

ABIDE I and II phenotypes and DCAN ABCD-HCP derivatives are public in the FCP-INDI Amazon S3 bucket
(`s3://fcp-indi/data/Projects/ABIDE/` and `s3://fcp-indi/data/Projects/ABIDE2/`); see
https://fcon_1000.projects.nitrc.org/indi/abide/ for the data usage terms. CoRR test–retest data preprocessed with
fMRIPrep are in the same bucket (`s3://fcp-indi/data/Projects/CORR/Outputs/fmriprep/`), with phenotypes at
`s3://fcp-indi/data/Projects/CORR/RawData/CoRR_AggregatedPhenotypicData.csv`; the HCP-MMP1.0 annotation on fsaverage
is at https://doi.org/10.6084/m9.figshare.3498446.v2.

## Running

The scripts expect the original project layout: this folder's `scripts/` and `results/` next to `data/`
(phenotypes), `derivatives/` (created by the scripts) and `relbase/` (the package at the repository root).
Order:

| step | scripts | purpose |
|---|---|---|
| 0 | B00a–B00d, B01, B02, B02v, B03 | locate and download DCAN derivatives, check run boundaries, motion QC and FC, parcel table |
| 1 | B11, B11s, B11c | features (full length, halves, runs) and sensitivity variants (FD 0.3, odd/even, Gordon, ComBat-GAM) |
| 2 | B12 (R, gamlss) | normative models; `Rscript B12_gamlss_norms_B1.R <cores> <max features per set, 0 = all> [variant]` |
| 3 | B13, B13v | site recentring and calibration in typically developing participants |
| 4 | B14, B14v, B14v2 | primary endpoints P1/P2 and checks |
| 5 | B15, B17 | secondary endpoints S2/S4/S5, sensitivity analyses and S6 |
| 6 | B16, B18, B19, B20f | figures and tables |
| 7 | B20, B21, B22 | post hoc noise simulation, relbase validation and application, same- vs between-site overlap |
| 8 | B25, B26, B28, B28b, B28c, B29, B30 | additional checks, empirical-marginal baseline, edge-level analyses, ICC(3,1) and multivariate reliability, subtype baseline |
| 9 | C00–C03 | CoRR inventory, download, features and within- vs between-session reliability (C02 --no-gsr, C03 --variant nogsr: without global signal regression) |
| 10 | B27, B24 | supplementary tables; check of manuscript numbers against result files |

Python scripts were run with `python -I` (Python 3.12); package versions are listed in the Methods.
Seeds: 42 (primary), 7, 123 and 2024.
