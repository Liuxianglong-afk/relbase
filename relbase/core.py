"""Reliability-matched baselines for the between-person overlap of extreme normative deviations.

Generative model (P features per person):
    true deviation      T_i = sqrt(s) * C + sqrt(1 - s) * U_i
    observed deviation  z_i = k * [sqrt(r) * T_i + sqrt(1 - r) * E_i]
C is a pattern shared by the whole group (drawn once per replicate), U_i an individual pattern,
E_i measurement noise; s is the shared fraction and r the feature reliability (scalar or one per feature).
C and U are drawn with the spatial correlation R_signal, E with R_noise (identity if not given).

With `marginal` (observed deviation scores, or their quantiles), the Gaussian values (k = 1) are mapped
through the normal CDF onto the empirical distribution, so that simulated extremes have the observed tail
frequencies while the spatial and test-retest dependence of the Gaussian model is kept (Gaussian copula).
"""
import math

import numpy as np

try:
    from scipy.special import ndtr as _ndtr
except ImportError:  # scipy is optional
    _ndtr = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))), otypes=[float])

DEFAULT_SEEDS = (42, 7, 123, 2024)
DEFAULT_S_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def extreme_indicator(z, threshold=2.0):
    """Participants x (2 * features) indicator of positive and negative extremes."""
    z = np.asarray(z, dtype=float)
    return np.hstack([z > threshold, z < -threshold]).astype(np.float64)


def mean_jaccard(z, threshold=2.0, blocks=None):
    """Mean pairwise Jaccard similarity of signed extreme sets; pairs with no extremes are skipped.

    `blocks` (one label per participant, e.g. site) restricts the mean to pairs from different blocks.
    Use it when deviations were adjusted with block-level estimates (site recentring, harmonization):
    errors in those estimates are shared within a block and change within-block overlap.
    """
    x = extreme_indicator(z, threshold)
    inter = x @ x.T
    s = x.sum(1)
    union = s[:, None] + s[None, :] - inter
    iu = np.triu_indices(len(x), 1)
    u, i = union[iu], inter[iu]
    ok = u > 0
    if blocks is not None:
        b = np.asarray(blocks)
        ok &= (b[:, None] != b[None, :])[iu]
    return float((i[ok] / u[ok]).mean()) if ok.any() else 0.0


def max_prevalence(z, threshold=2.0):
    """Largest proportion of participants sharing an extreme deviation of the same sign in one feature."""
    return float(extreme_indicator(z, threshold).mean(0).max())


def mean_prevalence(z, threshold=2.0):
    """Proportion of participants with an extreme deviation, averaged over features and signs."""
    return float(extreme_indicator(z, threshold).mean())


METRICS = {"jaccard": mean_jaccard, "max_prevalence": max_prevalence, "mean_prevalence": mean_prevalence}


def spearman_brown(r, factor):
    """Reliability after changing the amount of data by `factor` (2 = from half to full length)."""
    r = np.asarray(r, dtype=float)
    return factor * r / (1 + (factor - 1) * r)


def icc21(a, b):
    """ICC(2,1) per feature for two measurements (participants x features each)."""
    y = np.stack([np.asarray(a, float), np.asarray(b, float)], axis=1)  # n x 2 x P
    n, k = y.shape[0], 2
    gm = y.mean(axis=(0, 1))
    ss_rows = k * ((y.mean(1) - gm) ** 2).sum(0)
    ss_cols = n * ((y.mean(0) - gm) ** 2).sum(0)
    ss_tot = ((y - gm) ** 2).sum(axis=(0, 1))
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = (ss_tot - ss_rows - ss_cols) / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def estimate_inputs(z_full, z_half1, z_half2):
    """Model inputs from a reference group with full-length and split-half deviation scores.

    Returns R_signal (correlation of full-length z across features), R_noise (correlation of the
    half difference), scale (SD of full-length z), reliability_half (ICC(2,1) per feature) and
    reliability_full (Spearman-Brown to full length).
    """
    z_full, h1, h2 = (np.asarray(x, float) for x in (z_full, z_half1, z_half2))
    rel_half = icc21(h1, h2)
    return {
        "R_signal": np.corrcoef(z_full.T),
        "R_noise": np.corrcoef(((h1 - h2) / np.sqrt(2)).T),
        "scale": float(z_full.std()),
        "reliability_half": rel_half,
        "reliability_full": spearman_brown(np.clip(rel_half, 0, 1), 2),
        "marginal_quantiles": marginal_quantiles(z_full),
    }


def marginal_quantiles(z, n=2001):
    """Quantiles of pooled deviation scores at n evenly spaced probabilities (a compact, group-level marginal)."""
    z = np.asarray(z, dtype=float).ravel()
    return np.quantile(z[np.isfinite(z)], np.linspace(0, 1, n))


def _cholesky(R, jitter):
    R = np.asarray(R, dtype=float)
    return np.linalg.cholesky(R + jitter * np.eye(R.shape[0]))


class BaselineModel:
    """Simulates deviation maps under the model above and summarizes their between-person overlap."""

    def __init__(self, R_signal=None, R_noise=None, n_features=None, scale=1.0, threshold=2.0, jitter=1e-6, marginal=None):
        if R_signal is None and n_features is None:
            raise ValueError("give R_signal or n_features")
        self.P = int(np.asarray(R_signal).shape[0]) if R_signal is not None else int(n_features)
        self.L_signal = _cholesky(R_signal, jitter) if R_signal is not None else None
        self.L_noise = _cholesky(R_noise, jitter) if R_noise is not None else None
        for L in (self.L_signal, self.L_noise):
            if L is not None and L.shape[0] != self.P:
                raise ValueError("R_signal and R_noise must have the same size")
        self.scale = float(scale)
        self.threshold = float(threshold)
        self.marginal = None
        if marginal is not None:
            m = np.sort(np.asarray(marginal, dtype=float).ravel())
            self.marginal = m[np.isfinite(m)]
            if len(self.marginal) < 3:
                raise ValueError("marginal needs at least 3 values")

    def _to_marginal(self, x):
        """Map standard normal values onto the empirical marginal (linear interpolation between order statistics)."""
        m = self.marginal
        pos = _ndtr(x) * (len(m) - 1)
        return np.interp(pos, np.arange(len(m)), m)

    def _draw(self, rng, L, m):
        x = rng.standard_normal((m, self.P))
        return x if L is None else x @ L.T

    def _reliability(self, reliability):
        r = np.broadcast_to(np.asarray(reliability, dtype=float), (self.P,))
        if np.any((r < 0) | (r > 1)):
            raise ValueError("reliability must lie in [0, 1]")
        return r

    def _true(self, rng, n, shared_fraction, n_subtypes):
        """True deviations: each participant's subtype pattern (weight s) plus an individual pattern (weight 1 - s)."""
        s, k = float(shared_fraction), int(n_subtypes)
        if k < 1:
            raise ValueError("n_subtypes must be at least 1")
        C = self._draw(rng, self.L_signal, k)
        g = np.zeros(n, dtype=int) if k == 1 else rng.permutation(np.arange(n) % k)  # equal-sized subtypes
        return np.sqrt(s) * C[g] + np.sqrt(1 - s) * self._draw(rng, self.L_signal, n)

    def _measure(self, rng, T, r):
        latent = np.sqrt(r) * T + np.sqrt(1 - r) * self._draw(rng, self.L_noise, T.shape[0])
        if self.marginal is not None:
            return self._to_marginal(latent)
        return self.scale * latent

    def simulate(self, rng, n, reliability, shared_fraction, n_subtypes=1):
        """One simulated group: n participants x P features.

        With n_subtypes = k > 1, participants are split into k equal subtypes, each with its own shared pattern;
        shared_fraction is then the fraction of true variance shared within a subtype.
        """
        r = self._reliability(reliability)
        return self._measure(rng, self._true(rng, n, shared_fraction, n_subtypes), r)

    def simulate_pair(self, rng, n, reliability, shared_fraction=0.0, n_subtypes=1):
        """Two independent measurements (e.g. the halves of a scan) of the same simulated group."""
        r = self._reliability(reliability)
        T = self._true(rng, n, shared_fraction, n_subtypes)
        return self._measure(rng, T, r), self._measure(rng, T, r)

    def overlap_distribution(self, n, reliability, shared_fraction, metric="jaccard", reps=25, seed=42, n_subtypes=1):
        f = METRICS[metric]
        rng = np.random.default_rng(seed)
        return np.array([f(self.simulate(rng, n, reliability, shared_fraction, n_subtypes), self.threshold) for _ in range(reps)])

    def baseline_curve(self, n, reliability, s_grid=DEFAULT_S_GRID, metric="jaccard", reps=25, seeds=DEFAULT_SEEDS, n_subtypes=1):
        """Expected overlap for each shared fraction; replicates pooled over seeds."""
        r_off = int(1e5 * float(reliability)) if np.ndim(reliability) == 0 else 0
        rows = []
        for s in s_grid:
            per_seed = [self.overlap_distribution(n, reliability, s, metric, reps, sd + int(1000 * s) + r_off, n_subtypes) for sd in seeds]
            pooled = np.concatenate(per_seed)
            rows.append({"s": float(s), "mean": float(np.mean([v.mean() for v in per_seed])),
                         "sd_over_seeds": float(np.std([v.mean() for v in per_seed])),
                         "q025": float(np.quantile(pooled, 0.025)), "q975": float(np.quantile(pooled, 0.975))})
        return rows

    def test_heterogeneous_null(self, observed, n, reliability, metric="jaccard", reps=1000, seed=42):
        """One-sided Monte Carlo test of whether observed overlap exceeds that of fully individual deviations (s = 0)."""
        null = self.overlap_distribution(n, reliability, 0.0, metric, reps, seed)
        return {"observed": float(observed), "null_mean": float(null.mean()), "null_q975": float(np.quantile(null, 0.975)),
                "p_one_sided": float((1 + (null >= observed).sum()) / (reps + 1)), "reps": reps}


def identification_rate(a, b):
    """Share of participants whose map in `a` correlates best with their own map in `b` (and vice versa), averaged.

    a, b: participants x features, rows in the same participant order. Chance level is 1 / n.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    za = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    zb = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    R = za @ zb.T
    idx = np.arange(len(a))
    return float((np.mean(R.argmax(1) == idx) + np.mean(R.argmax(0) == idx)) / 2)


def discriminability(a, b):
    """Bridgeford et al. (2021) discriminability for two measurements per participant (chance level 0.5).

    For each measurement, the share of measurements of other participants that are farther away (Euclidean)
    than the participant's other measurement.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)

    def sq(x, y):
        return (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2 * x @ y.T
    Dab, Daa, Dbb = sq(a, b), sq(a, a), sq(b, b)
    w = np.diag(Dab)
    out = []
    for i in range(n):
        for row in (np.r_[Dab[i], Daa[i]], np.r_[Dab[:, i], Dbb[i]]):
            out.append(np.mean(np.delete(row, [i, n + i]) > w[i]))
    return float(np.mean(out))


def implied_shared_fraction(observed, curve):
    """Shared fraction whose expected overlap matches `observed`, with an interval from the 2.5-97.5% bands.

    `curve` is the output of BaselineModel.baseline_curve. Values outside the simulated range are clipped
    and flagged.
    """
    s = np.array([c["s"] for c in curve])
    out = {}
    for key, col in (("estimate", "mean"), ("lower", "q975"), ("upper", "q025")):
        y = np.array([c[col] for c in curve])
        if np.any(np.diff(y) <= 0):
            out[key] = None  # curve not increasing in s: inversion is undefined
            continue
        out[key] = float(np.interp(observed, y, s))
    means = [c["mean"] for c in curve]
    out["below_s0"] = bool(observed < means[0])
    out["above_s1"] = bool(observed > means[-1])
    return out
