"""Unit tests on synthetic data. Run: python -m unittest discover -s tests (from the relbase folder)."""
import unittest

import numpy as np

from relbase import (BaselineModel, estimate_inputs, icc21, implied_shared_fraction, max_prevalence, mean_jaccard,
                     mean_prevalence, spearman_brown)


class TestMetrics(unittest.TestCase):
    def test_jaccard_by_hand(self):
        z = np.array([[3.0, 0.0, -3.0],   # extremes: f0+, f2-
                      [3.0, 3.0, 0.0],    # f0+, f1+
                      [0.0, 0.0, 0.0]])   # none
        # pairs: (0,1) 1/3; (0,2) 0/2; (1,2) 0/2 -> mean 1/9
        self.assertAlmostEqual(mean_jaccard(z), 1 / 9)
        self.assertAlmostEqual(max_prevalence(z), 2 / 3)
        self.assertAlmostEqual(mean_prevalence(z), 4 / 18)

    def test_blocks_exclude_within_block_pairs(self):
        z = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        self.assertAlmostEqual(mean_jaccard(z), 1 / 3)
        # participants 0 and 1 share a block: only pairs (0,2) and (1,2) remain, both 0
        self.assertEqual(mean_jaccard(z, blocks=["a", "a", "b"]), 0.0)

    def test_sign_matters(self):
        z = np.array([[3.0], [-3.0]])
        self.assertEqual(mean_jaccard(z), 0.0)

    def test_spearman_brown(self):
        self.assertAlmostEqual(float(spearman_brown(0.5, 2)), 2 / 3)
        self.assertAlmostEqual(float(spearman_brown(2 / 3, 0.5)), 0.5)


class TestICC(unittest.TestCase):
    def test_identical_is_one(self):
        x = np.random.default_rng(0).standard_normal((50, 4))
        np.testing.assert_allclose(icc21(x, x), 1.0)

    def test_recovers_true_reliability(self):
        rng = np.random.default_rng(1)
        n, r = 20000, 0.3
        t = rng.standard_normal((n, 3))
        a = np.sqrt(r) * t + np.sqrt(1 - r) * rng.standard_normal((n, 3))
        b = np.sqrt(r) * t + np.sqrt(1 - r) * rng.standard_normal((n, 3))
        np.testing.assert_allclose(icc21(a, b), r, atol=0.02)

    def test_absolute_agreement_penalizes_shift(self):
        rng = np.random.default_rng(2)
        t = rng.standard_normal((5000, 1))
        # a constant shift d adds d^2 / 2 to the denominator: 1 / (1 + 0.5) for unit variance and d = 1
        self.assertAlmostEqual(float(icc21(t, t + 1.0)[0]), 2 / 3, delta=0.02)


class TestModel(unittest.TestCase):
    def setUp(self):
        P = 40
        idx = np.arange(P)
        self.R = 0.3 ** np.abs(idx[:, None] - idx[None, :])
        self.model = BaselineModel(self.R, self.R, scale=1.1)

    def test_scale(self):
        z = self.model.simulate(np.random.default_rng(3), 4000, 0.4, 0.5)
        self.assertAlmostEqual(float(z.std()), 1.1, delta=0.03)

    def test_reliability_matches_icc(self):
        # two measurements of the same true deviations should agree at the given reliability
        m = BaselineModel(n_features=30)
        rng = np.random.default_rng(4)
        n, r = 5000, np.linspace(0.1, 0.9, 30)
        T = rng.standard_normal((n, 30))
        a = np.sqrt(r) * T + np.sqrt(1 - r) * rng.standard_normal((n, 30))
        b = np.sqrt(r) * T + np.sqrt(1 - r) * rng.standard_normal((n, 30))
        np.testing.assert_allclose(icc21(a, b), r, atol=0.04)
        self.assertEqual(m.simulate(rng, 10, r, 0.0).shape, (10, 30))

    def test_curve_increases_with_shared_fraction(self):
        curve = self.model.baseline_curve(60, 0.6, reps=5, seeds=(42, 7))
        means = [c["mean"] for c in curve]
        self.assertTrue(all(b > a for a, b in zip(means, means[1:])))

    def test_reproducible(self):
        a = self.model.overlap_distribution(30, 0.5, 0.5, reps=3, seed=11)
        b = self.model.overlap_distribution(30, 0.5, 0.5, reps=3, seed=11)
        np.testing.assert_array_equal(a, b)

    def test_rejects_bad_reliability(self):
        with self.assertRaises(ValueError):
            self.model.simulate(np.random.default_rng(0), 5, 1.2, 0.0)

    def test_null_test(self):
        res = self.model.test_heterogeneous_null(1.0, 40, 0.5, reps=50)
        self.assertLess(res["p_one_sided"], 0.05)
        res = self.model.test_heterogeneous_null(0.0, 40, 0.5, reps=50)
        self.assertEqual(res["p_one_sided"], 1.0)


class TestInference(unittest.TestCase):
    def test_interpolation(self):
        curve = [{"s": 0.0, "mean": 0.01, "q025": 0.008, "q975": 0.012},
                 {"s": 1.0, "mean": 0.11, "q025": 0.09, "q975": 0.13}]
        out = implied_shared_fraction(0.06, curve)
        self.assertAlmostEqual(out["estimate"], 0.5)
        self.assertLess(out["lower"], out["estimate"])
        self.assertGreater(out["upper"], out["estimate"])
        self.assertTrue(implied_shared_fraction(0.005, curve)["below_s0"])

    def test_estimate_inputs(self):
        rng = np.random.default_rng(5)
        t = rng.standard_normal((300, 10))
        h1 = t + rng.standard_normal((300, 10))
        h2 = t + rng.standard_normal((300, 10))
        d = estimate_inputs((h1 + h2) / 2, h1, h2)
        self.assertEqual(d["R_signal"].shape, (10, 10))
        np.testing.assert_allclose(np.median(d["reliability_half"]), 0.5, atol=0.1)
        np.testing.assert_allclose(d["reliability_full"], spearman_brown(np.clip(d["reliability_half"], 0, 1), 2))


if __name__ == "__main__":
    unittest.main()
