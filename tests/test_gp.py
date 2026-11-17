import unittest

import numpy as np
import torch

from surrokit import (Constraint, NotEnoughData, Problem, fit, lengthscales,
                      predict)
from tests.common import PROB, history
from tests.test_features import problem as feature_problem
from tests.test_features import product
from tests.test_features import rows as feature_rows


class TestFit(unittest.TestCase):
    def test_fit_and_predict_shapes(self):
        X, Y = history(10)
        model = fit(PROB, X, Y)
        mean, sigma = predict(model, [[0.5, 5.0], [0.1, 1.0]])
        self.assertEqual(mean.shape, (2, 2))
        self.assertEqual(sigma.shape, (2, 2))
        self.assertTrue((sigma >= 0).all())

    def test_not_enough_data(self):
        with self.assertRaises(NotEnoughData):
            fit(PROB, [[0.5, 5.0]], [[1.0, 2.0]])

    def test_shape_mismatch(self):
        X, Y = history(10)
        with self.assertRaises(ValueError):
            fit(PROB, X, Y[:5])
        with self.assertRaises(ValueError):
            fit(PROB, [[0.5]] * 10, Y)  # wrong dim

    def test_constraint_axis_checked_against_y_width(self):
        p = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0),
                    noise=(0.01,),
                    constraint=Constraint(axis=1, min=0.0))
        X, _ = history(10)
        Y1 = [[row[0]] for row in history(10)[1]]  # (n, 1)
        with self.assertRaises(ValueError):
            fit(p, X, Y1)

    def test_pinned_noise_reaches_likelihood(self):
        X, Y = history(10)
        model = fit(PROB, X, Y)
        # Fixed-noise likelihood present when Problem.noise is set.
        self.assertTrue(hasattr(model.likelihood, "noise"))
        # Standardized noise is small (pinned 0.01 sigma, not MLL-fitted).
        n = model.likelihood.noise.detach()
        self.assertLess(float(n.max()), 0.1)

    def test_pinned_noise_roundtrip_exact(self):
        X, Y = history(10)
        model = fit(PROB, X, Y)
        m = 2
        noise_t = model.likelihood.noise.detach()
        noise_std = (noise_t.reshape(-1) if noise_t.numel() == m
                     else noise_t.reshape(m, -1)[:, 0]).sqrt()
        stdvs = model.outcome_transform.stdvs.detach().reshape(-1)
        raw = (noise_std * stdvs).tolist()
        for i, declared in enumerate(PROB.noise):
            self.assertAlmostEqual(raw[i], declared, places=6)

    def test_free_noise_when_none(self):
        X, Y = history(10)
        p = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0))
        model = fit(p, X, Y)  # no error; MLL-fitted noise
        mean, _ = predict(model, [[0.5, 5.0]])
        self.assertEqual(mean.shape, (1, 2))

    def test_interpolates_history(self):
        X, Y = history(10)
        model = fit(PROB, X, Y)
        mean, _ = predict(model, X)
        for i in range(len(X)):
            self.assertAlmostEqual(mean[i][0], Y[i][0], delta=0.05)


class TestNoiseLengthValidation(unittest.TestCase):
    def test_noise_length_mismatch_raises(self):
        from tests.common import history
        X, Y = history(6)  # m = 2
        for bad in ((0.01,), (0.01, 0.01, 0.01)):
            prob = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0),
                           noise=bad)
            with self.assertRaisesRegex(ValueError, "sigmas"):
                fit(prob, X, Y)


# One output: noise sliced with Y, as a sob-only client does.
PROB1 = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0), noise=(0.01,))


class TestLengthscales(unittest.TestCase):
    def test_two_outputs(self):
        X, Y = history(10)
        ls = lengthscales(fit(PROB, X, Y))
        self.assertIsInstance(ls, np.ndarray)
        self.assertEqual(ls.shape, (2, 2))
        self.assertTrue((ls > 0).all())

    def test_one_output(self):
        X, Y = history(10)
        ls = lengthscales(fit(PROB1, X, [[r[0]] for r in Y]))
        self.assertEqual(ls.shape, (1, 2))
        self.assertTrue((ls > 0).all())

    def test_features_add_columns(self):
        X, Y = feature_rows(12, seed=0)
        ls = lengthscales(fit(feature_problem(product), X, Y))
        self.assertEqual(ls.shape, (2, 5))  # 4 knobs + 1 feature
        self.assertTrue((ls > 0).all())

    def test_matches_the_kernel(self):
        X, Y = history(10)
        model = fit(PROB, X, Y)
        np.testing.assert_array_equal(
            lengthscales(model),
            model.covar_module.lengthscale.detach().reshape(2, 2).numpy())

    def test_unexpected_kernel_raises(self):
        from gpytorch.kernels import ScaleKernel
        X, Y = history(10)
        model = fit(PROB, X, Y)
        model.covar_module = ScaleKernel(model.covar_module)
        with self.assertRaisesRegex(TypeError, "lengthscale"):
            lengthscales(model)

    def test_fit_logs_lengthscales(self):
        X, Y = history(10)
        with self.assertLogs("surrokit", level="INFO") as cm:
            fit(PROB, X, Y)
        line = [m for m in cm.output if "GP lengthscales" in m]
        self.assertEqual(len(line), 1, cm.output)
        for token in ("out0 x0=", "out1 x0=", "x1="):
            self.assertIn(token, line[0])

    def test_fit_logs_feature_lengthscales(self):
        X, Y = feature_rows(12, seed=0)
        with self.assertLogs("surrokit", level="INFO") as cm:
            fit(feature_problem(product), X, Y)
        line = [m for m in cm.output if "GP lengthscales" in m][0]
        self.assertIn("x3=", line)
        self.assertIn("f0=", line)
