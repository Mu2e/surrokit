import unittest
from unittest import mock

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError

from surrokit import (Constraint, NotEnoughData, Problem, fit, lengthscales,
                      predict)
from surrokit import gp
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


def _loss(model):
    """MAP objective of a fitted model, evaluated as the fit does."""
    from gpytorch.mlls import ExactMarginalLogLikelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    loss = gp._map_loss(mll)
    mll.eval()
    return loss


def _single_start_fit(prob, X, Y):
    with mock.patch.object(gp, "FIT_EXTRA_STARTS", 0):
        return fit(prob, X, Y)


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


class TestMultiStartFit(unittest.TestCase):
    def test_deterministic_across_global_rng_states(self):
        X, Y = history(10)
        torch.manual_seed(1)
        a = fit(PROB, X, Y).state_dict()
        torch.manual_seed(12345)
        torch.rand(17)
        b = fit(PROB, X, Y).state_dict()
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_never_worse_than_the_default_start(self):
        cases = [(PROB, *history(10)),
                 (feature_problem(), *feature_rows(20, seed=1)),
                 (feature_problem(product), *feature_rows(8, seed=0))]
        for prob, X, Y in cases:
            with self.subTest(n=len(X), features=prob.features is not None):
                torch.manual_seed(0)
                single = _loss(_single_start_fit(prob, X, Y))
                torch.manual_seed(0)
                multi = _loss(fit(prob, X, Y))
                self.assertLessEqual(multi, single)

    def test_escapes_a_worse_default_optimum(self):
        # On this fixture the default start settles ~0.09 (per data
        # point) above the optimum a prior-drawn start finds.
        X, Y = feature_rows(12, seed=0)
        single = _loss(_single_start_fit(feature_problem(), X, Y))
        multi = _loss(fit(feature_problem(), X, Y))
        self.assertLess(multi, single - 0.05)

    def test_default_start_kept_without_a_real_gain(self):
        # A start must beat the default by more than FIT_MIN_GAIN; with an
        # unreachable margin the fit is exactly the single-start fit.
        X, Y = feature_rows(12, seed=0)
        torch.manual_seed(0)
        single = _single_start_fit(feature_problem(), X, Y).state_dict()
        torch.manual_seed(0)
        with mock.patch.object(gp, "FIT_MIN_GAIN", 1e9):
            multi = fit(feature_problem(), X, Y).state_dict()
        for k in single:
            self.assertTrue(torch.equal(single[k], multi[k]), k)

    def test_global_rng_state_as_without_extra_starts(self):
        X, Y = history(10)
        torch.manual_seed(7)
        _single_start_fit(PROB, X, Y)
        single = torch.get_rng_state()
        torch.manual_seed(7)
        fit(PROB, X, Y)
        self.assertTrue(torch.equal(torch.get_rng_state(), single))

    def test_logs_the_winning_start(self):
        X, Y = history(10)
        with self.assertLogs("surrokit", level="INFO") as cm:
            fit(PROB, X, Y)
        line = [m for m in cm.output if "GP fit: start" in m]
        self.assertEqual(len(line), 1, cm.output)
        self.assertRegex(line[0], r"start \d of 5 won .*best=\S+ "
                                  r"worst=\S+ default=\S+; 0 start")

    def test_failed_starts_are_skipped_and_counted(self):
        import botorch.fit
        real = botorch.fit.fit_gpytorch_mll
        calls = []

        def default_fails(mll, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise ModelFittingError("default start failed")
            return real(mll, **kw)

        X, Y = history(10)
        with mock.patch("botorch.fit.fit_gpytorch_mll", default_fails), \
                self.assertLogs("surrokit", level="INFO") as cm:
            model = fit(PROB, X, Y)
        self.assertEqual(len(calls), 1 + gp.FIT_EXTRA_STARTS)
        self.assertFalse(model.training)
        line = [m for m in cm.output if "GP fit: start" in m][0]
        self.assertIn("default=failed; 1 start(s) failed", line)

    def test_every_start_failing_raises(self):
        X, Y = history(10)
        with mock.patch("botorch.fit.fit_gpytorch_mll",
                        side_effect=ModelFittingError("no fit")), \
                self.assertLogs("surrokit", level="WARNING"):
            with self.assertRaisesRegex(ModelFittingError,
                                        "all 5 GP fit starts failed"):
                fit(PROB, X, Y)

