"""Problem.features: client functions of the knobs as extra GP inputs."""
import unittest

import numpy as np
import torch

from surrokit import PICKER_CHOICES, Constraint, Problem, ask, fit, predict

LO, HI = (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)


def product(X):
    """One feature: the product of the first two knobs."""
    return (X[..., 0] * X[..., 1]).unsqueeze(-1)


def rows(n, seed):
    """Seeded (X, Y): axis 0 depends on the knobs only through product()."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 4, generator=g, dtype=torch.float64)
    y0 = torch.sin(12.0 * product(X).squeeze(-1))
    y1 = 3.0 - X[:, 2]
    return X.tolist(), torch.stack([y0, y1], dim=-1).tolist()


def problem(features=None, **kw):
    return Problem(bounds_lo=LO, bounds_hi=HI, noise=(0.01, 0.01),
                   features=features, **kw)


class TestDeclaration(unittest.TestCase):
    def test_default_is_no_features(self):
        self.assertIsNone(Problem(bounds_lo=LO, bounds_hi=HI).features)

    def test_features_must_be_callable(self):
        with self.assertRaises(ValueError):
            problem(features=3)


class TestFit(unittest.TestCase):
    def test_no_features_keeps_the_knob_only_model(self):
        X, Y = rows(12, seed=0)
        model = fit(problem(), X, Y)
        self.assertEqual(model.covar_module.lengthscale.shape[-1], 4)

    def test_features_become_extra_inputs(self):
        X, Y = rows(12, seed=0)
        model = fit(problem(product), X, Y)
        self.assertEqual(model.covar_module.lengthscale.shape[-1], 5)
        mean, sigma = predict(model, X[:3])
        self.assertEqual(mean.shape, (3, 2))
        self.assertTrue(np.isfinite(mean).all() and (sigma >= 0).all())

    def test_features_lower_the_held_out_error(self):
        X, Y = rows(30, seed=1)
        Xt, Yt = rows(200, seed=2)
        truth = np.array(Yt)[:, 0]

        def rmse(prob):
            mean, _ = predict(fit(prob, X, Y), Xt)
            return float(np.sqrt(np.mean((mean[:, 0] - truth) ** 2)))

        self.assertLess(rmse(problem(product)), 0.5 * rmse(problem()))

    def test_two_rows_fit(self):
        X, Y = rows(2, seed=0)
        mean, _ = predict(fit(problem(product), X, Y), X)
        self.assertTrue(np.isfinite(mean).all())

    def test_wrong_shape_raises(self):
        X, Y = rows(12, seed=0)
        with self.assertRaises(ValueError):
            fit(problem(lambda X: X[..., 0] * X[..., 1]), X, Y)  # no k axis

    def test_non_finite_raises(self):
        X, Y = rows(12, seed=0)
        with self.assertRaises(ValueError):
            fit(problem(lambda X: torch.log(X[..., :1] - 2.0)), X, Y)

    def test_constant_feature_raises(self):
        X, Y = rows(12, seed=0)
        with self.assertRaises(ValueError):
            fit(problem(lambda X: torch.ones_like(X[..., :1])), X, Y)

    def test_function_that_ignores_batch_dims_raises(self):
        X, Y = rows(12, seed=0)

        def flat_only(X):  # correct for (n, d), wrong for (b, n, d)
            return (X.reshape(-1, 4)[:, 0] * X.reshape(-1, 4)[:, 1]).unsqueeze(-1)

        with self.assertRaises(ValueError):
            fit(problem(flat_only), X, Y)


class TestAsk(unittest.TestCase):
    def test_every_picker_returns_in_bounds_knobs(self):
        X, Y = rows(12, seed=0)
        prob = problem(product, constraint=Constraint(axis=1, min=2.2))
        for picker in PICKER_CHOICES:
            with self.subTest(picker=picker):
                picks = ask(prob, X, Y, q=2, picker=picker, seed=3, pool=512)
                self.assertEqual(len(picks), 2)
                for row in picks:
                    self.assertEqual(len(row), 4)  # knobs only, no features
                    self.assertTrue(all(0.0 <= v <= 1.0 for v in row))

    def test_the_acquisition_calls_features_with_batched_inputs(self):
        X, Y = rows(12, seed=0)
        ndims = set()

        def recording(X):
            ndims.add(X.ndim)
            return product(X)

        ask(problem(recording), X, Y, q=2, picker="qlnei", seed=3)
        self.assertTrue(max(ndims) >= 3, ndims)


if __name__ == "__main__":
    unittest.main()
