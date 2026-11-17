"""Fixed-noise SingleTaskGP fit + posterior predict.

Problem.features, when set, become extra GP inputs: an input transform
inside the model appends them to the knobs, so every caller -- predict,
the pickers, the acquisition optimizer -- keeps working in knob space.

float64 on CPU throughout: histories are small (hundreds of rows), CPU
beats GPU including transfer, and float64 keeps optimizer paths
deterministic. The library never touches torch's global default dtype.
"""
from __future__ import annotations

import copy
import logging

import torch

from .problem import NotEnoughData, Problem

log = logging.getLogger("surrokit")

DEVICE = torch.device("cpu")

# Multi-start hyperparameter fit. The default start (the prior mode) can
# land in a local optimum of the marginal likelihood: a GP confidently
# wrong about a region the loop then never samples again. FIT_EXTRA_STARTS
# more fits start from hyperparameters drawn from their priors and the
# best MAP objective wins. FIT_SEED seeds those draws inside a forked RNG,
# so they are deterministic and never touch the caller's torch stream.
FIT_EXTRA_STARTS = 4
FIT_SEED = 0


def to_f64(a) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float64, device=DEVICE)


def bounds_tensor(problem: Problem) -> torch.Tensor:
    lo = torch.tensor(problem.bounds_lo, dtype=torch.float64, device=DEVICE)
    hi = torch.tensor(problem.bounds_hi, dtype=torch.float64, device=DEVICE)
    return torch.stack([lo, hi], dim=0)


def fit(problem: Problem, X, Y):
    """Fit a SingleTaskGP on (X, Y). Validates shapes against the Problem;
    raises NotEnoughData below 2 rows, ValueError on any mismatch."""
    Xt, Yt = to_f64(X), to_f64(Y)
    if Xt.ndim != 2 or Yt.ndim != 2:
        raise ValueError(f"X and Y must be 2D, got {Xt.ndim}D / {Yt.ndim}D")
    if Xt.shape[0] != Yt.shape[0]:
        raise ValueError(f"X has {Xt.shape[0]} rows but Y has {Yt.shape[0]}")
    if Xt.shape[1] != problem.dim:
        raise ValueError(f"X is {Xt.shape[1]}D but problem.dim is "
                         f"{problem.dim}")
    if problem.constraint is not None and problem.constraint.axis >= Yt.shape[1]:
        raise ValueError(f"constraint.axis {problem.constraint.axis} out of "
                         f"range for {Yt.shape[1]} output axes")
    if problem.noise is not None and len(problem.noise) != Yt.shape[1]:
        raise ValueError(f"noise has {len(problem.noise)} sigmas but Y has "
                         f"{Yt.shape[1]} output axes")
    if Xt.shape[0] < 2:
        raise NotEnoughData(f"{Xt.shape[0]} history rows; need >= 2 to fit")
    return _fit_model(Xt, Yt, bounds_tensor(problem), problem.noise,
                      problem.features)


# Knob-box sample that fixes the feature scaling: a property of the
# problem, not of the history, so two rows scale like two hundred.
FEATURE_RANGE_SAMPLES = 4096


def _feature_bounds(features, bounds) -> torch.Tensor:
    """(2, k) range of `features` over a fixed Sobol sample of the knob
    box. Raises ValueError for a function that does not return finite
    (..., n, k), ignores leading batch dimensions, or has a constant
    output."""
    from torch.quasirandom import SobolEngine

    d = bounds.shape[-1]
    unit = SobolEngine(d, scramble=True, seed=0).draw(
        FEATURE_RANGE_SAMPLES, dtype=bounds.dtype)
    S = bounds[0] + unit * (bounds[1] - bounds[0])
    F = features(S)
    if (not isinstance(F, torch.Tensor) or F.ndim != 2
            or F.shape[0] != S.shape[0] or F.shape[1] < 1):
        got = tuple(F.shape) if isinstance(F, torch.Tensor) else type(F).__name__
        raise ValueError(f"features must map (n, {d}) knobs to an (n, k) "
                         f"tensor, got {got} for n={S.shape[0]}")
    if not torch.isfinite(F).all():
        raise ValueError("features returned a non-finite value inside the "
                         "knob box")
    k = F.shape[1]
    Fb = features(S[:6].reshape(2, 3, d))
    if (not isinstance(Fb, torch.Tensor) or Fb.shape != (2, 3, k)
            or not torch.allclose(Fb, F[:6].reshape(2, 3, k))):
        raise ValueError("features must keep leading batch dimensions: "
                         f"(2, 3, {d}) knobs must give (2, 3, {k})")
    lo, hi = F.min(dim=0).values, F.max(dim=0).values
    flat = (hi - lo) <= 0
    if flat.any():
        raise ValueError(f"feature(s) {torch.nonzero(flat).flatten().tolist()} "
                         "are constant over the knob box")
    return torch.stack([lo, hi], dim=0).to(bounds)


def _input_transform(bounds, features):
    """Normalize the knobs; with features, append them first and
    normalize knobs and features together."""
    from botorch.models.transforms.input import (ChainedInputTransform,
                                                 InputTransform, Normalize)

    class Append(InputTransform, torch.nn.Module):
        # botorch's AppendFeatures is for perturbation sets (it grows
        # the n axis); this one only widens the last axis.
        transform_on_train = True
        transform_on_eval = True
        transform_on_fantasize = True

        def transform(self, X):
            return torch.cat([X, features(X)], dim=-1)

    d = bounds.shape[-1]
    if features is None:
        return Normalize(d=d, bounds=bounds)
    full = torch.cat([bounds, _feature_bounds(features, bounds)], dim=-1)
    return ChainedInputTransform(
        append=Append(),
        normalize=Normalize(d=full.shape[-1], bounds=full))


def _map_loss(mll) -> float:
    """The MAP objective: negative marginal log likelihood including the
    hyperparameter prior terms, summed over outputs, in train mode on the
    training data -- what botorch's fit minimizes. Leaves mll in train
    mode."""
    from botorch.optim.closures import get_loss_closure

    mll.train()
    with torch.no_grad():
        return float(get_loss_closure(mll)().sum())


def _fit_multistart(mll) -> None:
    """Fit mll from 1 + FIT_EXTRA_STARTS starts, load the one with the
    lowest MAP objective and leave mll in eval mode.

    Start 0 is fit_gpytorch_mll exactly as a single fit runs it (prior
    mode, botorch's own retry policy, the global RNG), so where it wins
    the model is the single-start model. Starts 1.. reset to the initial
    state and draw from the priors under torch.manual_seed(FIT_SEED) in a
    forked RNG. A start that raises, or that botorch judges unconverged,
    is skipped and counted; ModelFittingError if every start fails.
    """
    from botorch.exceptions.errors import ModelFittingError
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim.utils import sample_all_priors

    init = copy.deepcopy(mll.state_dict())
    fits = []      # (loss, start, state), in start order
    failed = []    # (start, error)

    def attempt(start, **kwargs):
        try:
            fit_gpytorch_mll(mll, **kwargs)
        except (ModelFittingError, RuntimeError) as err:
            failed.append((start, err))
            log.warning("GP fit start %d failed: %s: %s", start,
                        type(err).__name__, err)
            return
        fits.append((_map_loss(mll), start, copy.deepcopy(mll.state_dict())))

    attempt(0)
    with torch.random.fork_rng():
        torch.manual_seed(FIT_SEED)
        for start in range(1, 1 + FIT_EXTRA_STARTS):
            mll.train()
            mll.load_state_dict(init)
            sample_all_priors(mll.model)
            # One attempt: botorch's retry would redraw, and a failed
            # start is counted, not retried.
            attempt(start, max_attempts=1)
    n_starts = 1 + FIT_EXTRA_STARTS
    if not fits:
        raise ModelFittingError(
            f"all {n_starts} GP fit starts failed; last error: "
            f"{type(failed[-1][1]).__name__}: {failed[-1][1]}"
        ) from failed[-1][1]
    # min() keeps the first of equal losses, i.e. the lowest start index.
    loss, best, state = min(fits, key=lambda f: f[0])
    mll.train()
    mll.load_state_dict(state)
    mll.eval()
    default = (f"{fits[0][0]:.6g}" if fits[0][1] == 0 else "failed")
    log.info("GP fit: start %d of %d won (0 = default); MAP objective "
             "best=%.6g worst=%.6g default=%s; %d start(s) failed", best,
             n_starts, loss, max(f[0] for f in fits), default, len(failed))


def lengthscales(model):
    """Fitted lengthscales as a numpy array (m, d_in): one row per output,
    one column per GP input in the model's normalized input units (the
    knobs, then any Problem.features). Raises TypeError for a covariance
    that is not a single ARD kernel with one lengthscale per input."""
    ck = getattr(model, "covar_module", None)
    if ck is None or not getattr(ck, "has_lengthscale", False):
        raise TypeError("lengthscales() needs a model whose covar_module "
                        "has lengthscales (e.g. an ARD RBF kernel), got "
                        f"{type(ck).__name__}")
    ls = ck.lengthscale.detach()
    m, d_in = model.num_outputs, ck.ard_num_dims
    want = (1, d_in) if m == 1 else (m, 1, d_in)
    if d_in is None or tuple(ls.shape) != want:
        raise TypeError(f"unexpected lengthscale shape {tuple(ls.shape)} "
                        f"for {m} output(s) and ard_num_dims={d_in}; "
                        f"expected {want}")
    return ls.reshape(m, d_in).cpu().numpy()


def _fit_model(X, Y, bounds, noise, features=None):
    """Fit a SingleTaskGP (input Normalize + outcome Standardize).

    noise (ABSOLUTE per-output sigma) squares into train_Yvar -> a
    fixed-noise likelihood. Left free, MLL routinely over-fits noise on
    small replicated datasets and erases real optima; clients with
    replicate-measured noise should always pin it.

    Hyperparameters are fit from several starts (_fit_multistart).
    """
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    m = Y.shape[-1]
    train_Yvar = None
    if noise is not None:
        # Broadcast sigma^2 across rows: noise is a property of the
        # client's measurement budget, not the point.
        sig = torch.tensor([float(v) for v in noise],
                           dtype=Y.dtype, device=Y.device)
        train_Yvar = (sig ** 2).expand(Y.shape[0], m).contiguous()

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        train_Yvar=train_Yvar,
        input_transform=_input_transform(bounds, features),
        outcome_transform=Standardize(m=m),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _fit_multistart(mll)
    # Noise audit: likelihood noise is standardized (x stdvs = raw);
    # fixed-noise carries (m, n) -- collapse to per-output.
    noise_t = model.likelihood.noise.detach()
    noise_std = (noise_t.reshape(-1) if noise_t.numel() == m
                 else noise_t.reshape(m, -1)[:, 0]).sqrt()
    stdvs = model.outcome_transform.stdvs.detach().reshape(-1)
    raw = [f"{v:.3e}" for v in (noise_std * stdvs).tolist()]
    src = "FIXED (problem.noise)" if train_Yvar is not None else "MLL-fitted"
    log.info("GP noise sigma per output [%s]: raw=%s standardized=%s",
             src, raw, [f"{v:.3f}" for v in noise_std.tolist()])
    # Lengthscale audit, in normalized input units (the knob box is
    # [0, 1] per knob). Knobs x0.., then features f0.. (no names).
    ls = lengthscales(model)
    d = bounds.shape[-1]
    names = ([f"x{i}" for i in range(d)]
             + [f"f{j}" for j in range(ls.shape[1] - d)])
    log.info("GP lengthscales per output (normalized inputs): %s",
             "; ".join(f"out{i} " + " ".join(
                 f"{n}={v:.3g}" for n, v in zip(names, row))
                 for i, row in enumerate(ls)))
    return model


def predict(model, X):
    """Posterior mean and stddev at X: two numpy arrays, each (n, m)."""
    Xq = to_f64(X)
    with torch.no_grad():
        post = model.posterior(Xq)
        mean = post.mean
        sig = post.variance.clamp_min(0).sqrt()
    return mean.cpu().numpy(), sig.cpu().numpy()
