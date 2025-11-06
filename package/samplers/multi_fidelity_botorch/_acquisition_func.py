from __future__ import annotations

from collections.abc import Callable

from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import cholesky_jitter
from gpytorch.settings import max_cholesky_size
import numpy as np
from optuna.study import Study
from optuna.trial import TrialState
import torch


def qmfkg_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    bounds: "torch.Tensor",
    pending_x: "torch.Tensor | None",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Multi-Fidelity Knowledge Gradient (MFKG).

    Acquisition function for multi-fidelity Bayesian optimization that uses
    past data evaluated at different fidelity levels to jointly determine
    the next candidate point and fidelity level.

    Args:
        train_x:
            Previous parameter configurations. A :class:`torch.Tensor` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. Values are not normalized.
        train_obj:
            Previously observed objectives. A :class:`torch.Tensor` of shape
            ``(n_trials, n_objectives)``. Values are not normalized. Assumes maximization.
        train_con:
            Objective constraints. A :class:`torch.Tensor` of shape ``(n_trials, n_constraints)``.
            A constraint is violated if strictly larger than 0. If no constraints are
            involved, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A :class:`torch.Tensor` of shape ``(2, n_params)``.
            The first and the second rows correspond to the lower and upper bounds.
        pending_x:
            Pending parameter configurations. A :class:`torch.Tensor` of shape
            ``(n_pending, n_params)``.

    Returns:
        Tuple of (candidates, fidelity). Candidates are the next set of parameters,
        and fidelity is the computed fidelity value for the next evaluation.
    """

    if train_obj.size(-1) != 1:
        raise ValueError("MFKG is only supported for single-objective optimization.")

    fidelity_dims = [train_x.size(-1) - 1]

    train_x = normalize(train_x, bounds=bounds)
    train_y = train_obj

    # Remove near-duplicate points to improve numerical stability
    # This helps prevent singular covariance matrices
    unique_mask = torch.ones(train_x.shape[0], dtype=torch.bool)
    for i in range(train_x.shape[0]):
        if unique_mask[i]:
            # Find points very close to current point
            distances = torch.norm(train_x[i + 1 :] - train_x[i], dim=1)
            duplicate_mask = distances < 1e-6
            if duplicate_mask.any():
                # Mark duplicates for removal
                duplicate_indices = torch.where(duplicate_mask)[0] + i + 1
                unique_mask[duplicate_indices] = False

    train_x = train_x[unique_mask]
    train_y = train_y[unique_mask]

    # Use much larger jitter for better numerical stability
    # This helps when the covariance matrix is not positive definite
    # Increased from default 1e-8 to 1e-3/1e-4 to handle numerical issues
    with cholesky_jitter(float_value=1e-3, double_value=1e-4), max_cholesky_size(2000):
        model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            data_fidelities=fidelity_dims,
            outcome_transform=Standardize(m=train_y.size(-1)),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit with more robust settings for numerical stability
        # max_attempts increased to give more chances for successful fit
        fit_gpytorch_mll(
            mll,
            options={
                "maxiter": 100,  # Reduce iterations to avoid numerical issues
                "disp": False,  # Suppress convergence warnings
            },
            max_attempts=10,  # Try multiple times with different initializations
        )

    acqf = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=256,
        X_pending=(normalize(pending_x, bounds=bounds) if pending_x is not None else None),
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 8, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    # Extract fidelity from the candidate
    fidelity_dim = fidelity_dims[0]
    fidelity_value = candidates[0, fidelity_dim].item()

    # Remove fidelity from candidates (return only non-fidelity parameters)
    non_fidelity_candidates = torch.cat(
        [candidates[:, :fidelity_dim], candidates[:, fidelity_dim + 1 :]], dim=1
    )

    return non_fidelity_candidates, torch.tensor([fidelity_value])


def qmfmes_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    bounds: "torch.Tensor",
    pending_x: "torch.Tensor | None",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Multi-Fidelity Max-Value Entropy Search (MFMES).

    Acquisition function for multi-fidelity Bayesian optimization that determines
    the next candidate point and fidelity level using an information entropy
    maximization approach.

    .. seealso::
        :func:`qmfkg_candidates_func` for argument and return value descriptions.
    """

    if train_obj.size(-1) != 1:
        raise ValueError("MFMES is only supported for single-objective optimization.")

    fidelity_dims = [train_x.size(-1) - 1]

    train_x = normalize(train_x, bounds=bounds)
    train_y = train_obj

    # Remove near-duplicate points to improve numerical stability
    # This helps prevent singular covariance matrices
    unique_mask = torch.ones(train_x.shape[0], dtype=torch.bool)
    for i in range(train_x.shape[0]):
        if unique_mask[i]:
            # Find points very close to current point
            distances = torch.norm(train_x[i + 1 :] - train_x[i], dim=1)
            duplicate_mask = distances < 1e-6
            if duplicate_mask.any():
                # Mark duplicates for removal
                duplicate_indices = torch.where(duplicate_mask)[0] + i + 1
                unique_mask[duplicate_indices] = False

    train_x = train_x[unique_mask]
    train_y = train_y[unique_mask]

    # Use much larger jitter for better numerical stability
    # This helps when the covariance matrix is not positive definite
    # Increased from default 1e-8 to 1e-3/1e-4 to handle numerical issues
    with cholesky_jitter(float_value=1e-3, double_value=1e-4), max_cholesky_size(2000):
        model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            data_fidelities=fidelity_dims,
            outcome_transform=Standardize(m=train_y.size(-1)),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit with more robust settings for numerical stability
        # max_attempts increased to give more chances for successful fit
        fit_gpytorch_mll(
            mll,
            options={
                "maxiter": 100,  # Reduce iterations to avoid numerical issues
                "disp": False,  # Suppress convergence warnings
            },
            max_attempts=10,  # Try multiple times with different initializations
        )

    acqf = qMultiFidelityMaxValueEntropy(
        model=model,
        num_fantasies=256,
        candidate_set=torch.rand(256, train_x.size(-1)),  # Candidate set
        X_pending=(normalize(pending_x, bounds=bounds) if pending_x is not None else None),
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 8, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    # Extract fidelity from the candidate
    fidelity_dim = fidelity_dims[0]
    fidelity_value = candidates[0, fidelity_dim].item()

    # Remove fidelity from candidates (return only non-fidelity parameters)
    non_fidelity_candidates = torch.cat(
        [candidates[:, :fidelity_dim], candidates[:, fidelity_dim + 1 :]], dim=1
    )

    return non_fidelity_candidates, torch.tensor([fidelity_value])


def get_default_mf_candidates_func(
    candidates_func_type: str = "mfkg",
) -> Callable[
    [
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor | None",
    ],
    tuple["torch.Tensor", "torch.Tensor"],
]:
    """Select default multi-fidelity acquisition function."""
    if candidates_func_type.lower() == "mfkg":
        return qmfkg_candidates_func
    elif candidates_func_type.lower() == "mfmes":
        return qmfmes_candidates_func
    else:
        raise ValueError(f"Unknown acquisition function: {candidates_func_type}")


def _handle_acquisition_failure(
    study: Study,
) -> float:
    """Handle acquisition function failure with smart fidelity fallback."""
    # Compute smart fidelity based on recent trials and exploration/exploitation balance
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    if len(completed_trials) == 0:
        # If no completed trials, use medium fidelity
        fallback_fidelity = 0.0
    else:
        recent_trials = completed_trials[-min(5, len(completed_trials)) :]  # Last 5 trials
        recent_fidelities = []

        for t in recent_trials:
            fid = t.system_attrs.get("MFBOSampler:Fidelity", 1.0)
            recent_fidelities.append(fid)

        if recent_fidelities:
            avg_recent_fidelity = np.mean(recent_fidelities)

            if avg_recent_fidelity > 0.8:
                fallback_fidelity = np.random.uniform(0.3, 0.6)
            elif avg_recent_fidelity < 0.4:
                fallback_fidelity = np.random.uniform(0.6, 0.9)
            else:
                fallback_fidelity = np.random.uniform(0.4, 0.8)
        else:
            fallback_fidelity = 0.7

    # Store the fallback fidelity
    return fallback_fidelity
