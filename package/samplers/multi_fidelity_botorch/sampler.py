from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Literal

import numpy
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from ._acquisition_func import get_default_mf_candidates_func


with try_import() as _imports:
    from botorch.utils.sampling import manual_seed
    import torch


class MFBotorchSampler(BaseSampler):
    """Multi-fidelity Bayesian optimization sampler.

    Uses BoTorch's multi-fidelity optimization algorithms to suggest parameter
    configurations at different fidelity levels. Fidelity parameters are
    transformed to continuous space and passed to BoTorch.

    .. seealso::
        See BoTorch's `Multi-Fidelity Optimization <https://botorch.org/tutorials/multi_fidelity_bo>`_
        tutorial for details.

    Args:
        candidates_func:
            Function to suggest next candidates. Can specify custom multi-fidelity function.
            If omitted, automatically selected based on ``acquisition_function``.
        acquisition_function:
            Type of acquisition function to use. Either "mfkg" (Multi-Fidelity Knowledge Gradient)
            or "mfmes" (Multi-Fidelity Max-Value Entropy Search).
        fidelity_dims:
            List of fidelity parameter dimension indices. If omitted, uses last dimension.
        n_startup_trials:
            Number of initial trials. Uses independent sampling up to this count.
        independent_sampler:
            Independent sampler for initial trials and conditional parameters.
        seed:
            Random seed.
        device:
            torch device for BoTorch computations. Specify CUDA device for acceleration.
    """

    def __init__(
        self,
        *,
        candidates_func: (
            Callable[
                [
                    "torch.Tensor",
                    "torch.Tensor",
                    "torch.Tensor",
                    "torch.Tensor | None",
                ],
                tuple["torch.Tensor", "torch.Tensor"],
            ]
            | None
        ) = None,
        acquisition_function: Literal["mfkg", "mfmes"] = "mfkg",
        n_startup_trials: int = 10,
        independent_sampler: BaseSampler | None = None,
        seed: int | None = None,
        device: "torch.device | None" = None,
    ):
        _imports.check()

        self._candidates_func = candidates_func
        self._acquisition_function = acquisition_function
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed

        self._study_id: int | None = None
        self._search_space = IntersectionSearchSpace()
        self._device = device or torch.device("cpu")
        self._current_trial_params: dict[str, Any] | None = None
        self._current_trial_fidelity: float | None = None

        self._startup_fidelity = []
        for i in range(self._n_startup_trials):
            if i % 3 == 0:  # Low fidelity
                fidelity = numpy.random.uniform(0.0, 0.33)
            elif i % 3 == 1:  # Mid fidelity
                fidelity = numpy.random.uniform(0.33, 0.67)
            else:  # High fidelity
                fidelity = numpy.random.uniform(0.67, 1.0)
            self._startup_fidelity.append(fidelity)

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            raise RuntimeError("BotorchMultiFidelitySampler cannot handle multiple studies.")

        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        assert isinstance(search_space, dict)

        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        trials = completed_trials

        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            startup_fidelity = self._startup_fidelity[n_trials]
            study._storage.set_trial_system_attr(
                trial._trial_id, "MFBOSampler:Fidelity", startup_fidelity
            )
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)

        if n_objectives != 1:
            raise ValueError(
                "Multi-fidelity optimization supports only single-objective problems."
            )

        values: numpy.ndarray | torch.Tensor = numpy.empty((n_trials, 1), dtype=numpy.float64)
        params: numpy.ndarray | torch.Tensor
        bounds: numpy.ndarray | torch.Tensor = trans.bounds

        params = numpy.empty(
            (n_trials, trans.bounds.shape[0] + 1), dtype=numpy.float64
        )  # +1 for fidelity

        fidelity_bounds = numpy.array([[0.0, 1.0]], dtype=numpy.float64)
        bounds = numpy.concatenate([bounds, fidelity_bounds], axis=0)

        for trial_idx, trial in enumerate(trials):
            if trial.state == TrialState.COMPLETE:
                regular_params = trans.transform(trial.params)
                trial_fidelity = trial.system_attrs.get("MFBOSampler:Fidelity", 1.0)

                # Combine regular parameters with fidelity (fidelity as last dimension)
                params[trial_idx, :-1] = regular_params
                params[trial_idx, -1] = trial_fidelity

                assert len(study.directions) == len(trial.values)
                for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                    assert value is not None
                    if direction == StudyDirection.MINIMIZE:
                        value *= -1
                    values[trial_idx, obj_idx] = value

        values = torch.from_numpy(values).to(self._device)
        params = torch.from_numpy(params).to(self._device)
        bounds = torch.from_numpy(bounds).to(self._device)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = get_default_mf_candidates_func(
                acquisition_function=self._acquisition_function
            )

        with manual_seed(self._seed):
            # Call multi-fidelity candidate function to get both candidates and fidelity
            try:
                candidates, fidelity = self._candidates_func(
                    params,
                    values,
                    bounds,
                    None,
                )

                # Store the computed fidelity for use in sample_independent
                fidelity_value = fidelity.item()
                self._current_trial_fidelity = fidelity_value
                study._storage.set_trial_system_attr(
                    trial._trial_id + 1, "MFBOSampler:Fidelity", fidelity_value
                )

                if self._seed is not None:
                    self._seed += 1

            except Exception as e:
                # Handle numerical issues and provide appropriate fallback
                import warnings

                warnings.warn(
                    f"Multi-fidelity acquisition function failed due to numerical issues: {e}. "
                    "Falling back to random parameter sampling with computed fidelity.",
                    UserWarning,
                )

                # Generate random parameters and smart fidelity fallback
                self._handle_acquisition_failure(study)
                return {}

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if (
            candidates.size(0) != bounds.size(1) - 1
        ):  # -1 because fidelity dimension is removed from candidates
            raise ValueError(
                "Candidates size must match with the given bounds (excluding fidelity). "
                f"Actual candidates: {candidates.size(0)}, expected: {bounds.size(1) - 1}."
            )

        # Need to add back fidelity dimension for untransform
        # Fidelity was the last dimension in train_x, so we need to reconstruct it
        fidelity_value = trial.system_attrs["MFBOSampler:Fidelity"]

        # Reconstruct full parameter vector with fidelity as last dimension
        full_candidates = torch.zeros(candidates.size(0) + 1)
        full_candidates[:-1] = candidates  # All regular parameters
        full_candidates[-1] = fidelity_value  # Fidelity as last dimension

        # Store the computed parameters and return empty dict to ensure
        # sample_independent is called to preserve fidelity
        regular_params = trans.untransform(full_candidates[:-1].cpu().numpy())

        # Store for use in sample_independent
        self._current_trial_params = regular_params
        # Also store the fidelity from trial attributes
        self._current_trial_fidelity = trial.system_attrs.get("MFBOSampler:Fidelity")

        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # Get the trial number to determine if we're in startup mode
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        n_trials = len(completed_trials)

        # Check if fidelity is already set (should be set by acquisition function for non-startup trials)
        current_fidelity = trial.system_attrs.get("MFBOSampler:Fidelity")
        # Set fidelity based on trial type and stored values
        if current_fidelity is None:
            if n_trials < self._n_startup_trials:
                study._storage.set_trial_system_attr(trial._trial_id, "MFBOSampler:Fidelity", 1.0)
            else:
                # For non-startup trials, use stored fidelity from acquisition function
                if self._current_trial_fidelity is not None:
                    study._storage.set_trial_system_attr(
                        trial._trial_id,
                        "MFBOSampler:Fidelity",
                        self._current_trial_fidelity,
                    )
                else:
                    # Fallback to default if no stored fidelity
                    study._storage.set_trial_system_attr(
                        trial._trial_id, "MFBOSampler:Fidelity", 1.0
                    )

        # For non-startup trials, use stored parameters from acquisition function
        if n_trials >= self._n_startup_trials and self._current_trial_params is not None:
            if param_name in self._current_trial_params:
                return self._current_trial_params[param_name]

        # For startup trials or if no stored params, use random sampling
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(numpy.iinfo(numpy.int32).max)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)

    def _handle_acquisition_failure(
        self,
        study: Study,
    ) -> None:
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
                avg_recent_fidelity = numpy.mean(recent_fidelities)

                if avg_recent_fidelity > 0.8:
                    fallback_fidelity = numpy.random.uniform(0.3, 0.6)
                elif avg_recent_fidelity < 0.4:
                    fallback_fidelity = numpy.random.uniform(0.6, 0.9)
                else:
                    fallback_fidelity = numpy.random.uniform(0.4, 0.8)
            else:
                fallback_fidelity = 0.7

        # Store the fallback fidelity
        self._current_trial_fidelity = fallback_fidelity
