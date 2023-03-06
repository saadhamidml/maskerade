"""Surrogate models for the integrand in Bayesian Quadrature."""
from typing import Sequence, Mapping, Union, Callable
import torch
from torch import Tensor
import gpytorch
from gpytorch import settings
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.means import Mean, ConstantMean
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.exact_prediction_strategies import prediction_strategy

from .kernels import (
    InterDomainScaleKernel, EDMMDKernel
)

class GPModel(gpytorch.models.ExactGP):
    """Vanilla GP with additional methods for working with
    hyperparameters concatenated into one tensor, rather than as stored
    as properties of submodules.
    """
    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: Likelihood,
            mean_module: Mean,
            covar_module: Kernel
    ):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        """Compute posterior at x for each GP in model (if batch of
        GPs).
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class WarpedGPModel(GPModel):
    """Warped GP Model."""
    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: _GaussianLikelihoodBase,
            mean_module: Mean,
            covar_module: Kernel,
            warping_function: Callable
    ):
        self.unwarped_train_targets = train_y
        self.warping_function = warping_function
        super(WarpedGPModel, self).__init__(
            train_x,
            self.warping_function(train_y),
            likelihood,
            mean_module,
            covar_module
        )

    def set_train_data(self, inputs=None, targets=None, strict: bool = False):
        """Update training data.

        Set strict to False to allow the shape of the training data to
        change.
        """
        self.unwarped_train_targets = targets
        super().set_train_data(
            inputs=inputs,
            targets=self.warping_function(targets),
            strict=strict
        )

    def warped_posterior(self, *args, **kwargs):
        """Posterior in the warped space."""
        return super().__call__(*args, **kwargs)


class WsabiLGPModel(WarpedGPModel):
    """WSABI-L model."""
    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: _GaussianLikelihoodBase,
            mean_module: Mean,
            covar_module: Kernel,
            alpha_factor: float = 0.8
    ):
        self.alpha_factor = alpha_factor
        self.alpha = self.alpha_factor * torch.min(train_y)

        def warping_function(x):
            return torch.sqrt(2 * (x - self.alpha))
        super(WsabiLGPModel, self).__init__(
            train_x,
            train_y,
            likelihood,
            mean_module,
            covar_module,
            warping_function
        )

        # Clamp mean function to zero.
        self.mean_module.constant.data = torch.tensor(0.)
        self.mean_module.constant.requires_grad_(False)
        # Clamp noise to 1e-3
        self.likelihood.noise = torch.tensor(1e-3)
        self.likelihood.raw_noise.requires_grad_(False)

    def set_train_data(self, inputs=None, targets=None, strict: bool = False):
        """Update training data.

        Set strict to False to allow the shape of the training data to
        change.
        """
        self.alpha = self.alpha_factor * torch.min(targets)
        super().set_train_data(
            inputs=inputs,
            targets=targets,
            strict=strict
        )

    def __call__(self, *args, **kwargs):
        warped_distribution = super().__call__(*args, **kwargs)
        if self.training:
            return warped_distribution
        warped_mean = warped_distribution.mean
        mean_x = self.alpha + warped_mean ** 2 / 2
        if self.batched:
            covar_factors = torch.einsum(
                'bi,bj->bij',
                warped_mean,
                warped_mean
            )
        else:
            covar_factors = torch.einsum(
                'i,j->ij',
                warped_mean,
                warped_mean
            )
        covar_x = warped_distribution.lazy_covariance_matrix * covar_factors
        return MultivariateNormal(mean_x, covar_x)


class InterDomainWsabiLGPModel(WarpedGPModel):
    def __init__(
            self,
            train_x: Sequence[Tensor],
            train_y: Sequence[Tensor],
            train_targets_log_offset: Tensor,
            likelihood: _GaussianLikelihoodBase,
            mean_module: Mean,
            covar_module: Kernel,
            alpha_factor: float = 0.8,
    ):
        self.unwarped_train_targets_log_offset = train_targets_log_offset
        self.alpha_factor = alpha_factor
        self.alpha = (
                self.alpha_factor
                * torch.tensor([torch.min(i) for i in train_y]).min()
        )

        def warping_function(xs: Sequence[Tensor]) -> Sequence[Tensor]:
            return [torch.sqrt(2 * (x - self.alpha)) for x in xs]

        super(InterDomainWsabiLGPModel, self).__init__(
            train_x,
            train_y,
            likelihood,
            mean_module,
            covar_module,
            warping_function
        )

        self.batched = False

        # Clamp mean function to zero.
        self.mean_module.constant.data = torch.tensor(0.)
        self.mean_module.constant.requires_grad_(False)
        # Clamp noise to 1e-3
        self.likelihood.noise = torch.tensor(1e-3)
        self.likelihood.raw_noise.requires_grad_(False)

    def set_train_data(self, inputs=None, targets=None, strict: bool = False):
        """Update training data.

        Set strict to False to allow the shape of the training data to
        change.
        """
        self.alpha = (
                self.alpha_factor
                * torch.tensor([torch.min(i) for i in targets]).min()
        )
        super().set_train_data(
            inputs=inputs,
            targets=targets,
            strict=strict
        )

    @staticmethod
    def prediction(
            train_inputs,
            train_output,
            train_targets,
            likelihood,
            full_output
    ):
        strategy = prediction_strategy(
            train_inputs=train_inputs,
            train_prior_dist=train_output,
            train_labels=torch.cat(train_targets, dim=0),
            likelihood=likelihood,
        )
        full_mean, full_covar = (
            full_output.loc, full_output.lazy_covariance_matrix
        )

        # Determine the shape of the joint distribution
        batch_shape = full_output.batch_shape
        joint_shape = full_output.event_shape
        tasks_shape = joint_shape[1:]  # For multitask learning
        test_shape = torch.Size([
            joint_shape[0] - strategy.train_shape[0],
            *tasks_shape
        ])

        # Make the prediction
        with settings._use_eval_tolerance():
            (
                predictive_mean, predictive_covar
            ) = strategy.exact_prediction(
                full_mean, full_covar
            )

        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(
            *batch_shape, *test_shape
        ).contiguous()
        output = full_output.__class__(
            predictive_mean, predictive_covar
        )

        return output

    def __call__(self, *args, **kwargs):
        # corresponding number of GMM components.
        train_inputs = list(
            self.train_inputs
        ) if self.train_inputs is not None else []
        if self.training:
            n_blocks = len(self.train_inputs)
            full_inputs = train_inputs
        else:
            inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args[0]]
            n_blocks = len(self.train_inputs) + len(inputs)
            full_inputs = train_inputs + inputs
        n_elements = torch.tensor([i.size(0) for i in full_inputs])
        n_train_elements = n_elements[0:len(self.train_inputs)].sum()
        total_length = n_elements.sum().item()
        mean = torch.empty(total_length)
        covariance = torch.empty(total_length, total_length)
        i_start = 0
        j_start = 0
        for i in range(n_blocks):
            i_end = n_elements[0:i+1].sum()
            mean[i_start: i_end] = self.mean_module(full_inputs[i])
            for j in range(i, n_blocks):
                j_end = n_elements[0:j+1].sum()
                covariance[
                    i_start: i_end, j_start: j_end
                ] = self.covar_module(
                    full_inputs[i], full_inputs[j]
                ).evaluate()
                j_start = j_end
            i_start = i_end
            j_start = i_end  # Only Upper Triangle blocks computed
        covariance = (
                torch.triu(covariance)
                + torch.triu(covariance, diagonal=1).transpose(dim0=0, dim1=1)
        )

        train_output = MultivariateNormal(
            mean[0:n_train_elements],
            covariance[0:n_train_elements, 0:n_train_elements]
        )
        if self.training:
            return train_output
        full_output = MultivariateNormal(mean, covariance)

        warped_distribution = self.prediction(
            train_inputs,
            train_output,
            self.train_targets,
            self.likelihood,
            full_output
        )
        warped_mean = warped_distribution.mean

        if self.batched:
            covar_factors = torch.einsum(
                'bi,bj->bij',
                warped_mean,
                warped_mean
            )
        else:
            covar_factors = torch.einsum(
                'i,j->ij',
                warped_mean,
                warped_mean
            )

        mean_x = self.alpha + warped_mean ** 2 / 2
        covar_x = warped_distribution.lazy_covariance_matrix * covar_factors
        return MultivariateNormal(mean_x, covar_x)


def build_model(
        warping: str = 'linearised_square_root',
        approximation: str = 'exact',
        mean_function: Mapping[str, str] = None,
        covariance_function: Mapping[str, Union[str, Mapping]] = None,
        likelihood: Mapping[str, str] = None,
        marginal_log_likelihood: Mapping[str, str] = None,
        train_inputs: Union[Tensor, Sequence[Tensor]] = None,
        train_targets: Union[Tensor, Sequence[Tensor]] = None,
        train_targets_log_offset: Tensor = None,
        dimensions: int = 1,
        sacred_run=None,
        **kwargs
) -> Sequence:
    if mean_function is None:
        mean_function = {'type': 'constant'}
    if covariance_function is None:
        covariance_function = {'type': 'energy_distance_mmd'}
    if likelihood is None:
        likelihood = {'type': 'gaussian'}
    if marginal_log_likelihood is None:
        marginal_log_likelihood = {'type': 'exact'}

    if covariance_function['type'] == 'energy_distance_mmd':
        covariance_module = InterDomainScaleKernel(EDMMDKernel(
            dimensions=dimensions
        ))
    else:
        raise NotImplementedError(
            f'{covariance_function["type"]} covariance function not implemented'
        )
    if mean_function['type'] == 'constant':
        mean_module = ConstantMean()
    else:
        raise NotImplementedError(
            f'{mean_function["type"]} mean function not implemented'
        )
    if likelihood['type'] == 'gaussian':
        likelihood_module = GaussianLikelihood()
    else:
        raise NotImplementedError(
            f'{likelihood["type"]} likelihood not implemented'
        )
    if marginal_log_likelihood['type'] == 'exact':
        mll_handle = ExactMarginalLogLikelihood
    else:
        raise NotImplementedError(
            f'{marginal_log_likelihood["type"]} mll not implemented'
        )

    if approximation == 'exact':
        if warping == 'linearised_square_root':
            model = InterDomainWsabiLGPModel(
                train_x=train_inputs,
                train_y=train_targets,
                train_targets_log_offset=train_targets_log_offset,
                likelihood=likelihood_module,
                mean_module=mean_module,
                covar_module=covariance_module,
                alpha_factor=kwargs.get('alpha_factor', 0.8)
            )
            mll_module = mll_handle(model.likelihood, model)
            return {'model': model, 'mll': mll_module}
    else:
        raise NotImplementedError(
            f'{approximation} approximation not implemented'
        )
