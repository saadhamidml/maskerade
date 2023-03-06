"""Define models for modelling the data."""
from typing import Sequence, Mapping, Union
from copy import copy
import math
from gpytorch.means.linear_mean import LinearMean
import torch
from torch import Tensor, Size
from gpytorch.models import ExactGP
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.means import Mean, ConstantMean
from gpytorch.kernels import (
    Kernel,
    ScaleKernel,
    RBFKernel,
    MaternKernel
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import Prior, SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal
from spectralgp.models import SpectralModel, ProductKernelSpectralModel

from .priors import build_prior, automatic_prior_specification
from .kernels import (
    SpectralMixtureKernel,
    KeOpsSpectralMixtureKernel,
    BayesianSpectralMixtureKernel,
    KeOpsBayesianSpectralMixtureKernel
)
from utils import nyquist_frequencies

from models.gaussian_process import priors


class ExactGPModel(ExactGP):
    """Vanilla GP with additional methods for working with
    hyperparameters concatenated into one tensor, rather than as stored
    as properties of submodules.
    """
    def __init__(
            self,
            train_inputs: Tensor,
            train_targets: Tensor,
            likelihood: Likelihood,
            mean_module: Mean,
            covar_module: Kernel
    ):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module
        try:
            self.covar_module.initialize_from_data(train_inputs, train_targets)
        except (AttributeError, IndexError):
            # AttributeError if not Spectral Mixture Kernel
            # IndexError if no training data
            pass

        self.batched = False

    @property
    def hyperparameters_as_tensor(self) -> torch.Tensor:
        """Hyperparameters of all GPs as a tensor.

        Returns tensor of shape (num_gps x num_hyperparameters).
        """
        hyperparameters = []
        for _, _, closure, _ in self.named_priors():
            clos = closure()
            hp = clos.squeeze()
            if self.batched and hp.ndimension() < 2:
                batch_shape = clos.size(0)
                hp = hp.view(batch_shape, -1)
            elif not self.batched:
                hp = hp.view(1, -1)
            hyperparameters.append(hp)
        return torch.cat(hyperparameters, dim=1)

    @hyperparameters_as_tensor.setter
    def hyperparameters_as_tensor(self, tensor: torch.Tensor):
        """Set hyperparameters from tensor of shape (num_gps x
        num_hyperparameters).
        """
        if tensor.ndimension() == 1:
            start_index = 0
            for *_, closure, setting_closure in self.named_priors():
                clos = closure().squeeze()
                # n_dims = clos.size(0) if clos.ndimension() > 0 else 1
                n_dims = clos.numel()
                end_index = start_index + n_dims
                setting_closure(tensor[start_index:end_index])
                start_index = end_index
        else:
            hyperparameter_dict = {}
            start_index = 0
            for name, _, closure, _ in self.named_priors():
                clos = closure().squeeze()
                # if clos.ndimension() > 1:
                #     n_dims = clos.size(1)
                # elif clos.ndimension() > 0:
                #     n_dims = clos.size(0)
                # else:
                #     n_dims = 1
                n_dims = clos.numel()
                end_index = start_index + n_dims
                hyp = tensor[:, start_index:end_index]
                hyperparameter_dict[name] = hyp.view(-1, *closure().size())
                start_index = end_index
            self.pyro_load_from_samples(hyperparameter_dict)
            self.batched = True

    @property
    def hyperparameter_priors(self) -> Sequence[Prior]:
        """Get tuple of priors over all hyperparameters."""
        priors = []
        for _, prior, _, _ in self.named_priors():
            priors.append(prior)
        return tuple(priors)

    def sample_hyperparameters(
            self, n_samples: int = 1, low_discrepancy: bool = False
        ) -> torch.Tensor:
        """Draw n_samples samples from priors of all hyperparameters.

        Returns tensor of shape (num_samples x num_hyperparameters).
        """
        samples = []
        for _, prior, _, _ in self.named_priors():
            if low_discrepancy:
                sample = prior.low_discrepancy_sample((n_samples,))
            else:
                sample = prior.sample((n_samples,))
            sample = self.postprocess_hyperparameter_sample(sample, n_samples)
            samples.append(sample)
        samples = torch.cat(samples, dim=1).squeeze()
        return samples

    @staticmethod
    def postprocess_hyperparameter_sample(sample, n_samples):
        # sample = sample.squeeze()
        # n_dims = sample.ndimension()
        # if n_dims < 2:
        #     if n_samples > 1:
        #         sample = sample.view(-1, 1)
        #     else:
        #         sample = sample.view(1, -1)
        return sample.view(n_samples, -1)

    def forward(self, x):
        """Compute posterior at x for each GP in model (if batch of
        GPs).
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        if (
                hasattr(self.covar_module, 'n_random_fourier_features')
                and self.covar_module.n_random_fourier_features is not None
        ):
            if self.training:
                raise RuntimeError('Should not call model directly during training')
            predictands = args[0]
            design_matrix = self.covar_module.random_fourier_features(
                self.train_inputs[0]
            )
            feature_ips = (
                design_matrix
                @ design_matrix.transpose(0, 1)
                + self.covar_module.n_random_fourier_features
                * self.likelihood.noise
                * torch.eye(2 * self.covar_module.n_random_fourier_features)
            )
            fip_chol = torch.cholesky(feature_ips)
            feature_target = (
                design_matrix @ self.train_targets
            ).unsqueeze_(1)

            predictand_features = self.covar_module.random_fourier_features(
                predictands
            )
            mean = (
                predictand_features.transpose(0, 1)
                @ torch.cholesky_solve(feature_target, fip_chol)
            ).squeeze()

            variance = self.likelihood.noise * (
                torch.sum(
                    predictand_features
                    * torch.cholesky_solve(predictand_features, fip_chol),
                    dim=0
                )
                + 1
            )
            return mean, variance

            # covariance = self.likelihood.noise * (
            #     predictand_features.transpose(0, 1)
            #     @ torch.cholesky_solve(predictand_features, fip_chol)
            #     + torch.eye(predictands.size(0))
            # )
            # return MultivariateNormal(mean, covariance)
        else:
            return super().__call__(*args, **kwargs)


class GPModelCollection:
    def __init__(
            self,
            approximation: str = 'exact',
            mean_function: Mapping[str, str] = None,
            covariance_function: Mapping[str, Union[str, Mapping]] = None,
            likelihood: Mapping[str, str] = None,
            marginal_log_likelihood: Mapping[str, str] = None,
            train_inputs: Tensor = None,
            train_targets: Tensor = None,
            output_scale: float = None,
            use_cuda: bool = False
    ):
        self.approximation = approximation
        self.mean_function = mean_function
        self.covariance_function = covariance_function
        self.likelihood = likelihood
        self.marginal_log_likelihood = marginal_log_likelihood
        self.num_mixtures = self.covariance_function['num_mixtures']
        self.n_random_fourier_features = self.covariance_function.get(
            'n_random_fourier_features', None
        )
        self.n_models = len(self.num_mixtures)
        self.train_inputs = train_inputs
        self.nyquist_frequencies = nyquist_frequencies(self.train_inputs)
        self.train_targets = train_targets
        if output_scale is not None:
            self.output_scale = output_scale
        else:
            self.output_scale = torch.std(train_targets)
        self.use_cuda = use_cuda
        self.training = True
        # self.model_evidences = torch.ones(n_models).to(self.train_inputs)
        # self.model_evidence_covariances = torch.ones(
        #     n_models, n_models
        # ).to(self.train_inputs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def create_model(
            self, model_index: int, hyperparameters: Tensor = None
    ) -> Mapping:
        if hyperparameters is not None:
            if hyperparameters.ndimension() < 2:
                batch_size = 1
            else:
                batch_size = hyperparameters.size(0)
        else:
            batch_size = None
        try:
            covariance_function = copy(
                self.covariance_function['prior']['priors'][model_index]
            )
        except TypeError:
            covariance_function = automatic_prior_specification(
                dimensions=(
                    self.train_inputs.size(1)
                    if self.train_inputs.ndimension() > 1 else 1
                ),
                n_mixtures=self.covariance_function['num_mixtures'][
                    model_index
                ],
                nyquist_frequencies=self.nyquist_frequencies
            )
        covariance_function['type'] = self.covariance_function['type']
        covariance_function['num_mixtures'] = self.covariance_function[
            'num_mixtures'
        ][model_index]
        covariance_function[
            'n_random_fourier_features'
        ] = self.n_random_fourier_features
        model_components = build_model(
            collection=False,
            approximation=self.approximation,
            mean_function=self.mean_function,
            covariance_function=covariance_function,
            likelihood=self.likelihood,
            marginal_log_likelihood=self.marginal_log_likelihood,
            train_inputs=self.train_inputs,
            train_targets=self.train_targets,
            output_scale=self.output_scale,
            batch_size=batch_size,
            hyperparameters=hyperparameters,
            use_cuda=self.use_cuda
        )
        for param in model_components['model'].parameters():
            param.requires_grad = False
        if not self.training:
            model_components['model'].eval()
        return model_components

    def create_models(self, hyperparameters: Sequence[Tensor], batch=True) -> Mapping:
        """If tensor is empty for a domain, the relevant model is not
        made.
        """
        model_list = []
        mll_list = []
        for model_index in range(self.n_models):
            if hyperparameters[model_index].size(0) != 0:
                if batch:
                    model_components = self.create_model(
                        model_index, hyperparameters[model_index]
                    )
                    model_list.append(model_components['model'])
                    mll_list.append(model_components['mll'])
                else:
                    for h in hyperparameters[model_index]:
                        model_components = self.create_model(model_index, h)
                        model_list.append(model_components['model'])
                        mll_list.append(model_components['mll'])
        for m in model_list:
            for param in m.parameters():
                param.requires_grad = False
            if not self.training:
                m.eval()
        return {'model': model_list, 'mll': tuple(mll_list)}

    @property
    def hyperparameter_priors(self) -> Sequence[Sequence[Prior]]:
        priors = []
        for model_index in range(self.n_models):
            try:
                p = self.covariance_function['prior']['priors'][model_index]
            except TypeError:
                p = automatic_prior_specification(
                    dimensions=(
                        self.train_inputs.size(1)
                        if self.train_inputs.ndimension() > 1 else 1
                    ),
                    n_mixtures=self.covariance_function['num_mixtures'][
                        model_index
                    ],
                    nyquist_frequencies=self.nyquist_frequencies
                )
            device = self.train_inputs.device
            p_weights = build_prior(**p['weights'], device=device)
            p_means = build_prior(**p['means'], device=device)
            p_scales = build_prior(**p['scales'], device=device)
            priors.append([p_weights, p_means, p_scales])
        return priors

    def sample_hyperparameters(
            self, model_index: int,
            num_samples: int,
            low_discrepancy: bool = False
    ) -> Sequence[Tensor]:
        model_components = self.create_model(model_index)
        model = model_components['model']
        theta = model.sample_hyperparameters(
            num_samples, low_discrepancy=low_discrepancy
        )
        del model
        del model_components
        mll = self.compute_likelihood(model_index, theta)
        # model.hyperparameters_as_tensor = theta
        # with torch.no_grad():
        #     mll = model.likelihood(
        #         model(model.train_inputs[0])
        #     ).log_prob(model.train_targets)
        return theta, mll

    def compute_likelihood(
            self,
            model_index,
            hyperparameters,
            covariance_decomposition_cache=None
    ) -> Tensor:
        model_components = self.create_model(model_index, hyperparameters)
        model = model_components['model']
        if covariance_decomposition_cache is not None:
            raise NotImplementedError(
                'Cached covariance decompositions not implemented'
            )
        with torch.no_grad():
            if self.n_random_fourier_features is not None:
                design_matrix = model.covar_module.random_fourier_features(
                    model.train_inputs[0]
                )
                feature_ips = (
                    design_matrix
                    @ design_matrix.transpose(0, 1)
                    + self.n_random_fourier_features
                    * model.likelihood.noise
                    * torch.eye(2 * self.n_random_fourier_features)
                )
                fip_chol = torch.cholesky(feature_ips)
                feature_target = (
                    design_matrix @ self.train_targets
                ).unsqueeze_(1)

                t1 = -(
                    self.train_targets
                    @ self.train_targets
                    - feature_target.transpose(0, 1)
                    @ torch.cholesky_solve(feature_target, fip_chol)
                ) / (2 * model.likelihood.noise)
                t2 = -fip_chol.diag().log().sum(-1)
                t3 = (
                    self.n_random_fourier_features * torch.log(
                        self.n_random_fourier_features * model.likelihood.noise
                    )
                    - 0.5 * self.train_targets.nelement() * torch.log(
                        2 * math.pi * model.likelihood.noise
                    )
                )
                marginal_log_likelihood = (t1 + t2 + t3).squeeze()
            else:
                marginal_log_likelihood = model.likelihood(model(
                    model.train_inputs[0])
                ).log_prob(model.train_targets)
        return marginal_log_likelihood

    def __call__(
            self,
            hyperparameters: Sequence[Tensor],
            test_inputs: Tensor,
            covariance_decomposition_cache=None,
            return_kernel=False,
            batch=True
    ) -> Sequence[MultivariateNormal]:
        model_components = self.create_models(hyperparameters, batch=batch)
        model_list = model_components['model']
        # Assume all model outputs desired, unless specified.
        # Output is a (list of) batch(es) of multivariate normals.
        if covariance_decomposition_cache is not None:
            # TODO: Use cached decompositions
            raise NotImplementedError(
                'Cached covariance decompositions not implemented'
            )
        elif return_kernel:
            kernels = []
            predictions = []
            for m in model_list:
                predictions.append(m.likelihood(m(test_inputs)))
                batch_shape = m.train_inputs[0].size()[:-2]
                kernels.append(
                    m.covar_module(
                        torch.zeros(*batch_shape, 1, 1),
                        test_inputs.view(1, -1, 1).repeat(*batch_shape, 1, 1)
                    ).evaluate().view(-1, test_inputs.size(0))
                )
            kernels = torch.cat(kernels, dim=0)
            return predictions, kernels
        else:
            if self.n_random_fourier_features is None:
                predictions = []
                for m in model_list:
                    predictions.append(m.likelihood(m(test_inputs)))
                return predictions
            else:
                predictions = []
                for m in model_list:
                    predictions.append(m(test_inputs))
                return predictions


def build_model(
        collection: bool = False,
        approximation: str = 'exact',
        mean_function: Mapping[str, str] = None,
        covariance_function: Mapping[str, Union[str, Mapping]] = None,
        likelihood: Mapping[str, str] = None,
        marginal_log_likelihood: Mapping[str, str] = None,
        train_inputs: Tensor = None,
        train_targets: Tensor = None,
        output_scale: Tensor = None,
        batch_size: Size = None,
        hyperparameters: Tensor = None,
        use_cuda: bool = False,
        sacred_run=None,
        **kwargs
) -> Sequence:
    dimensions = train_inputs.size(1) if train_inputs.ndimension() > 1 else 1

    if mean_function is None:
        mean_function = {'type': 'constant'}
    if covariance_function is None:
        covariance_function = {'type': 'gaussian_spectral_mixture'}
    if likelihood is None:
        likelihood = {'type': 'gaussian'}
    if marginal_log_likelihood is None:
        marginal_log_likelihood = {'type': 'exact'}

    if collection:
        model = GPModelCollection(
            approximation=approximation,
            mean_function=mean_function,
            covariance_function=covariance_function,
            likelihood=likelihood,
            marginal_log_likelihood=marginal_log_likelihood,
            train_inputs=train_inputs,
            train_targets=train_targets,
            output_scale=output_scale,
        )
        return {'model': model, 'mll': None}

    if covariance_function['type'] == 'bayesian_gaussian_spectral_mixture':
        device = train_inputs.device
        mixture_weights_prior = build_prior(
            **covariance_function['weights'], device=device
        )
        mixture_means_prior = build_prior(
            **covariance_function['means'], device=device
        )
        mixture_scales_prior = build_prior(
            **covariance_function['scales'], device=device
        )
        covariance_module = BayesianSpectralMixtureKernel(
            num_mixtures=covariance_function['num_mixtures'],
            n_random_fourier_features=covariance_function.get(
                'n_random_fourier_features', None
            ),
            ard_num_dims=dimensions,
            mixture_weights_prior=mixture_weights_prior,
            mixture_means_prior=mixture_means_prior,
            mixture_scales_prior=mixture_scales_prior,
            output_scale=output_scale
        )
    elif covariance_function['type'] == 'keops_bayesian_gaussian_spectral_mixture':
        device = train_inputs.device
        mixture_weights_prior = build_prior(
            **covariance_function['weights'], device=device
        )
        mixture_means_prior = build_prior(
            **covariance_function['means'], device=device
        )
        mixture_scales_prior = build_prior(
            **covariance_function['scales'], device=device
        )
        covariance_module = KeOpsBayesianSpectralMixtureKernel(
            num_mixtures=covariance_function['num_mixtures'],
            ard_num_dims=dimensions,
            mixture_weights_prior=mixture_weights_prior,
            mixture_means_prior=mixture_means_prior,
            mixture_scales_prior=mixture_scales_prior,
            output_scale = output_scale
        )
    elif covariance_function['type'] == 'spectral_gaussian_process':
        data_lh = GaussianLikelihood(SmoothedBoxPrior(1e-8, 1e-3))
        if dimensions > 1:
            data_mod = ProductKernelSpectralModel(
                train_inputs,
                train_targets,
                data_lh,
                normalize=False,
                symmetrize=False,
                shared=True,
                num_locs=covariance_function['n_frequencies'],
                spacing=covariance_function['spacing']
            )
        else:
            data_mod = SpectralModel(
                train_inputs,
                train_targets,
                data_lh,
                normalize=False,
                symmetrize=False,
                num_locs=covariance_function['n_frequencies'],
                spacing=covariance_function['spacing']
            )
        # data_lh.raw_noise = torch.tensor(-3.5)
        data_lh.noise = 1e-3
        return {'model': data_mod, 'mll': None}
    elif covariance_function['type'] == 'keops_gaussian_spectral_mixture':
        covariance_module = KeOpsSpectralMixtureKernel(
            num_mixtures=covariance_function['num_mixtures'],
            ard_num_dims=dimensions
        )
    elif covariance_function['type'] == 'gaussian_spectral_mixture':
        covariance_module = SpectralMixtureKernel(
            num_mixtures=covariance_function['num_mixtures'],
            ard_num_dims=dimensions
        )
    elif covariance_function['type'] == 'radial_basis_function':
        covariance_module = ScaleKernel(RBFKernel())
        covariance_module.outputscale = covariance_function['initialisation'][
            'outputscale'
        ]
        covariance_module.base_kernel.lengthscale = covariance_function[
            'initialisation'
        ]['lengthscale']
    elif covariance_function['type'] == 'matern':
        covariance_module = ScaleKernel(
            MaternKernel(nu=covariance_function['nu'])
        )
        covariance_module.outputscale = covariance_function['initialisation'][
            'outputscale'
        ]
        covariance_module.base_kernel.lengthscale = covariance_function[
            'initialisation'
        ]['lengthscale']
    else:
        raise NotImplementedError(
            f'{covariance_function["type"]} covariance function not implemented'
        )

    if mean_function['type'] == 'constant':
        mean_module = ConstantMean()
    elif mean_function['type'] == 'linear':
        mean_module = LinearMean(1)
        mean_module.bias.data = torch.tensor([-3.])
        mean_module.weights.data = torch.tensor([[8.]])
    else:
        raise NotImplementedError(
            f'{mean_function["type"]} mean function not implemented'
        )

    if likelihood['type'] == 'gaussian':
        likelihood_module = GaussianLikelihood()
        likelihood_module.noise = likelihood.get('noise', 1e-3)
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
        model = ExactGPModel(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood_module,
            mean_module=mean_module,
            covar_module=covariance_module,
        )
        if hyperparameters is not None:
            model.batch_size = batch_size
            model.hyperparameters_as_tensor = hyperparameters
        if use_cuda:
            model = model.cuda()
            model.likelihood = model.likelihood.cuda()
        mll_module = mll_handle(model.likelihood, model)
        return {'model': model, 'mll': mll_module}
    else:
        raise NotImplementedError(
            f'{approximation} approximation not implemented'
        )
