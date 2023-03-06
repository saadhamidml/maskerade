"""Extends GPyTorch Kernels."""
from typing import Union, Tuple
import logging
from multimethod import multimethod
import math
import torch
from torch import Tensor
from torch.distributions import Categorical, Normal
from gpytorch.kernels import Kernel, SpectralMixtureKernel as SMKernel
from gpytorch.kernels.keops.keops_kernel import KeOpsKernel
from gpytorch.lazy.keops_lazy_tensor import KeOpsLazyTensor
from gpytorch.constraints import GreaterThan
from gpytorch.priors import Prior, UniformPrior, NormalPrior
from gpytorch.constraints import Positive
from pykeops.torch import LazyTensor as KEOLazyTensor
from sklearn.mixture import GaussianMixture

from .priors import DirichletPrior, TruncatedNormalPrior, LogNormalPrior


logger = logging.getLogger()


class SpectralMixtureKernel(SMKernel):
    def __init__(self, n_random_fourier_features=None, **kwargs):
        super(SpectralMixtureKernel, self).__init__(**kwargs)
        self.n_random_fourier_features = n_random_fourier_features

    def random_fourier_features(
            self, x: Tensor, n_features: int = None
    ) -> Tensor:
        """Draw random samples from the GMM defined by the current
        hyperparameters and return the corresponding feature matrices.
        """
        if len(self.batch_shape) != 0:
            raise NotImplementedError
        if n_features is None:
            n_features = self.n_random_fourier_features
        if x.ndimension() == 1:
            x = x.unsqueeze(1)
        # First sample from mixture weights to determine which
        # Gaussian to take samples from.
        weights_samples = Categorical(self.mixture_weights).sample(
            (n_features,)
        )
        mixture_numbers, counts = torch.unique(
            weights_samples, return_counts=True
        )
        # Now sample from each Gaussian.
        features = []
        for mixture_number, n_samples in zip(mixture_numbers, counts):
            distribution = Normal(
                loc=self.mixture_means[mixture_number].squeeze(),
                scale=self.mixture_scales[mixture_number].squeeze()
            )
            features.append(distribution.sample((n_samples,)))
        # Concatenate samples into single tensor.
        features = torch.cat(features, dim=0).view(n_features, -1)
        # Apply features to each element in the input tensor.
        arguments = 2 * math.pi * features @ x.transpose(0, 1)
        design_matrix = torch.cat(
            (torch.cos(arguments), torch.sin(arguments)),
            dim=0
        )
        return design_matrix

    def feature_inner_product(self, x1: Tensor, x2: Tensor = None) -> Tensor:
        """Inner product in feature space."""
        f1 = self.random_fourier_features(x1)
        if x2 is not None and not torch.allclose(x1, x2):
            f2 = self.random_fourier_features(x2)
        else:
            f2 = f1
        return f1 @ f2.transpose(0, 1)


class KeOpsSpectralMixtureKernel(KeOpsKernel):
    """SpectralMixtureKernel implemented with PyKeOPs."""

    is_stationary = True  # kernel is stationary even though it does not have a lengthscale

    def __init__(
        self,
        num_mixtures=None,
        ard_num_dims=1,
        batch_shape=torch.Size([]),
        mixture_scales_prior=None,
        mixture_scales_constraint=None,
        mixture_means_prior=None,
        mixture_means_constraint=None,
        mixture_weights_prior=None,
        mixture_weights_constraint=None,
        **kwargs,
    ):
        if num_mixtures is None:
            raise RuntimeError("num_mixtures is a required argument")
        if mixture_means_prior is not None or mixture_scales_prior is not None or mixture_weights_prior is not None:
            logger.warning("Priors not implemented for SpectralMixtureKernel")

        # This kernel does not use the default lengthscale
        super(KeOpsSpectralMixtureKernel, self).__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)
        self.num_mixtures = num_mixtures

        if mixture_scales_constraint is None:
            mixture_scales_constraint = Positive()

        if mixture_means_constraint is None:
            mixture_means_constraint = Positive()

        if mixture_weights_constraint is None:
            mixture_weights_constraint = Positive()

        self.register_parameter(
            name="raw_mixture_weights", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.num_mixtures))
        )
        ms_shape = torch.Size([*self.batch_shape, self.num_mixtures, 1, self.ard_num_dims])
        self.register_parameter(name="raw_mixture_means", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))
        self.register_parameter(name="raw_mixture_scales", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))

        self.register_constraint("raw_mixture_scales", mixture_scales_constraint)
        self.register_constraint("raw_mixture_means", mixture_means_constraint)
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)

    @property
    def mixture_scales(self):
        return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)

    @mixture_scales.setter
    def mixture_scales(self, value):
        self._set_mixture_scales(value)

    def _set_mixture_scales(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        return self.raw_mixture_means_constraint.transform(self.raw_mixture_means)

    @mixture_means.setter
    def mixture_means(self, value):
        self._set_mixture_means(value)

    def _set_mixture_means(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_means)
        self.initialize(raw_mixture_means=self.raw_mixture_means_constraint.inverse_transform(value))

    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)

    @mixture_weights.setter
    def mixture_weights(self, value):
        self._set_mixture_weights(value)

    def _set_mixture_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))

    def initialize_from_data_empspect(self, train_x, train_y):
        """
        Initialize mixture components based on the empirical spectrum of the data.

        This will often be better than the standard initialize_from_data method.
        """
        import numpy as np
        from scipy.fftpack import fft
        from scipy.integrate import cumtrapz

        N = train_x.size(-2)
        emp_spect = np.abs(fft(train_y.cpu().detach().numpy())) ** 2 / N
        M = math.floor(N / 2)

        freq1 = np.arange(M + 1)
        freq2 = np.arange(-M + 1, 0)
        freq = np.hstack((freq1, freq2)) / N
        freq = freq[: M + 1]
        emp_spect = emp_spect[: M + 1]

        total_area = np.trapz(emp_spect, freq)
        spec_cdf = np.hstack((np.zeros(1), cumtrapz(emp_spect, freq)))
        spec_cdf = spec_cdf / total_area

        a = np.random.rand(1000, self.ard_num_dims)
        p, q = np.histogram(a, spec_cdf)
        bins = np.digitize(a, q)
        slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freq[bins] - freq[bins - 1])
        intercepts = spec_cdf[bins - 1] - slopes * freq[bins - 1]
        inv_spec = (a - intercepts) / slopes

        from sklearn.mixture import GaussianMixture

        GMM = GaussianMixture(n_components=self.num_mixtures, covariance_type="diag").fit(inv_spec)
        means = GMM.means_
        varz = GMM.covariances_
        weights = GMM.weights_

        self.mixture_means = means
        self.mixture_scales = varz
        self.mixture_weights = weights

    def initialize_from_data(self, train_x, train_y, **kwargs):
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_x.ndimension() == 2:
            train_x = train_x.unsqueeze(0)

        train_x_sort = train_x.sort(1)[0]
        max_dist = train_x_sort[:, -1, :] - train_x_sort[:, 0, :]
        min_dist_sort = (train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]).squeeze(0)
        min_dist = torch.zeros(1, self.ard_num_dims, dtype=train_x.dtype, device=train_x.device)
        for ind in range(self.ard_num_dims):
            min_dist[:, ind] = min_dist_sort[((min_dist_sort[:, ind]).nonzero())[0], ind]

        # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
        self.raw_mixture_scales.data.normal_().mul_(max_dist).abs_().pow_(-1)
        self.raw_mixture_scales.data = self.raw_mixture_scales_constraint.inverse_transform(
            self.raw_mixture_scales.data
        )
        # Draw means from Unif(0, 0.5 / minimum distance between two points)
        self.raw_mixture_means.data.uniform_().mul_(0.5).div_(min_dist)
        self.raw_mixture_means.data = self.raw_mixture_means_constraint.inverse_transform(self.raw_mixture_means.data)
        # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
        self.raw_mixture_weights.data.fill_(train_y.std() / self.num_mixtures)
        self.raw_mixture_weights.data = self.raw_mixture_weights_constraint.inverse_transform(
            self.raw_mixture_weights.data
        )

    def _create_input_grid(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.

        The grid dimensions depend on the kernel's evaluation mode.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`)
            :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the gridded `x1` and `x2`.
            The shape depends on the kernel's mode

            * `full_covar`: (`b x n x 1 x d` and `b x 1 x m x d`)
            * `full_covar` with `last_dim_is_batch=True`: (`b x k x n x 1 x 1` and `b x k x 1 x m x 1`)
            * `diag`: (`b x n x d` and `b x n x d`)
            * `diag` with `last_dim_is_batch=True`: (`b x k x n x 1` and `b x k x n x 1`)
        """
        x1_, x2_ = x1, x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            if torch.equal(x1, x2):
                x2_ = x1_
            else:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        if diag:
            return x1_, x2_
        else:
            return x1_.unsqueeze(-2), x2_.unsqueeze(-3)

    def covar_func(
            self, x1: torch.Tensor, x2: torch.Tensor, diag=False
    ) -> Union[KEOLazyTensor, Tensor]:
        batch_shape = x1.shape[:-2]
        if diag or x1.size(-2) == 1 or x2.size(-2) == 1:
            # Expand x1 and x2 to account for the number of mixtures
            # Should make x1/x2 (b x k x n x d) for k mixtures
            x1_ = x1.unsqueeze(len(batch_shape))
            x2_ = x2.unsqueeze(len(batch_shape))

            # Compute distances
            if diag:
                mixture_means = self.mixture_means
                mixture_scales = self.mixture_scales
            else:
                x1_.unsqueeze_(-2)
                x2_.unsqueeze_(-3)
                mixture_means = self.mixture_means.unsqueeze(-2)
                mixture_scales = self.mixture_scales.unsqueeze(-2)
            diff = x1_ - x2_

            # Compute the exponential and cosine terms
            exp_term = diff.mul(mixture_scales).pow_(2).mul_(
                -2 * math.pi ** 2
            )
            cos_term = diff.mul(mixture_means).mul_(2 * math.pi)
            res = exp_term.exp_().prod(-1) * cos_term.sum(-1).cos_()

            # Sum over mixtures
            mixture_weights = self.mixture_weights

            while mixture_weights.dim() < res.dim():
                mixture_weights = mixture_weights.unsqueeze(-1)

            res = (res * mixture_weights).sum(len(batch_shape))
            return res
        with torch.autograd.enable_grad():
            x1_ = KEOLazyTensor(x1[..., :, None, :])
            x2_ = KEOLazyTensor(x2[..., None, :, :])
            mixture_weights = self.mixture_weights.view(
                *batch_shape, 1, 1, self.num_mixtures, 1
            )
            mixture_means = self.mixture_means.view(
                *batch_shape, 1, 1, self.num_mixtures, self.ard_num_dims
            )
            mixture_scales = self.mixture_scales.view(
                *batch_shape, 1, 1, self.num_mixtures, self.ard_num_dims
            )

            # B x N1 x N2 x D
            differences = x1_ - x2_
            res = 0
            for m in range(self.num_mixtures):
                # Reduce over D
                exp_term = (
                    differences ** 2
                    * (mixture_scales[..., m, :] ** 2)
                    * (-2 * math.pi ** 2)
                ).sum(dim=-1).exp()
                cos_term = (
                    (
                        differences * mixture_means[..., m, :] * (2 * math.pi)
                    ).sum(dim=-1)
                ).cos()

                # Reduce over M
                res += mixture_weights[..., m, :] * cos_term * exp_term

            # B x N1 x N2
            return res

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        batch_shape = x1.shape[:-2]
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, num_dims)
            )
        if not batch_shape == self.batch_shape:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have a batch_size of {} "
                "(based on the batch_size argument). Got {}.".format(self.batch_shape, batch_shape)
            )

        if diag:
            return self.covar_func(x1, x2, diag=True)

        covar_func = lambda x1_, x2_, diag=False: self.covar_func(x1_, x2_, diag)
        # The multiplication is necessary so that the output type is a
        # ConstantMulLazyTensor. This is necessary or GPyTorch throws an error
        # about not doing a cholesky decomposition for with KeOps. Likely
        # one should avoid this anyway by only using KeOps for sufficiently
        # large matrices.
        return KeOpsLazyTensor(x1, x2, covar_func)#.mul(torch.tensor(1.))


class BayesianSpectralMixtureKernel(SpectralMixtureKernel):
    """Extension of SpectralMixtureKernel to allow priors on the
    hyperparameters.

    There must be priors over all hyperparameters, priors over only some
    are not supported.
    """
    def __init__(
            self,
            num_mixtures=1,
            ard_num_dims=1,
            mixture_weights_prior=None,
            mixture_means_prior=None,
            mixture_scales_prior=None,
            output_scale=None,
            **kwargs
    ):
        super(BayesianSpectralMixtureKernel, self).__init__(
            num_mixtures=num_mixtures,
            ard_num_dims=ard_num_dims,
            mixture_scales_constraint=GreaterThan(torch.tensor(1e-9)),
            mixture_means_constraint=GreaterThan(torch.tensor(1e-9)),
            mixture_weights_constraint=GreaterThan(torch.tensor(1e-9)),
            **kwargs
        )
        if mixture_weights_prior is not None:
            self.register_prior(
                "mixture_weights_prior",
                mixture_weights_prior,
                lambda: self.mixture_weights,
                lambda v: self._set_mixture_weights(v)
            )
        if mixture_means_prior is not None:
            _expand_prior(mixture_means_prior, ard_num_dims)
            self.register_prior(
                "mixture_means_prior",
                mixture_means_prior,
                lambda: self.mixture_means,
                lambda v: self._set_mixture_means(torch.abs(v).clamp_min(1e-9))
            )
        if mixture_scales_prior is not None:
            _expand_prior(mixture_scales_prior, ard_num_dims)
            self.register_prior(
                "mixture_scales_prior",
                mixture_scales_prior,
                lambda: self.mixture_scales,
                lambda v: self._set_mixture_scales(v.clamp_min(1e-9))
            )
        self.output_scale = output_scale

    def initialize_from_computed_data_empspect(self, inv_spec):
        gmm = GaussianMixture(
            n_components=self.num_mixtures, covariance_type="diag"
        ).fit(inv_spec)
        weights = gmm.weights_
        means = gmm.means_
        varz = gmm.covariances_

        self.mixture_weights = weights
        self.mixture_means = means
        self.mixture_scales = varz

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super(BayesianSpectralMixtureKernel, self).forward(x1, x2, diag, last_dim_is_batch, **params) \
               * self.output_scale


class KeOpsBayesianSpectralMixtureKernel(KeOpsSpectralMixtureKernel):
    """Extension of SpectralMixtureKernel to allow priors on the
    hyperparameters.

    There must be priors over all hyperparameters, priors over only some
    are not supported.
    """
    def __init__(
            self,
            num_mixtures=1,
            ard_num_dims=1,
            mixture_weights_prior=None,
            mixture_means_prior=None,
            mixture_scales_prior=None,
            output_scale=None,
            **kwargs
    ):
        super(KeOpsBayesianSpectralMixtureKernel, self).__init__(
            num_mixtures=num_mixtures,
            ard_num_dims=ard_num_dims,
            mixture_scales_constraint=GreaterThan(torch.tensor(1e-9)),
            mixture_means_constraint=GreaterThan(torch.tensor(1e-9)),
            mixture_weights_constraint=GreaterThan(torch.tensor(1e-9)),
            **kwargs
        )
        if mixture_weights_prior is not None:
            self.register_prior(
                "mixture_weights_prior",
                mixture_weights_prior,
                lambda: self.mixture_weights,
                lambda v: self._set_mixture_weights(v)
            )
        if mixture_means_prior is not None:
            _expand_prior(mixture_means_prior, ard_num_dims)
            self.register_prior(
                "mixture_means_prior",
                mixture_means_prior,
                lambda: self.mixture_means,
                lambda v: self._set_mixture_means(torch.abs(v).clamp_min(1e-9))
            )
        if mixture_scales_prior is not None:
            _expand_prior(mixture_scales_prior, ard_num_dims)
            self.register_prior(
                "mixture_scales_prior",
                mixture_scales_prior,
                lambda: self.mixture_scales,
                lambda v: self._set_mixture_scales(v.clamp_min(1e-9))
            )
        self.output_scale = output_scale

    def initialize_from_computed_data_empspect(self, inv_spec):
        gmm = GaussianMixture(
            n_components=self.num_mixtures, covariance_type="diag"
        ).fit(inv_spec)
        weights = gmm.weights_
        means = gmm.means_
        varz = gmm.covariances_

        self.mixture_weights = weights
        self.mixture_means = means
        self.mixture_scales = varz

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super(KeOpsBayesianSpectralMixtureKernel, self).forward(x1, x2, diag, last_dim_is_batch, **params) \
               * self.output_scale


class FullSpectralMixtureKernel(SMKernel):
    is_stationary = True  # kernel is stationary even though it does not have a lengthscale

    def __init__(
            self,
            num_mixtures=None,
            ard_num_dims=1,
            batch_shape=torch.Size([]),
            mixture_scales_prior=None,
            mixture_means_prior=None,
            mixture_means_constraint=None,
            mixture_weights_prior=None,
            mixture_weights_constraint=None,
            **kwargs,
    ):
        if num_mixtures is None:
            raise RuntimeError("num_mixtures is a required argument")

        # This kernel does not use the default lengthscale
        super(SpectralMixtureKernel, self).__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            **kwargs
        )
        self.num_mixtures = num_mixtures

        if mixture_means_constraint is None:
            mixture_means_constraint = Positive()

        if mixture_weights_constraint is None:
            mixture_weights_constraint = Positive()

        self.register_parameter(
            name="raw_mixture_weights",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.num_mixtures))
        )
        means_shape = torch.Size([*self.batch_shape, self.num_mixtures, 1, self.ard_num_dims])
        self.register_parameter(name="raw_mixture_means",
                                parameter=torch.nn.Parameter(torch.zeros(means_shape)))
        scales_shape = torch.Size([*self.batch_shape, self.num_mixtures, 1, self.ard_num_dims * (self.ard_num_dims + 1) / 2])
        self.register_parameter(name="raw_mixture_cov_sqrts",
                                parameter=torch.nn.Parameter(torch.zeros(scales_shape)))

        self.register_constraint("raw_mixture_means", mixture_means_constraint)
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)

    @property
    def mixture_cov_sqrts(self):
        mcs = torch.zeros(*self.batch_shape, self.num_mixtures, 1, self.ard_num_dims, self.ard_num_dims)
        tril_indices = torch.tril_indices(row=self.ard_num_dims, col=self.ard_num_dims, offset=0)
        mcs[..., tril_indices[0], tril_indices[1]] = self.raw_mixture_cov_sqrts
        return mcs

    @mixture_cov_sqrts.setter
    def mixture_cov_sqrts(self, value):
        self._set_mixture_cov_sqrts(value)

    def _set_mixture_cov_sqrts(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_cov_sqrts)
        self.initialize(
            raw_mixture_cov_sqrts=self.raw_mixture_cov_sqrts_constraint.inverse_transform(value)
        )

    @property
    def mixture_scales(self):
        raise AttributeError('Use mixture_cov_sqrts instead.')

    @mixture_scales.setter
    def mixture_scales(self, value):
        raise AttributeError('Use mixture_cov_sqrts instead.')

    def _set_mixture_scales(self, value):
        raise AttributeError('Use mixture_cov_sqrts instead.')

    def random_fourier_features(
            self, x: Tensor, n_features: int = None
    ) -> Tensor:
        """Draw random samples from the GMM defined by the current
        hyperparameters and return the corresponding feature matrices.
        """
        if len(self.batch_shape) != 0:
            raise NotImplementedError
        if n_features is None:
            n_features = self.n_random_fourier_features
        if x.ndimension() == 1:
            x = x.unsqueeze(1)
        # First sample from mixture weights to determine which
        # Gaussian to take samples from.
        weights_samples = Categorical(self.mixture_weights).sample(
            (n_features,)
        )
        mixture_numbers, counts = torch.unique(
            weights_samples, return_counts=True
        )
        # Now sample from each Gaussian.
        features = []
        for mixture_number, n_samples in zip(mixture_numbers, counts):
            distribution = Normal(
                loc=self.mixture_means[mixture_number].squeeze(),
                scale=self.mixture_scales[mixture_number].squeeze()
            )
            features.append(distribution.sample((n_samples,)))
        # Concatenate samples into single tensor.
        features = torch.cat(features, dim=0).view(n_features, -1)
        # Apply features to each element in the input tensor.
        arguments = 2 * math.pi * features @ x.transpose(0, 1)
        design_matrix = torch.cat(
            (torch.cos(arguments), torch.sin(arguments)),
            dim=0
        )
        return design_matrix

    def feature_inner_product(self, x1: Tensor, x2: Tensor = None) -> Tensor:
        """Inner product in feature space."""
        f1 = self.random_fourier_features(x1)
        if x2 is not None and not torch.allclose(x1, x2):
            f2 = self.random_fourier_features(x2)
        else:
            f2 = f1
        return f1 @ f2.transpose(0, 1)

    def forward(self, x1, x2, last_dim_is_batch=False, **params):
        batch_shape = x1.shape[:-2]
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, num_dims)
            )
        if not batch_shape == self.batch_shape:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have a batch_size of {} "
                "(based on the batch_size argument). Got {}.".format(self.batch_shape, batch_shape)
            )

        # Expand x1 and x2 to account for the number of mixtures
        # [b, 1, n|m, d]
        x1_ = x1.unsqueeze(len(batch_shape))
        x2_ = x2.unsqueeze(len(batch_shape))

        # Compute distances - scaled by appropriate parameters
        # [b, 1, n|m, 1, d] @ [b, k, 1, d, d] -> [b, k, n|m, d]
        x1_exp = torch.matmul(x1_.unsqueeze(-2), self.mixture_cov_sqrts)
        x2_exp = torch.matmul(x2_.unsqueeze(-2), self.mixture_cov_sqrts)
        # [b, 1, n|m, d] * [b, k, 1, d] (and sum) -> [b, k, n|m]
        x1_cos = (x1_ * self.mixture_means).sum(dim=-1)
        x2_cos = (x2_ * self.mixture_means).sum(dim=-1)

        # Compute the exponential and cosine terms
        exp_term = (
            (x1_exp * x1_exp).sum(-1).unsqueeze_(-1)  # [b, k, n, 1]
            - 2 * torch.matmul(x1_exp, x2_exp.transpose(-1, -2))  # [b, k, n, m]
            + (x2_exp * x2_exp).sum(-1).unsqueeze_(-2)  # [b, k, 1, m]
        ).mul_(-2 * math.pi ** 2)  # [b, k, n, m]
        cos_term = (x1_cos.unsqueeze(-1) - x2_cos.unsqueeze(-2)).mul_(2 * math.pi)  # [b, k, n, m]
        res = exp_term.exp_() * cos_term.cos_()

        # Product over dimensions
        if last_dim_is_batch:
            res = res.squeeze(-1)

        # Sum over mixtures
        mixture_weights = self.mixture_weights

        if last_dim_is_batch:
            mixture_weights = mixture_weights.unsqueeze(-1)
        while mixture_weights.dim() < res.dim():
            mixture_weights = mixture_weights.unsqueeze(-1)

        res = (res * mixture_weights).sum(len(batch_shape))
        return res



@multimethod
def _expand_prior(prior: Prior, ard_num_dims: int = 1):
    """Alter GPyTorch Priors to add two dimensions to their samples.

    e.g. A prior that initially yields torch.Size([3]) is changed to
    yield torch.Size([3, 1, 1]).
    This is necessary due to the way GPyTorch internally handles
    hyperparameters. It is automatically accounted for except for priors
    over the scales and means of BayesianSpectralMixtureKernels.
    """
    raise NotImplementedError


@multimethod
def _expand_prior(prior: UniformPrior, ard_num_dims: int = 1):
    """Alter prior so that samples have two extra dimensions."""
    prior._batch_shape = torch.Size(
        (int(prior._batch_shape[0] / ard_num_dims), 1, ard_num_dims)
    )
    prior.low.resize_(*prior._batch_shape)
    prior.high.resize_(*prior._batch_shape)


@multimethod
def _expand_prior(prior: NormalPrior, ard_num_dims: int = 1):
    """Alter prior so that samples have two extra dimensions."""
    prior._batch_shape = torch.Size(
        (int(prior._batch_shape[0] / ard_num_dims), 1, ard_num_dims)
    )
    prior.loc.resize_(*prior._batch_shape)
    prior.scale.resize_(*prior._batch_shape)


@multimethod
def _expand_prior(
        prior: Union[LogNormalPrior, TruncatedNormalPrior],
        ard_num_dims: int = 1
    ):
    """Alter prior so that samples have two extra dimensions."""
    prior._batch_shape = torch.Size(
        (int(prior._batch_shape[0] / ard_num_dims), 1, ard_num_dims)
    )
    prior.base_dist._batch_shape = torch.Size(
        (int(prior.base_dist._batch_shape[0] / ard_num_dims), 1, ard_num_dims)
    )
    prior.loc.resize_(*prior.base_dist._batch_shape)
    prior.scale.resize_(*prior.base_dist._batch_shape)
