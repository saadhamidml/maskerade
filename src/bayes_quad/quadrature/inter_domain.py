from code import interact
from typing import Sequence, Tuple, Union
import warnings
import torch
from torch import Tensor
from gpytorch.priors import Prior
from gpytorch.distributions import MultivariateNormal

from .standard import IntegrandModel
from ..surrogates import InterDomainWsabiLGPModel


class InterDomainIntegrandModel(IntegrandModel):
    def __init__(
            self,
            prior: Sequence[Sequence[Prior]],
            surrogate: InterDomainWsabiLGPModel,
            num_samples: int = 100,
            reuse_samples: bool = False,
            jitter: float = 1e-9
    ):
        self.prior = prior
        self.surrogate = surrogate
        self.num_samples = num_samples
        self.reuse_samples = reuse_samples
        if self.reuse_samples:
            self.samples = None
            self.samples_samples_covariance = None
        self.jitter = jitter
        self.prediction_weights = None
        self.prediction_hyperparameters = None
        self.prediction_likelihoods = None

    def integral_posterior(
            self, return_covariance_blocks: bool = False
    ) -> Union[
        Tuple[
            MultivariateNormal,
            Sequence[Tensor],
            Sequence[Tensor],
            Sequence[Tensor]
        ],
        MultivariateNormal
    ]:
        data_kernel, data_cholesky, k_inv_z = self._get_data_kernel()
        data_integral_block = torch.zeros(data_kernel.size(0), len(self.surrogate.train_inputs))
        integral_integral_block = torch.zeros(len(self.surrogate.train_inputs), len(self.surrogate.train_inputs))
        num_iterations = 0
        # If too few QMC points are specified, then the integral_integral covariance can be negative.
        while integral_integral_block.sum() <= 0:
            if self.reuse_samples and self.samples is not None:
                samples = self.samples
            else:
                samples = self.sample_priors(self.num_samples)
                if self.reuse_samples:
                    self.samples = samples
            data_sample_covariances = self._data_sample_kernel(samples)
            di_block, sample_posterior_means = self._data_integral_covariance(
                k_inv_z=k_inv_z,
                data_sample_covariances=data_sample_covariances
            )
            ii_block = self._integral_integral_covariance(
                samples=samples,
                data_sample_covariances=data_sample_covariances,
                sample_posterior_means=sample_posterior_means,
                k_inv_z=k_inv_z
            )
            data_integral_block.add_(di_block)
            integral_integral_block.add_(ii_block)
            num_iterations += 1

        success = False
        RETRY_LIMIT = 500
        while not success and num_iterations < RETRY_LIMIT:
            try:
                data_integral_block_ = data_integral_block.sum(dim=1, keepdim=True) / num_iterations
                integral_integral_block_ = integral_integral_block.sum().view(1, 1) / num_iterations
                covariance_blocks = (
                    data_cholesky, data_integral_block_, integral_integral_block_
                )
                train_targets = torch.cat(self.surrogate.train_targets, dim=0)
                posterior_mean = (
                        data_integral_block_.transpose(0, 1)
                        @ torch.cholesky_solve(train_targets.unsqueeze(-1), data_cholesky)
                ).view(-1)
                posterior_correction = (
                        data_integral_block_.transpose(0, 1)
                        @ torch.cholesky_solve(data_integral_block_, data_cholesky)
                )
                posterior_covariance = integral_integral_block_ - posterior_correction

                posterior = MultivariateNormal(
                    posterior_mean, posterior_covariance
                )
                success = True
            except Exception as e:
                data_sample_covariances = self._data_sample_kernel(samples)
                di_block, sample_posterior_means = self._data_integral_covariance(
                    k_inv_z=k_inv_z,
                    data_sample_covariances=data_sample_covariances
                )
                ii_block = self._integral_integral_covariance(
                    samples=samples,
                    data_sample_covariances=data_sample_covariances,
                    sample_posterior_means=sample_posterior_means,
                    k_inv_z=k_inv_z
                )
                data_integral_block.add_(di_block)
                integral_integral_block.add_(ii_block)
                num_iterations += 1
                if num_iterations == RETRY_LIMIT:
                    raise e
        if return_covariance_blocks:
            return (
                posterior, samples, sample_posterior_means, covariance_blocks
            )
        else:
            return posterior

    def _prediction_weights(
            self,
            data_cholesky: Tensor,
            dbl_kernel_integrals: Tensor,
            likelihood_weights: Tensor
    ) -> Tensor:
        """Compute K^-1 @ A @ K^-1 * likelihood_weights.
        data_cholesky is lower Cholesky decomposition of K.
        dbl_kernel_integrals is A
        """
        # K^-1 @ A
        left_solve = torch.cholesky_solve(dbl_kernel_integrals, data_cholesky)
        # B K^-1 = C --> C K = B --> K^T C^T = B^T --> K C^T = B^T
        weights = torch.cholesky_solve(
            left_solve.transpose(0, 1), data_cholesky
        ).transpose(0, 1)
        return weights * likelihood_weights

    def compute_prediction_weights(self):
        """Computes num_data x num_data matrix of quadrature weights."""
        posterior, samples, _, covariance_blocks = self.integral_posterior(
            return_covariance_blocks=True
        )
        train_data_probability = posterior.mean.sum()
        data_cholesky = covariance_blocks[0]
        data_sample_covariances = self._data_sample_kernel(samples)
        dbl_kernel_integrals = [
            self._double_kernel_integral(dsc)
            for dsc in data_sample_covariances
        ]
        train_targets = torch.cat(self.surrogate.unwarped_train_targets, dim=0)
        # Indices of nonzero weights
        # nonzero_indices = train_targets.nonzero().view(-1)
        nonzero_indices = torch.arange(
            train_targets.size(0)
        )[train_targets > 1e-3].view(-1)
        # Outer product
        likelihood_weights = 2 * torch.sqrt(
            train_targets.unsqueeze(-1) @ train_targets.unsqueeze(0)
        )
        weights = [
            self._prediction_weights(
                data_cholesky, dki, likelihood_weights
            )[nonzero_indices][:, nonzero_indices].unsqueeze(0)
            for dki in dbl_kernel_integrals
        ]
        # Divide by P(D) * num_domains
        self.prediction_weights = torch.cat(
            weights, dim=0
        ).div_(train_data_probability).sum(dim=0)
        self.prediction_likelihoods = train_targets[nonzero_indices]
        train_inputs = self.surrogate.train_inputs
        self.prediction_hyperparameters = []
        start_index = 0
        for i, hyperparameters in enumerate(train_inputs):
            end_index = start_index + hyperparameters.size(0)
            mask = torch.logical_and(
                start_index <= nonzero_indices, nonzero_indices < end_index
            )
            nonzero_indices_i = nonzero_indices[mask] - start_index
            try:
                selected_hyperparameters = hyperparameters[nonzero_indices_i]
            except IndexError:
                selected_hyperparameters = torch.tensor([])
            self.prediction_hyperparameters.append(selected_hyperparameters)
            start_index = end_index
        # Assume alpha is approximately zero (in practise it usually is)
        # and ignore the subtraction to keep everything analytic
        # self.prediction_mean_shifts = self.surrogate.alpha / train_targets

    def _get_data_kernel(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get joint data covariances for all domains, and K^-1 @ z for
        individual domains.
        """
        self.surrogate.train()
        data_kernel = self.surrogate().covariance_matrix.detach()
        data_cholesky = torch.cholesky(data_kernel)
        train_targets = torch.cat(self.surrogate.train_targets, dim=0)
        k_inv_z = torch.cholesky_solve(
            train_targets.unsqueeze(-1), data_cholesky
        )
        return data_kernel, data_cholesky, k_inv_z
        # n_elements = torch.tensor(
        #     [i.size(0) for i in self.surrogate.train_inputs]
        # )
        # k_inv_zs = []
        # i_start = 0
        # for i in range(len(n_elements)):
        #     i_end = n_elements[0:i + 1].sum()
        #     covariance = data_kernel[i_start:i_end, i_start:i_end]
        #     decomposition = torch.cholesky(covariance)
        #     k_inv_zs.append(
        #         torch.cholesky_solve(
        #             self.surrogate.train_targets[i].unsqueeze(-1),
        #             decomposition
        #         )
        #     )
        #     i_start = i_end
        # return data_kernel, k_inv_zs

    def sample_priors(self, num_samples) -> Sequence[Tensor]:
        """Draw samples for each domain and concatenate them"""
        samples = []
        for domain_prior in self.prior:
            weights_prior = domain_prior[0]
            means_prior = domain_prior[1]
            scales_prior = domain_prior[2]
            weights_samples = weights_prior.sample((num_samples,))
            means_samples = means_prior.sample((num_samples,))
            scales_samples = scales_prior.sample((num_samples,))
            # weights_samples = ExactGPModel.postprocess_hyperparameter_sample(
            #     weights_samples, num_samples
            # )
            # means_samples = ExactGPModel.postprocess_hyperparameter_sample(
            #     means_samples, num_samples
            # )
            # scales_samples = ExactGPModel.postprocess_hyperparameter_sample(
            #     scales_samples, num_samples
            # )
            weights_samples = weights_samples.view(num_samples, -1)
            means_samples = means_samples.view(num_samples, -1)
            scales_samples = scales_samples.view(num_samples, -1)
            sample = torch.cat(
                (weights_samples, means_samples, scales_samples), dim=1
            )
            samples.append(sample)
        return samples

    def single_kernel_integral(self, predictand, samples):
        """Compute single kernel integral via monte carlo sum"""
        covariances = self.surrogate.covar_module(predictand, samples)
        return covariances.evaluate().detach().sum(dim=1).div_(self.num_samples)

    def _data_sample_kernel(
            self, samples: Sequence[Tensor]
    ) -> Sequence[Tensor]:
        """Compute covariances between all pairs of data and sample
        sets.
        """
        data = self.surrogate.train_inputs
        results = []
        for d in data:
            d_results = []
            for s in samples:
                dsc = self.surrogate.covar_module(d, s)
                d_results.append(dsc)
            results.append(d_results)
        concatenate_over_data = [
            torch.cat([row[i].evaluate().detach() for row in results], dim=0)
            for i in range(len(samples))
        ]
        return concatenate_over_data

    def _double_kernel_integral(
            self,
            data_sample_covariances: Tensor,
            sample_posterior_means: Tensor = None
    ) -> Tensor:
        """Compute double kernel integrals."""
        # Simple Monte Carlo sum. Yields num_data x num_data matrix.
        if sample_posterior_means is None:
            outer_product = (
                data_sample_covariances
                @ data_sample_covariances.transpose(0, 1)
            )
        else:
            assert sample_posterior_means.ndimension() > 1
            outer_product = (
                data_sample_covariances
                @ sample_posterior_means
            )
        return outer_product.div_(self.num_samples)

    def _data_integral_covariance(
            self,
            k_inv_z: Tensor,
            data_sample_covariances: Sequence[Tensor]
    ) -> Tuple[Tensor, Sequence[Tensor]]:
        """Produce data-integral covariance block with
        int K(D, x) m_D(x) dx = dbl_kernel_integral @ K(D, D)^-1 @ z
        for each domain pair.
        """
        # # Number of data points in each domain
        # n_elements = torch.tensor([i.size(0) for i in k_inv_zs])
        # n_data = n_elements.sum().item()
        # Number of domains
        n_domains = len(data_sample_covariances)
        data_integral_columns = []
        sample_posterior_means = []
        for d in range(n_domains):
            sample_posterior_mean = (
                    data_sample_covariances[d].transpose(0, 1) @ k_inv_z
            )
            data_integral_column = self._double_kernel_integral(
                data_sample_covariances[d],
                sample_posterior_mean
            )
            data_integral_columns.append(data_integral_column)
            sample_posterior_means.append(sample_posterior_mean)
            # # right_matrix is K^-1 @ z for the data in domain d.
            # matmul = lambda mat: mat @ k_inv_zs[d]
            # # Do the matrix multiplications to get n_domain tensors
            # # for the covariance between the data in domain d and the
            # # integrals over each domain.
            # d_start = d * n_domains
            # d_end = (d + 1) * n_domains
            # dic = [matmul(dbl) for dbl in dbl_kernel_integrals[d_start: d_end]]
            # # Concatenate the covariances to get a block of rows.
            # dib_block = torch.cat(dic, dim=1)
            # # Place the calculated block in the correct position in the
            # # overall data_integral_covariance matrix.
            # data_integral_block[i_start: i_end] = dib_block
            # i_start = i_end
        data_integral_block = torch.cat(data_integral_columns, dim=1)
        return data_integral_block, sample_posterior_means

    def _triple_kernel_integral(
            self,
            samples_l: Tensor,
            sample_posterior_means_l: Tensor,
            samples_r: Tensor,
            sample_posterior_means_r: Tensor
    ) -> Tensor:
        """Triple kernel integral"""
        # # Compute sample_sample kernel
        # ssc = self.surrogate.covar_module(
        #     samples_l, samples_r
        # ).evaluate().view(-1)
        # # Repeat data_sample covariances
        # dsc_l_interleaved = torch.repeat_interleave(
        #     data_sample_covariances_l.evaluate(), repeats=self.num_samples, dim=1
        # )
        # dsc_r_tiled = data_sample_covariances_r.repeat(1, self.num_samples)
        #
        # # Elementwise product
        # left = dsc_l_interleaved * ssc
        # # Outer product of rows
        # outer_product = left @ dsc_r_tiled.transpose(0, 1).evaluate()
        # return outer_product.div_(self.num_samples**2)

        if self.reuse_samples and self.samples_samples_covariance is not None:
            ssc = self.samples_samples_covariance
        else:
            ssc = self.surrogate.covar_module(
                samples_l, samples_r
            ).evaluate().detach()
            # Set diagonal to zero [van der Wilk et al. (2018)]
            # ssc.fill_diagonal_(0)
            if self.reuse_samples:
                self.samples_samples_covariance = ssc
        double_sum = (
                sample_posterior_means_l.transpose(0, 1)
                @ ssc
                @ sample_posterior_means_r
        )
        # diagonal_sum = torch.sum(
        #     sample_posterior_means_l.squeeze() * ssc.diagonal() * sample_posterior_means_r.squeeze()
        # )
        # result = double_sum - diagonal_sum
        # return result.div_(self.num_samples * (self.num_samples - 1))  # div from [van der Wilk et al. (2018)
        return double_sum.div_(self.num_samples)
    
    def _sample_posterior_means(self, samples: Sequence[Tensor], k_inv_z: Tensor) -> Sequence[Tensor]:
        """Posterior mean in warped space at `samples`.
        
        :param samples: Sample locations for each domain.
        :return: Posterior mean in warped space.
        """
        data_sample_covariances = self._data_sample_kernel(samples)
        n_domains = len(data_sample_covariances)
        sample_posterior_means = []
        for d in range(n_domains):
            sample_posterior_mean = (
                    data_sample_covariances[d].transpose(0, 1) @ k_inv_z
            )
            sample_posterior_means.append(sample_posterior_mean)
        return sample_posterior_means

    def _integral_integral_covariance(
            self,
            samples: Sequence[Tensor],
            data_sample_covariances: Sequence[Tensor],
            sample_posterior_means: Sequence[Tensor],
            k_inv_z: Tensor
    ):
        """Integral covariances between domains."""
        samples_l = self.sample_priors(self.num_samples)
        samples_r = self.sample_priors(self.num_samples)
        sample_posterior_means_l = self._sample_posterior_means(samples_l, k_inv_z)
        sample_posterior_means_r = self._sample_posterior_means(samples_r, k_inv_z)
        n_blocks = len(samples_l)
        covariance = torch.empty(n_blocks, n_blocks)
        for i in range(n_blocks):
            for j in range(i, n_blocks):
                covariance[i, j] = self._triple_kernel_integral(
                    samples_l[i],
                    sample_posterior_means_l[i],
                    samples_r[j],
                    sample_posterior_means_r[j]
                )
        covariance = (
                torch.triu(covariance)
                + torch.triu(covariance, diagonal=1).transpose(0, 1)
        )
        return covariance
