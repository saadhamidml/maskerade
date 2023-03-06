"""Inference for GPs."""
from typing import Sequence, Mapping, Union
from pathlib import Path
from time import time
import math
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import GP
from gpytorch.mlls import MarginalLogLikelihood
from spectralgp.models import SpectralModel
from spectralgp.samplers import AlternatingSampler

from bayes_quad.quadrature.inter_domain import InterDomainIntegrandModel
from models.gaussian_process.models import GPModelCollection

import gc


def evaluate_optimised_hyperparameters(
        model: GP,
        test_inputs: Tensor,
        test_targets: Tensor,
) -> Sequence[Tensor]:
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.likelihood(model(test_inputs))
    with gpytorch.settings.fast_computations(log_prob=True):
        marginalised_log_likelihood = posterior.log_prob(test_targets).item()
    rmse = (posterior.mean - test_targets).pow(2).mean().sqrt().item()
    output = marginalised_log_likelihood, rmse
    return output


def bayesian_quadrature(
        model: GPModelCollection,
        learn_output: InterDomainIntegrandModel,
        test_inputs: Tensor,
        test_targets: Tensor = None,
        sacred_run=None,
        **numerics,
):
    # Make predictions at all sampled hyperparameters
    n_predictands = test_inputs.size(0)
    model.eval()
    quadrature_weights = learn_output.prediction_weights
    hyperparameters = learn_output.prediction_hyperparameters
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(hyperparameters, test_inputs, batch=False)
    if model.n_random_fourier_features is not None:
        # Too much data to work out log prob of test set.
        means = []
        variances = []
        for p in predictions:
            means.append(p[0].unsqueeze(0))
            variances.append(p[1].unsqueeze(0))
        means = torch.cat(means, dim=0)
        variances = torch.cat(variances, dim=0)
        variance_sums = variances.unsqueeze(1) + variances.unsqueeze(0)
        means_ = means.unsqueeze(1).expand_as(variance_sums)
        distributions = Normal(means_, variance_sums)
        log_prob_weights = distributions.log_prob(
            means.unsqueeze(0).expand_as(means_)
        ) / 2
        del means_
        means_variances = means * variances
        del means
        precision_sums = 1 / variance_sums
        del variance_sums
        log_normalisation_correction = (
                (0.75 * 1) * torch.tensor(2.).log()
                + (1 / 4) * torch.tensor(math.pi).log()
                - 1.25 * precision_sums.log()
        )
        del precision_sums
        log_weight_factors = log_prob_weights + log_normalisation_correction
        del log_prob_weights
        del log_normalisation_correction
        # precision sums already incorporated in normalisation correction.
        combined_means = log_weight_factors.exp_() * (
            means_variances.unsqueeze(1) + means_variances.unsqueeze(0)
        )
        output_mean = (
            quadrature_weights.unsqueeze(-1) * combined_means
        ).sum_to_size(combined_means.size(-1))
        rmse = (output_mean - test_targets).pow_(2).mean().sqrt()
        return np.nan, rmse.item(), np.nan
    if test_targets is not None:
        likelihoods = learn_output.prediction_likelihoods
        try:
            test_probs = [p.log_prob(test_targets) for p in predictions]
        except:
            with gpytorch.settings.fast_computations(log_prob=False):
                test_probs = [p.log_prob(test_targets) for p in predictions]
        try:
            test_probs = torch.cat(test_probs, dim=0)
        except RuntimeError:
            test_probs = torch.tensor(test_probs)
        print(f'Log Probs of Test Set: {test_probs}')
        # Subtract the max to make things numerically stable. This is an
        # approximation that is only exact if learn_output.surrogate.alpha is
        # zero.
        max_test_prob = test_probs.max()
        test_probs = (test_probs - max_test_prob).exp_()
        warped_data = torch.sqrt(
            2 * (likelihoods * test_probs - learn_output.surrogate.alpha)
        ).unsqueeze(-1)
        # Expand to num_domains
        warped_data = warped_data.expand(quadrature_weights.size(0), -1, -1)
        del test_probs
    means = []
    covariances = []
    for p in predictions:
        means.append(
            p.mean.unsqueeze(0) if p.mean.ndimension() == 1 else p.mean
        )
        covariances.append(
            p.covariance_matrix.unsqueeze(0) if p.mean.ndimension() == 1
            else p.covariance_matrix
        )
    means = torch.cat(means, dim=0)
    covariances = torch.cat(covariances, dim=0)

    # collect batches of quadrature weights and proceed with computation of combination predictions
    n_hyperparams = sum([x.shape[0] for x in learn_output.prediction_hyperparameters])
    n_weights = n_hyperparams ** 2
    batch_size = n_weights
    cur_index = 0

    output_mean = torch.zeros(n_predictands)
    output_variance = torch.zeros(n_predictands)

    while cur_index < n_weights:
        try:
            # TODO: Only need to compute for upper triangle, since symmetric.

            #TODO: Python doesn't clear memory when try raises exception;
            # need alternative solution.

            # index pairs for current weight range
            # kls = quadrature_weights.view(-1).argsort(descending=True)[cur_index : cur_index+batch_size]
            # ks, ls = (kls // n_hyperparams).unsqueeze(1), (kls % n_hyperparams).unsqueeze(0)
            # del kls
            ks = torch.arange(n_hyperparams).view(-1, 1)
            ls = torch.arange(n_hyperparams).view(1, -1)
            covariance_sums = covariances[ks] + covariances[ls]
            # P(m_k;m_l, sigma_k + sigma_l) ^0.5
            with gpytorch.settings.fast_computations(log_prob=False):
                mvs = MultivariateNormal(
                    means[ks], covariance_sums
                )
                log_prob_weights = mvs.log_prob(means[ls]) / 2

            covariance_sum_chols = mvs.scale_tril  # [N, N, D, D]
            
            Li_covs_1 = torch.triangular_solve(
                covariances.unsqueeze(0), covariance_sum_chols, upper=False
            )[0]
            Li_covs_2 = torch.triangular_solve(
                covariances.unsqueeze(1), covariance_sum_chols, upper=False
            )[0]
            Li_means_1 = torch.triangular_solve(
                means.unsqueeze(-1).unsqueeze(0), covariance_sum_chols, upper=False
            )[0]
            Li_means_2 = torch.triangular_solve(
                means.unsqueeze(-1).unsqueeze(1), covariance_sum_chols, upper=False
            )[0]

            combined_covariances = 2 * Li_covs_1.transpose(-2, -1) @ Li_covs_2  # [N, N, D, D]
            combined_means = (
                Li_covs_2.transpose(-2, -1) @ Li_means_1
                + Li_covs_1.transpose(-2, -1) @ Li_means_2
            ).squeeze()  # [N, N, D]

            combined_variances = combined_covariances.diagonal(dim1=-2, dim2=-1)

            # factor form square-root of normal (in log-space)
            combined_cov_chols = torch.cholesky(combined_covariances)
            log_normalisation_correction = (
                + (0.5 * n_predictands) * torch.tensor(2.).log()
                + (n_predictands / 4) * torch.tensor(math.pi).log()
                + 0.5 * combined_cov_chols.diagonal(
                    dim1=-2, dim2=-1
                ).log().sum(-1)
            )

            log_weight_factors = log_prob_weights + log_normalisation_correction
            del log_prob_weights
            del log_normalisation_correction
            # qw will have zeros so cannot stay in log space
            weights = (quadrature_weights[ls, ks] * log_weight_factors.exp()).unsqueeze_(-1)  # [N, N, D]

            output_mean += (weights * combined_means).sum(0).sum(0)  # [D]
            output_variance += (weights * (combined_variances + combined_means.pow(2))).sum(0).sum(0)  # [D]

            cur_index += batch_size
        except Exception as e:
            batch_size = min(batch_size // 2, n_weights - cur_index)
            print(e)
        finally:
            gc.collect()
            # torch.cuda.empty_cache()
    del means
    del covariances

    output_variance -= output_mean.pow(2)

    # Work out log_prob and rmse of test set.
    if test_targets is not None:
        test_set_prob = (
                warped_data.transpose(1, 2) @ quadrature_weights @ warped_data
        ).sum()
        test_set_prob_log = test_set_prob.log() + max_test_prob
        rmse = (
            output_mean - test_targets
        ).pow_(2).mean().sqrt()
        return test_set_prob_log.item(), rmse.item()
    else:
        return output_mean, output_variance, weights.view(-1), combined_means.view(-1, n_predictands), combined_variances.view(-1, n_predictands)


def fkl_model_average(data_mod, data_lh, alt_sampler, train_x, train_y, test_x, in_dims):
    data_mod.eval()
    data_lh.eval()
    data_mod_means = torch.zeros_like(data_mod(test_x).mean)
    total_variance = torch.zeros_like(data_lh(data_mod(test_x)).variance)
    with torch.no_grad(), gpytorch.settings.fast_computations(False, False, False):
        #marg_samples_num = min(len(alt_sampler.fhsampled[0][0]), alt_sampler.fgsampled[0].shape[-1])
        marg_samples_num = alt_sampler.gsampled[0].shape[-1]
        for x in range(0, marg_samples_num):
            for dim in range(0,in_dims):
                data_mod.covar_module.set_latent_params(alt_sampler.gsampled[dim][0, :, x], idx=dim)
            data_mod.set_train_data(train_x, train_y) # to clear out the cache
            data_mod_means += data_mod(test_x).mean
            y_preds = data_lh(data_mod(test_x))
            # y_var = f_var + data_noise
            y_var = y_preds.variance
            total_variance += (y_var + torch.pow(data_mod(test_x).mean,2))
    meaned_data_mod_means = data_mod_means / float(marg_samples_num)
    total_variance = total_variance/float(marg_samples_num) - torch.pow(meaned_data_mod_means,2)

    return meaned_data_mod_means, total_variance


def evaluate_functional_kernel_learning(
        test_x: Tensor,
        test_y: Tensor,
        model: SpectralModel,
        sampler: AlternatingSampler,
        n_integration_samples: int = 10,
        **kwargs
) -> Sequence[float]:
    n_dimensions = test_x.size(1) if test_x.ndimension() > 1 else 1
    if n_dimensions == 1:
        last_samples = min(n_integration_samples, sampler.gsampled[0].size(-1))
        # preprocess the spectral samples #
        out_samples = sampler.gsampled[0][0, :, -last_samples:].detach()

        log_probs = torch.zeros(last_samples)
        means = torch.zeros(last_samples, test_y.size(0))

        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_computations(False, False, False):
            for ii in range(last_samples):
                model.covar_module.set_latent_params(out_samples[:, ii])
                model.prediction_strategy = None
                log_probs[ii] = model.likelihood(model(test_x)).log_prob(test_y)
                means[ii] = model(test_x).mean

        marginal_log_prob = (
                log_probs.logsumexp(dim=0)
                - torch.tensor(last_samples).to(test_y).log()
        ).item()

        rmse = (means.mean(dim=0) - test_y).pow(2).mean().sqrt().item()
    else:
        meaned_data_mod_means, total_variance = fkl_model_average(
            model,
            model.likelihood,
            sampler,
            model.train_inputs[0],
            model.train_targets,
            test_x,
            n_dimensions
        )
        nll_sum = 0.0
        d = meaned_data_mod_means - test_y
        rmse = torch.sqrt(torch.mean(torch.pow(d, 2))).item()
        nll = 0.5 * torch.log(2. * math.pi * total_variance) + torch.pow(
            (meaned_data_mod_means - test_y), 2) / (2. * total_variance)
        nll_sum += nll.sum()
        marginal_log_prob = -nll_sum.item()
    return marginal_log_prob, rmse


def infer(
        model: Union[GP, GPModelCollection, SpectralModel] = None,
        mll: MarginalLogLikelihood = None,
        learn_output=None,
        test_inputs: Tensor = None,
        test_targets: Tensor = None,
        numerics: Mapping = None,
        sacred_run=None,
        log_dir=Path('./'),
        **kwargs
):
    """Compute probability of test data under predictive posterior."""
    # Get into evaluation (predictive posterior) mode
    start_time = time()
    model.eval()
    if numerics['strategy'] == 'maximum_marginal_likelihood':
        output = evaluate_optimised_hyperparameters(
            model,
            test_inputs,
            test_targets
        )
    elif numerics['strategy'] == 'bayesian_quadrature':
        output = bayesian_quadrature(
            model,
            learn_output,
            test_inputs,
            test_targets,
            **numerics,
            sacred_run=sacred_run,
            **kwargs
        )
    elif numerics['strategy'] == 'ess_expectation_maximisation':
        output = evaluate_functional_kernel_learning(
            test_inputs,
            test_targets,
            model,
            learn_output,
            **numerics
        )

    inference_time = time() - start_time
    if sacred_run is not None:
        sacred_run.log_scalar(
            'test_data_log_marginal_likelihood',
            output[0]
        )
        sacred_run.log_scalar(
            'test_rmse',
            output[1]
        )
        sacred_run.log_scalar(
            'inference_time',
            inference_time
        )

    return output
