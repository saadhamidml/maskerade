import logging
from pathlib import Path
import math
import numpy as np
import torch
from scipy import stats

from problems.airfoil import ingest, prepare, segregate
from models import gaussian_process
from models.gaussian_process.priors import LogNormalPrior
from models.gaussian_process.learning.bayesian_quadrature import sample_initial
from bayes_quad import surrogates
from bayes_quad.learning import (
    optimise_hyperparameters as optimise_surrogate,
    inter_domain_warped_mll_as_loss_function
)
from bayes_quad.quadrature.inter_domain import InterDomainIntegrandModel

torch.set_default_dtype(torch.float64)

log_dir = Path('../logs/monte_carlo')
dataframe = ingest(Path('../data'))
feature_preprocessor, target_preprocessor = prepare(dataframe)
train_inputs, train_targets, test_inputs, test_targets = segregate(
    dataframe,
    feature_preprocessor=feature_preprocessor,
    target_preprocessor=target_preprocessor,
    test_size=0.2,
    shuffle=False,
    cross_validation=None,
    use_cuda=False
)

model_components = gaussian_process.build_model(
    train_inputs=train_inputs,
    train_targets=train_targets,
    use_cuda=False,
    collection=True,
    covariance_function={
        'type': "bayesian_gaussian_spectral_mixture",
        'num_mixtures': [1, 2, 3, 4, 5],
        'prior': "automatic"
    }
)
model = model_components['model']

hyperparameters, log_likelihoods = sample_initial(
    model,
    initialisation={'strategy': 'sobol', 'num_samples': 500}
)
max_log_float = torch.zeros(1).squeeze()
max_log_likelihood = torch.tensor(
    [torch.max(l) for l in log_likelihoods]
).max()
likelihood_log_offset = max_log_float - max_log_likelihood
offset_likelihoods = [
    torch.exp(l + likelihood_log_offset) for l in log_likelihoods
]

if train_inputs.ndimension() == 1:
    input_dimensions = 1
else:
    input_dimensions = train_inputs.size(1)
surrogate_components = surrogates.build_model(
    train_inputs=hyperparameters,
    train_targets=offset_likelihoods,
    train_targets_log_offset=likelihood_log_offset,
    dimensions=input_dimensions
)
surrogate_model = surrogate_components['model']
surrogate_metric = surrogate_components['mll']
surrogate_model.covar_module.outputscale = 1.414 * torch.exp(
    max_log_float
)
surrogate_model.covar_module.base_kernel.lengthscale = 10.0

for kernel_integration_num_samples in [100, 500, 1000]:
    means = []
    for i in range(5):
        kernel_integration_num_samples = int(kernel_integration_num_samples)
        integrand_model = InterDomainIntegrandModel(
            prior=model.hyperparameter_priors,
            surrogate=surrogate_model,
            num_samples=kernel_integration_num_samples,
        )
        data_kernel, data_cholesky, k_inv_z = integrand_model._get_data_kernel()
        samples = integrand_model.sample_priors(kernel_integration_num_samples)
        data_sample_covariances = integrand_model._data_sample_kernel(samples)
        data_integral_block, sample_posterior_means = integrand_model._data_integral_covariance(
            k_inv_z=k_inv_z,
            data_sample_covariances=data_sample_covariances
        )
        data_integral_block = data_integral_block.sum(dim=1, keepdim=True)
        train_targets = torch.cat(integrand_model.surrogate.train_targets, dim=0)
        posterior_mean = (
            data_integral_block.transpose(0, 1)
            @ torch.cholesky_solve(train_targets.unsqueeze(-1), data_cholesky)
        ).view(-1)
        means.append(posterior_mean.item())
    print(f'{kernel_integration_num_samples}: {np.mean(means):.3f} pm {stats.sem(means):.3f}')
