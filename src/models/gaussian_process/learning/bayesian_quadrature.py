import logging
from typing import Sequence, Mapping, Union
from pathlib import Path
from time import time
import math
import numpy as np
import torch
from torch import Tensor
from gpytorch.mlls import MarginalLogLikelihood

from models.gaussian_process.priors import LogNormalPrior

from ..models import ExactGPModel, GPModelCollection
from ..inference import bayesian_quadrature as bq_inference
from bayes_quad.surrogates import build_model
from bayes_quad.learning import (
    optimise_hyperparameters as optimise_surrogate,
    inter_domain_warped_mll_as_loss_function
)
from bayes_quad.quadrature.inter_domain import InterDomainIntegrandModel
from bayes_quad.acquisition_functions import acquire


def marginal_likelihood_at_theta(
        model: ExactGPModel,
        mll: MarginalLogLikelihood,
        train_inputs: Tensor,
        train_targets: Tensor,
        theta: Tensor
) -> Sequence[Tensor]:
    """Evaluate Marginal Likelihood at a particular hyperparameter
    setting.

    Note that "marginal" here refers to the marginalisation of the
    function over which the GP is defined. This is the ordinary
    marginal likelihood that is typically maximised when training
    GPs.
    """
    model.hyperparameters_as_tensor = theta
    marginal_log_likelihood = model.likelihood(
        model(train_inputs)
    ).log_prob(train_targets)
    return marginal_log_likelihood.detach()


def sample_initial(model: GPModelCollection, initialisation: Mapping = None):
    hyperparameters = []
    likelihoods = []
    initialisation = {
        'strategy': 'sobol'
    } if initialisation is None else initialisation
    if initialisation['strategy'] == 'sobol':
        low_discrepancy = True
    elif initialisation['strategy'] == 'sample_prior':
        low_discrepancy = False
    else:
        raise NotImplementedError
    num_samples_per_model = int(
        initialisation.get('num_samples', 2 * model.n_models) / model.n_models
    )
    for model_index in range(model.n_models):
        try:
            samples = model.sample_hyperparameters(
                model_index,
                num_samples_per_model,
                low_discrepancy=low_discrepancy
            )
            hyperparameters.append(
                samples[0].view(num_samples_per_model, -1)
            )
            likelihoods.append(samples[1].view(-1))
        except:
            h_mi = []
            l_mi = []
            for _ in range(num_samples_per_model):
                samples = model.sample_hyperparameters(
                    model_index,
                    1,
                    low_discrepancy=low_discrepancy
                )
                h_mi.append(samples[0].view(1, -1))
                l_mi.append(samples[1].view(-1))
            hyperparameters.append(torch.cat(h_mi, dim=0))
            likelihoods.append(torch.cat(l_mi, dim=0))
    return hyperparameters, likelihoods


def sample(
        model: GPModelCollection,
        mll: Sequence[MarginalLogLikelihood],
        train_inputs: Tensor,
        train_targets: Tensor,
        test_inputs: Tensor = None,
        test_targets: Tensor = None,
        load_id: int = None,
        evaluation_budget: int = 100,
        time_budget: int = 600,
        initialisation: Mapping = None,
        surrogate: Mapping = None,
        acquisition_function: Mapping = None,
        kernel_integration_num_samples: int = 100,
        reuse_kernel_integration_samples: bool = False,
        infer_while_learning: bool = False,
        infer_every: int = 1,
        sacred_run=None,
        log_dir: Union[Path, str] = Path('./'),
        **kwargs
):
    """Bayesian Quadrature to marginalise over hyperparameters.

    This function builds a surrogate model for the likelihood surface,
    p(D|\theta), over the hyperparameter space.
    Active sampling is used select new points at which to sample the
    likelihood.
    The p(D) under the model is then computed via Bayesian Quadrature.

    p(D) and the sampled hyperparameters and likelihoods are stored in
    the model object, ready for inference. The model is turned into a
    batch of GPs, each with its hyperparameters set to one from the
    sample.
    """
    total_time = 0.
    start_time = time()

    if load_id is None:
        hyperparameters, log_likelihoods = sample_initial(
            model, initialisation
        )
    else:
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        learn_output_dir = log_dir.parent / f'{load_id}/learn_output'
        hyperparameters = []
        log_likelihoods = []
        for d in range(model.n_models):
            hyperparameters.append(
                torch.load(learn_output_dir / f'{d}_hyperparameters.pt')
            )
            log_likelihoods.append(
                torch.load(learn_output_dir / f'{d}_log_likelihoods.pt')
            )

    if acquisition_function.get('type', 'fit_bq') == 'sample_prior':
        hyperparameters_, log_likelihoods_ = sample_initial(
            model,
            {
                'strategy': 'sample_prior',
                'num_samples': (
                    evaluation_budget
                    - initialisation['num_samples']
                )
            }
        )
        hyperparameters = [
            torch.cat((hyperparameters[i], hyperparameters_[i]), dim=0)
            for i in range(len(hyperparameters))
        ]
        log_likelihoods = [
            torch.cat((log_likelihoods[i], log_likelihoods_[i]), dim=0)
            for i in range(len(log_likelihoods))
        ]
        n_iterations = 0
    else:
        n_iterations = int(
            (evaluation_budget - initialisation['num_samples'])
            / acquisition_function['batch_size']
        )


    max_log_float = torch.zeros(1).squeeze()
    max_log_likelihood = torch.tensor(
        [torch.max(l) for l in log_likelihoods]
    ).max()
    likelihood_log_offset = max_log_float - max_log_likelihood
    offset_likelihoods = [
        torch.exp(l + likelihood_log_offset) for l in log_likelihoods
    ]
    del log_likelihoods

    # Set up models for Bayesian Quadrature.
    if train_inputs.ndimension() == 1:
        input_dimensions = 1
    else:
        input_dimensions = train_inputs.size(1)
    surrogate_components = build_model(
        train_inputs=hyperparameters,
        train_targets=offset_likelihoods,
        train_targets_log_offset=likelihood_log_offset,
        dimensions=input_dimensions,
        **surrogate['model'],
    )
    surrogate_model = surrogate_components['model']
    surrogate_metric = surrogate_components['mll']
    integrand_model = InterDomainIntegrandModel(
            prior=model.hyperparameter_priors,
            surrogate=surrogate_model,
            num_samples=kernel_integration_num_samples,
            reuse_samples=reuse_kernel_integration_samples
        )

    if load_id is None:
        surrogate_model.covar_module.outputscale = 1 / math.e * torch.exp(
            max_log_float
        )
        surrogate_model.covar_module.register_prior(
            'outputscale_prior', LogNormalPrior(0.1, 1 / math.e ** 3), 'outputscale'
        )
        surrogate_model.covar_module.base_kernel.lengthscale = (
            surrogate['numerics'].get('initialisation', {}).get('lengthscale', 1.0)
        )
        triu_inds = torch.triu_indices(
            initialisation['num_samples'], initialisation['num_samples'], 1)
        distances = surrogate_model().covariance_matrix.div_(
            surrogate_model.covar_module.outputscale
        ).log_().mul_(-surrogate_model.covar_module.base_kernel.lengthscale)
        typical_length = distances[triu_inds[0], triu_inds[1]].detach().median().cpu().numpy().item() / math.e ** 3
        surrogate_model.covar_module.base_kernel.lengthscale = typical_length
        surrogate_model.covar_module.base_kernel.register_prior(
            'lengthscale_prior',
            LogNormalPrior(math.log(typical_length), 1 / math.e ** 3),
            'lengthscale'
        )
    else:
        surrogate_model.covar_module.outputscale = (
            torch.load(learn_output_dir / 'surrogate_outputscale.pt')
        )
        surrogate_model.covar_module.base_kernel.lengthscale = (
            torch.load(learn_output_dir / 'surrogate_lengthscale.pt')
        )
        integrand_model.compute_prediction_weights()


    if surrogate['numerics']['strategy'] == 'optimisation':
        optimise_surrogate(
            hyperparameters,
            offset_likelihoods,
            surrogate_model,
            inter_domain_warped_mll_as_loss_function(surrogate_metric),
            **surrogate['numerics'],
            sacred_run=sacred_run
        )
    surrogate_model.covar_module.raw_outputscale.requires_grad_(False)
    surrogate_model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
    total_time += time() - start_time
    print(f'Initial sampling time: {total_time}')

    for i in range(n_iterations):
        if infer_while_learning and (i + 1) % infer_every == 0:
            integrand_model.compute_prediction_weights()
            output = bq_inference(
                model=model,
                learn_output=integrand_model,
                test_inputs=test_inputs,
                test_targets=test_targets,
                sacred_run=sacred_run,
            )
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
                    'elapsed_time',
                    total_time
                )
            del output
            integrand_model.prediction_weights = None
            integrand_model.prediction_hyperparameters = None
            integrand_model.prediction_likelihoods = None
            model.train()
        logging.info(f'Acquisition: {i + 1}/{n_iterations}')
        start_time_i = time()
        # acquisition function
        try:
            model_index, acquisition, stop_flag = acquire(
                model,
                integrand_model,
                acquisition_function,
                sacred_run=sacred_run
            )
        except Exception as e:
            logging.warning(
                f'Acquisiton function optimisation failed with error:\n{e}\nEnding loop.'
            )
            break

        acquisition_log_likelihood_max = acquisition['log_likelihood'].max()
        if acquisition_log_likelihood_max > max_log_likelihood:
            offset_likelihoods = [
                torch.exp(
                    l.log()
                    + max_log_likelihood
                    - acquisition_log_likelihood_max
                ) for l in offset_likelihoods
            ]
            integrand_model.surrogate.unwarped_train_targets_log_offset = (
                max_log_float - acquisition_log_likelihood_max
            )
        hyperparameters[model_index] = torch.cat(
            (hyperparameters[model_index], acquisition['hyperparameters']),
            dim=0
        )
        offset_likelihoods[model_index] = torch.cat(
            (
                offset_likelihoods[model_index],
                torch.exp(
                    acquisition['log_likelihood']
                    + integrand_model.surrogate.unwarped_train_targets_log_offset
                ).view(-1)
            ),
            dim=0
        )
        integrand_model.surrogate.set_train_data(
            hyperparameters, offset_likelihoods
        )
        if (
                surrogate['numerics']['strategy'] == 'optimisation'
                and (i + 1) % surrogate['numerics']['optimise_every'] == 0
        ):
            surrogate_model.covar_module.outputscale = 1 / math.e  * torch.exp(max_log_float)
            num_samples = torch.tensor([h.size(0) for h in hyperparameters]).sum()
            triu_inds = torch.triu_indices(num_samples, num_samples, 1)
            distances = surrogate_model().covariance_matrix.div_(
                surrogate_model.covar_module.outputscale
            ).log_().mul_(-surrogate_model.covar_module.base_kernel.lengthscale)
            typical_length = distances[triu_inds[0], triu_inds[1]].detach().median().cpu().numpy().item() / math.e ** 3
            surrogate_model.covar_module.base_kernel.lengthscale = typical_length

            surrogate_model.covar_module.raw_outputscale.requires_grad_(True)
            surrogate_model.covar_module.base_kernel.raw_lengthscale.requires_grad_(True)
            # Optimise surrogate's hyperparameters.
            optimise_surrogate(
                hyperparameters,
                offset_likelihoods,
                surrogate_model,
                inter_domain_warped_mll_as_loss_function(surrogate_metric),
                **surrogate['numerics'],
                sacred_run=sacred_run
            )
            surrogate_model.covar_module.raw_outputscale.requires_grad_(False)
            surrogate_model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        iteration_time = time() - start_time_i
        total_time += iteration_time
        if (time_budget is not None and total_time > time_budget) or stop_flag:
            break

    if sacred_run is not None:
        try:
            sacred_run.log_scalar(
                'acquisition_iterations',
                (i + 1)
            )
        except:
            sacred_run.log_scalar(
                'acquisition_iterations',
                0
            )

    integrand_model.compute_prediction_weights()

    # Save learnt information.
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    learn_output_dir = log_dir / 'learn_output'
    learn_output_dir.mkdir(parents=True, exist_ok=True)
    for d, (h, l) in enumerate(zip(hyperparameters, offset_likelihoods)):
        torch.save(h, learn_output_dir / f'{d}_hyperparameters.pt')
        likelihood = (
            l.log()
            + integrand_model.surrogate.unwarped_train_targets_log_offset
        )
        torch.save(
            likelihood, learn_output_dir / f'{d}_log_likelihoods.pt'
        )
    torch.save(
        surrogate_model.covar_module.outputscale.squeeze(),
        learn_output_dir / 'surrogate_outputscale.pt'
    )
    torch.save(
        surrogate_model.covar_module.base_kernel.lengthscale.squeeze(),
        learn_output_dir / 'surrogate_lengthscale.pt'
    )

    return integrand_model
