"""Acquisition functions and optimisation routines for them."""
from typing import Sequence, Mapping, Callable
import logging
import warnings
from sacred.run import Run
from tqdm import trange
import torch
from torch import Tensor
import gpytorch

from bayes_quad.quadrature.standard import IntegrandModel
from bayes_quad.quadrature.inter_domain import InterDomainIntegrandModel
from utils import cholesky_update


def uncertainty_sampling(integrand_model: InterDomainIntegrandModel) -> Callable:
    """The posterior variance of the surrogate."""
    def f(x: Sequence[Tensor], batch_size=1) -> Tensor:
        if batch_size > 1:
            raise NotImplementedError

        integrand_model.surrogate.eval()
        log_variance = integrand_model.surrogate(x).variance.log()

        log_prior = []
        dimensions = integrand_model.surrogate.covar_module.base_kernel.dimensions
        denominator = 1 + 2 * dimensions
        for xi, priors in zip(x, integrand_model.prior):
            xi_num_components = int(xi.size(1) / denominator)
            xi_weights = xi[:, :xi_num_components]
            xi_means = xi[
                :, xi_num_components:xi_num_components * (1 + dimensions)
            ]
            xi_scales = xi[:, xi_num_components * (1 + dimensions):]
            log_prior.append(
                priors[0].log_prob(xi_weights)
                + priors[1].log_prob(xi_means).sum(dim=1)
                + priors[2].log_prob(xi_scales).sum(dim=1)
            )
        log_prior = torch.cat(log_prior, dim=0)

        # Log will preserve location of maximum whilst having nicer
        # numerics.
        return log_variance + 2 * log_prior
    return f


def fit_bq(
        integrand_model: InterDomainIntegrandModel,
        sacred_run: Run = None,
        **kwargs
):
    with torch.no_grad():
        (
            posterior, samples, sample_posterior_means, covariance_blocks
        ) = integrand_model.integral_posterior(
            return_covariance_blocks=True
        )
        data_cholesky = covariance_blocks[0]
        train_targets = torch.cat(integrand_model.surrogate.train_targets, dim=0)

        # cholesky decomposition of full covariance.
        data_int_cholesky = cholesky_update(
            data_cholesky,
            cross_covariance=covariance_blocks[1],
            self_covariance=covariance_blocks[2]
        )

    if sacred_run is not None:
        sacred_run.log_scalar(
            metric_name='integral.mean',
            value=posterior.mean.item()
        )
        sacred_run.log_scalar(
            metric_name='integral.scale',
            value=posterior.covariance_matrix.sqrt().item()
        )

    def f(x: Sequence[Tensor], batch_size: int = 1) -> Tensor:
        train_inputs = integrand_model.surrogate.train_inputs
        n_domains = len(train_inputs)
        data_predictand_cov = []
        predictand_integral_cov = []
        for i in range(n_domains):
            data_i_predictand_cov = []
            predictand_i_integral_cov = []
            for j in range(n_domains):
                data_i_predictand_j = integrand_model.surrogate.covar_module(
                    train_inputs[i], x[j]
                ).detach()
                predictand_i_sample_j = integrand_model.surrogate.covar_module(
                    x[i], samples[j]
                )
                predictand_i_integral_j = (
                    integrand_model._double_kernel_integral(
                        predictand_i_sample_j.evaluate(),
                        sample_posterior_means[j]
                    )
                )
                data_i_predictand_cov.append(data_i_predictand_j.evaluate())
                predictand_i_integral_cov.append(predictand_i_integral_j)
            data_predictand_cov.append(torch.cat(data_i_predictand_cov, dim=1))
            predictand_integral_cov.append(
                torch.cat(
                    predictand_i_integral_cov, dim=1
                ).sum(dim=1, keepdim=True)
            )
        del data_i_predictand_cov
        del predictand_i_integral_cov
        data_predictand_cov = torch.cat(data_predictand_cov, dim=0)
        predictand_integral_cov = torch.cat(predictand_integral_cov, dim=0)

        prediction_right = torch.cholesky_solve(
            train_targets.unsqueeze(1), data_cholesky
        )
        prediction_mean = data_predictand_cov.transpose(0, 1) @ prediction_right
        predictand_integral_cov = prediction_mean * predictand_integral_cov
        data_predictand_cov = data_predictand_cov * prediction_mean.squeeze()

        predictand_di_cov = torch.cat(
            (data_predictand_cov.transpose(0, 1), predictand_integral_cov),
            dim=1
        )
        batched_di_predictand_cov = predictand_di_cov.view(
            -1, batch_size, predictand_di_cov.size(1)
        ).transpose(1, 2)
        n_batches = batched_di_predictand_cov.size(0)
        right = torch.cholesky_solve(
            batched_di_predictand_cov, data_int_cholesky.unsqueeze(0).expand(
                n_batches, -1, -1
            )
        )
        int_cov_correction = batched_di_predictand_cov.transpose(1, 2) @ right
        del batched_di_predictand_cov
        del predictand_di_cov
        del predictand_integral_cov

        # K_f. K_f is K_gg * outer_prod of prediction_mean
        if batch_size == 1:
            predictand_predictand_cov = (
                integrand_model.surrogate.covar_module.outputscale
            )
            predictand_predictand_cov = (
                predictand_predictand_cov * prediction_mean**2
            )
            predictand_predictand_cov = predictand_predictand_cov.view(
                -1, 1, 1
            )
        else:
            predictand_predictand_cov = torch.empty(
                n_batches, batch_size, batch_size
            )
            n_batches_per_domain = int(n_batches / n_domains)
            for i in range(n_domains):
                start_index = 0
                for j in range(n_batches_per_domain):
                    end_index = start_index + batch_size
                    batch_index = i * n_batches_per_domain + j
                    predictand_predictand_cov[batch_index] = (
                        integrand_model.surrogate.covar_module(
                            x[i][start_index:end_index],
                            x[i][start_index:end_index]
                        ).evaluate()
                    )
                    relevant_means = prediction_mean[
                        batch_index * batch_size
                        :(batch_index + 1) * batch_size
                    ]
                    predictand_predictand_cov[batch_index].mul_(
                        relevant_means * relevant_means.transpose(0, 1)
                    )
                    start_index = end_index

        # K_fg @ K_dd^-1 @ K_gf. K_gf is dp_cov
        batched_dp_cov = data_predictand_cov.transpose(0, 1).view(
            -1, batch_size, data_predictand_cov.size(0)
        ).transpose(1, 2)
        cov_correction_right = torch.cholesky_solve(
            batched_dp_cov,
            data_cholesky.unsqueeze(0).expand(n_batches, -1, -1)
        )
        cov_correction = batched_dp_cov.transpose(1, 2) @ cov_correction_right

        prediction_cov = predictand_predictand_cov - cov_correction
        pred_cond_int_cov = predictand_predictand_cov - int_cov_correction

        mutual_information = (
            prediction_cov.logdet() - pred_cond_int_cov.logdet()
        )

        if (mutual_information < 0).any():
            warnings.warn('Negative mutual information')

        return mutual_information

    return f


def project_simplex(x, thresh):
    # projects on the unit simplex
    x = x - x.min()
    paramMu = -1.0
    loss = torch.max(x - paramMu, torch.zeros_like(x)).sum() - 1.0
    while loss.abs() > thresh:
        loss = torch.max(x - paramMu, torch.zeros_like(x)).sum() - 1.0
        df = -torch.sum((x - paramMu) > 0)
        paramMu = paramMu - (loss / df)
    return torch.max(x - paramMu, torch.zeros_like(x))


def project_contrained_simplex(
    constraint: gpytorch.constraints.Interval,
    num_mixtures: Sequence[int],
    raw_batches: Sequence[Tensor]
):
    """Project onto simplex in constrained space.
    
    :param constraint: GPyTorch constraint object.
    :param num_mixtures: Number of mixtures for each set in raw_batches.
    :param raw_batches: Batches in unconstrained space. Sequence over
    number of mixtures.

    raw_batches are updated in place.
    """
    # Project weights onto simplex with softmax
    for n, rb in zip(num_mixtures, raw_batches):
        # project_simplex() can only work on one row at a time.
        for c, raw_candidate in enumerate(rb):
            # Map back to original domain.
            candidate_weights = constraint.transform(
                raw_candidate[:n]
            )
            # Project onto simplex.
            projected_candidate = project_simplex(
                candidate_weights, 1e-4
            ).clamp_min_(1e-8)
            # Map projection to unconstrained domain.
            projected_raw_candidate = constraint.inverse_transform(
                projected_candidate
            )
            # Replace raw_weights in place.
            rb.data[c, :n] = projected_raw_candidate


def zero_nans(losses):
    nan_losses = torch.isnan(losses)
    if torch.isnan(losses).any():
        losses[nan_losses] = 0
        warnings.warn(
            f'{nan_losses.sum().item()} losses are NaN, increase number of MC samples.'
        )
    all_nans = torch.isnan(losses).all()
    return losses, all_nans


def collection_multi_start_optimise(
        objective_function: Callable,
        initial_point_generator: Callable,
        initialisation: Mapping,
        optimiser: Mapping,
        num_mixtures: Sequence[int],
        batch_size: int = 1,
        num_batches: int = 1,
        **kwargs
):
    initial_points = initial_point_generator(initialisation['num_samples'])
    n_domains = len(initial_points)
    constraint = gpytorch.constraints.GreaterThan(torch.tensor(1e-9))

    # Construct batch from best individual initial points.
    try:
        losses = -objective_function(initial_points).view(n_domains, -1)
        losses, _ = zero_nans(losses)
        losses_sorted, l_sort_inds = torch.sort(losses, dim=1)
        selected_indices = l_sort_inds[:, :num_batches * batch_size]
        raw_batches = [
            constraint.inverse_transform(ips[inds])
            for inds, ips in zip(selected_indices, initial_points)
        ]
    except:
        raw_batches = [
            constraint.inverse_transform(ips[: num_batches * batch_size])
            for ips in initial_points
        ]
    for rb in raw_batches:
        rb.requires_grad_(True)

    num_iterations = optimiser.get('num_iterations', 100)
    optimiser_type = optimiser.get('type', 'sgd')
    if optimiser_type == 'sgd':
        optimiser_instance = torch.optim.SGD(
            raw_batches,
            lr=optimiser.get('learning_rate', 1e-1)
        )
    elif optimiser_type == 'lbfgs':
        optimiser_instance = torch.optim.LBFGS(
            raw_batches,
            lr=optimiser.get('learning_rate', 1),
            max_iter=num_iterations,
            max_eval=num_iterations,  # Since Scipy defaults also do this.
            history_size=10,
            line_search_fn='strong_wolfe'
        )
        num_iterations = 1
    else:
        raise NotImplementedError

    with trange(int(num_iterations)) as t:
        t.set_description(f'Acquisition Function Optimisation')
        for i in t:
            def closure():
                if kwargs.get('project_simplex', True):
                    project_contrained_simplex(
                        constraint, num_mixtures, raw_batches
                    )
                optimiser_instance.zero_grad()
                batches = [constraint.transform(rb) for rb in raw_batches]
                losses = -objective_function(
                    batches, batch_size=batch_size
                )
                losses, _ = zero_nans(losses)
                loss = losses.sum()
                loss.backward()
                t.set_postfix(loss=loss.item())
                return loss
            optimiser_instance.step(closure)
    for rb in raw_batches:
        rb.requires_grad_(False)
    if kwargs.get('project_simplex', True):
        project_contrained_simplex(constraint, num_mixtures, raw_batches)
    batches = [constraint.transform(rb) for rb in raw_batches]
    if not (num_iterations == 0 and batch_size == 1):
        losses = -objective_function(
            batches, batch_size=batch_size
        ).view(n_domains, -1)
        losses, stop_flag = zero_nans(losses)
    else:
        losses = losses_sorted[:, :num_batches * batch_size]
    min_losses, indices = torch.min(losses, dim=1)
    selected_model = torch.argmin(min_losses)
    selected_batch = indices[selected_model]
    selected_model_bs = batches[selected_model.item()]
    selected_hyperparameters = selected_model_bs[
        selected_batch * batch_size:(selected_batch + 1) * batch_size
    ].detach().requires_grad_(False)
    logging.info(f'Selected:\n{selected_hyperparameters}')
    logging.info(f'Acquisition Value: {torch.min(min_losses).abs().item()}')
    return selected_model, selected_hyperparameters, stop_flag


def acquire(
        model,
        integrand_model: InterDomainIntegrandModel,
        acquisition_function: Mapping,
        sacred_run: Run = None,
):
    if acquisition_function['type'] == 'fit_bq':
        if acquisition_function['numerics']['initialisation'][
            'strategy'
        ] == 'sample_prior':
            initial_point_generator = integrand_model.sample_priors
        else:
            raise NotImplementedError
        objective_function = fit_bq(
            integrand_model, sacred_run=sacred_run, **acquisition_function
        )
        model_index, query, stop_flag = collection_multi_start_optimise(
            objective_function,
            initial_point_generator,
            **acquisition_function['numerics'],
            num_mixtures=model.num_mixtures,
            batch_size=acquisition_function['batch_size'],
            num_batches=acquisition_function['num_batches']
        )
    elif acquisition_function['type'] == 'uncertainty_sampling':
        if acquisition_function['numerics']['initialisation'][
            'strategy'
        ] == 'sample_prior':
            initial_point_generator = integrand_model.sample_priors
        else:
            raise NotImplementedError
        objective_function = uncertainty_sampling(integrand_model)
        model_index, query, stop_flag = collection_multi_start_optimise(
            objective_function,
            initial_point_generator,
            **acquisition_function['numerics'],
            num_mixtures=model.num_mixtures,
            batch_size=acquisition_function['batch_size'],
            num_batches=acquisition_function['num_batches']
        )
        integrand_model.surrogate.train()
    else:
        raise NotImplementedError

    with torch.no_grad(), gpytorch.settings.max_cg_iterations(1000):
        try:
            sample = model.compute_likelihood(model_index, query)
        except:
            sample = [
                model.compute_likelihood(model_index, q).item() for q in query
            ]
            sample = torch.tensor(sample)
    logging.info(f'Log Likelihood:\n{sample}')
    return (
        model_index,
        {
            'hyperparameters': query.view(
                acquisition_function['batch_size'], -1
            ),
            'log_likelihood': sample
        },
        stop_flag
    )
