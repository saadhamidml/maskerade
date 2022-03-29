from typing import Sequence, Mapping, Callable
import warnings
from tqdm import trange
from time import time
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import gpytorch
from gpytorch.models import GP
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.kernels import ScaleKernel


def mll_as_loss_function_without_prior(mll: MarginalLogLikelihood) -> Callable:
    def loss_function(train_x, train_y, model):
        return -model.likelihood(model(train_x)).log_prob(train_y).sum()
    return loss_function


def mll_as_loss_function(mll: MarginalLogLikelihood) -> Callable:
    def loss_function(train_x, train_y, model):
        return -mll(model(train_x), train_y).sum()
    return loss_function


def warped_mll_as_loss_function(mll: MarginalLogLikelihood) -> Callable:
    def loss_function(train_x, train_y, model):
        return -mll(model(train_x), model.warping_function(train_y).detach())
    return loss_function


def inter_domain_warped_mll_as_loss_function(mll: MarginalLogLikelihood) -> Callable:
    def loss_function(train_x, train_y, model):
        return -mll(
            model(*train_x),
            torch.cat(model.warping_function(train_y), dim=0).detach()
        )
    return loss_function


def evaluate_optimised_hyperparameters(
        model: GP,
        test_inputs: Tensor,
        test_targets: Tensor,
        infer_test_likelihood: bool = True,
) -> Sequence[Tensor]:
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.likelihood(model(test_inputs))
    if infer_test_likelihood:
        marginalised_log_likelihood = posterior.log_prob(
            test_targets
        ).item()
    else:
        from numpy import nan
        marginalised_log_likelihood = nan
    rmse = (posterior.mean - test_targets).pow(2).mean().sqrt().item()
    output_mean = posterior.mean
    output_variance = posterior.variance
    output_covariance = posterior.covariance_matrix
    output_stddev = output_variance.sqrt()
    upper_credible = output_mean + 2 * output_stddev
    lower_credible = output_mean - 2 * output_stddev
    outside_credible = torch.mean(
        (upper_credible < test_targets).float()
        + (test_targets < lower_credible).float()
    ).item()
    qq_plot_data = torch.solve(
        (test_targets - output_mean).view(-1, 1),
        torch.cholesky(output_covariance)
    )[0]
    output = (
        marginalised_log_likelihood, rmse, outside_credible, qq_plot_data
    )
    return output


def optimise_hyperparameters(
        train_x: Tensor,
        train_y: Tensor,
        model: torch.nn.Module,
        loss_function: Callable,
        optimiser: Mapping,
        test_inputs: Tensor = None,
        test_targets: Tensor = None,
        time_budget: int = None,
        infer_while_learning: bool = False,
        infer_test_likelihood: bool = True,
        retain_graph: bool = False,
        tensorboard: SummaryWriter = SummaryWriter(),
        global_step_offset: int = 0,
        sacred_run=None,
        **kwargs
):
    """Generic optimisation loop for PyTorch models."""
    total_time = 0.
    start_time = time()
    num_iterations = int(optimiser.get('num_iterations', 100))
    global_step_offset *= num_iterations
    if optimiser['type'] == 'sgd':
        optimiser_class = torch.optim.SGD
    optimiser = optimiser_class(
        [{'params': model.parameters()}, ],
        lr=optimiser.get('learning_rate', 1e-1)
    )
    model.train()
    total_time += time() - start_time
    with trange(num_iterations) as t:
        t.set_description(f'Hyperparameter Optimisation')
        for i in t:
            if infer_while_learning and (i % 100) == 0:
                output = evaluate_optimised_hyperparameters(
                    model=model,
                    test_inputs=test_inputs,
                    test_targets=test_targets,
                    infer_test_likelihood=infer_test_likelihood
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
                        'outside_credible',
                        output[2]
                    )
                    sacred_run.log_scalar(
                        'elapsed_time',
                        total_time
                    )
                model.train()
            start_time_i = time()

            def closure():
                optimiser.zero_grad()
                loss = loss_function(train_x, train_y, model)
                loss.backward(retain_graph=retain_graph)
                grad_norm = torch.zeros(1)
                for group in optimiser.param_groups:
                    for param in group['params']:
                        try:
                            grad_norm += param.grad.pow(2).sum()
                        except AttributeError:
                            warnings.warn('Some Parameters do not have gradients.')
                            pass
                grad_norm = torch.sqrt(grad_norm)
                t.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())
                tensorboard.add_scalar(
                    tag='marginal_log_likelihood',
                    scalar_value=-loss.item(),
                    global_step=global_step_offset + i
                )
                if isinstance(model.covar_module, ScaleKernel):
                    tensorboard.add_scalars(
                        main_tag='outputscales',
                        tag_scalar_dict={
                            'outputscale':
                                model.covar_module.outputscale.item(),
                            'noise': model.likelihood.noise.item()
                        },
                        global_step=global_step_offset + i
                    )
                    tensorboard.add_scalars(
                        main_tag='lengthscales',
                        tag_scalar_dict={
                            f'lengthscale_{index}': lengthscale.item()
                            for index, lengthscale in enumerate(
                                model.covar_module.base_kernel.lengthscale.squeeze().view(-1)
                            )
                        },
                        global_step=global_step_offset + i
                    )
                return loss
            optimiser.step(closure)
            iteration_time = time() - start_time_i
            total_time += iteration_time
            if time_budget is not None and total_time > time_budget:
                break
