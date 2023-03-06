from typing import Sequence, Mapping, Callable
import warnings
from tqdm import trange
from time import time
from sacred.run import Run
import torch
from torch import Tensor
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
    return marginalised_log_likelihood, rmse


def optimise_hyperparameters(
        train_inputs: Tensor,
        train_targets: Tensor,
        model: torch.nn.Module,
        loss_function: Callable,
        optimiser: Mapping,
        test_inputs: Tensor = None,
        test_targets: Tensor = None,
        time_budget: int = None,
        infer_while_learning: bool = False,
        infer_test_likelihood: bool = True,
        retain_graph: bool = False,
        sacred_run: Run = None,
        smoke_test: bool = False,
        **kwargs
):
    """Generic optimisation loop for PyTorch models."""
    num_iterations = 2 if smoke_test else int(
        optimiser.get('num_iterations', 5000)
    )
    if optimiser['type'] == 'lbfgs':
        optimiser_module = torch.optim.LBFGS(
            [{'params': model.parameters()}, ],
            lr=optimiser.get('learning_rate', 1),
            max_iter=num_iterations,
            max_eval=num_iterations,  # Since Scipy defaults also do this.
            history_size=10,
            line_search_fn='strong_wolfe'
        )
    elif optimiser['type'] == 'sgd':
        optimiser_module = torch.optim.SGD(
            [{'params': model.parameters()}, ],
            lr=optimiser.get('learning_rate', 1)
        )
    else:
        raise NotImplementedError

    def monitor(loss: Tensor):
        if sacred_run is not None:
            sacred_run.log_scalar(
                metric_name='surrogate.marginal_log_likelihood',
                value=-loss.item()
            )
            try:
                sacred_run.log_scalar(
                    metric_name='surrogate.outputscale',
                    value=model.covar_module.outputscale.item()
                )
                for index, lengthscale in enumerate(
                        model.covar_module.base_kernel.lengthscale.squeeze(
                        ).view(-1)
                ):
                    sacred_run.log_scalar(
                        metric_name=f'surrogate.lengthscale_{index}',
                        value=lengthscale.item()
                    )
            except:
                pass

    model.train()

    def closure():
        optimiser_module.zero_grad()
        loss = loss_function(train_inputs, train_targets, model)
        loss.backward(retain_graph=retain_graph)
        monitor(loss)  # Log loss and model parameters to Sacred.
        return loss
    optimiser_module.step(closure)
