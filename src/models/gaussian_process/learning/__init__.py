import logging
from typing import Union, Mapping, Sequence
from pathlib import Path
from time import time
import torch
from torch import Tensor
from gpytorch.models import GP
from gpytorch.mlls import MarginalLogLikelihood

from ..models import GPModelCollection
from . import (
    # mcmc,
    bayesian_quadrature,
    # variational_bayesian_monte_carlo,
    # variational_reinforce,
    ess_expectation_maximisation
)
from bayes_quad.learning import optimise_hyperparameters, mll_as_loss_function


def learn(
        model: Union[GP, GPModelCollection] = None,
        mll: Union[
            MarginalLogLikelihood, Sequence[MarginalLogLikelihood]
        ] = None,
        train_inputs: Tensor = None,
        train_targets: Tensor = None,
        test_inputs: Tensor = None,
        test_targets: Tensor = None,
        log_dir: Union[Path, str] = Path('./'),
        sacred_run=None,
        smoke_test=False,
        numerics: Mapping = None,
        **kwargs
):
    """Convenience function that sets up and executes numerics options
    specified in a Sacred Ingredient.
    """
    logging.info('Learning')
    assert model is not None, 'model argument must be specified'
    if numerics is None:
        numerics = {
            'strategy': 'maximum_marginal_likelihood',
            'infer_while_learning': False
        }
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Learn
    if numerics['strategy'] == 'maximum_marginal_likelihood':
        output = optimise_hyperparameters(
            train_inputs,
            train_targets,
            model,
            mll_as_loss_function(mll),
            test_inputs=test_inputs,
            test_targets=test_targets,
            sacred_run=sacred_run,
            **numerics
        )
    elif numerics['strategy'] == 'bayesian_quadrature':
        output = bayesian_quadrature.sample(
            model,
            mll,
            train_inputs,
            train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            sacred_run=sacred_run,
            log_dir=log_dir,
            **numerics
        )
    elif numerics['strategy'] == 'ess_expectation_maximisation':
        output = ess_expectation_maximisation.sample(
            model,
            test_x=test_inputs,
            test_y=test_targets,
            sacred_run=sacred_run,
            n_dimensions=(
                train_inputs.size(-1) if train_inputs.ndimension() > 1 else 1
            ),
            **numerics
        )
    else:
        raise NotImplementedError(
            f'{numerics["strategy"]} numerics strategy not implemented'
        )

    # try:
    #     torch.save(model.state_dict(), log_dir / 'model_state.pth')
    # except AttributeError:
    #     # GPModelCollection
    #     for i, m in enumerate(model.models):
    #         torch.save(m.state_dict(), log_dir / f'model_{i}_state.pth')
    #     torch.save(model.model_evidences, log_dir / f'model_evidences.pt')
    # if output is not None:
    #     torch.save(output.gsampled[0], log_dir / 'samples.pt')

    # None for most strategies
    return output
