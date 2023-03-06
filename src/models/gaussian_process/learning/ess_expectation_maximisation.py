from torch import Tensor

from models.gaussian_process.spectralgp import (
    TimedAlternatingSampler, ss_factory, ess_factory
)

def sample(
        model,
        n_ess_iterations: int = 100,
        n_ess_samples_per_iteration: int = 20,
        n_optimiser_iterations: int = 1,
        n_dimensions: int = 1,
        time_budget: int = None,
        skip_run: bool = False,
        infer_while_learning: bool = False,
        test_x: Tensor = None,
        test_y: Tensor = None,
        sacred_run=None,
        **kwargs
):
    data_mod = model
    data_lh = model.likelihood
    alt_sampler = TimedAlternatingSampler(
        [data_mod],
        [data_lh],
        ss_factory,
        [ess_factory],
        totalSamples=n_ess_iterations,
        numInnerSamples=n_ess_samples_per_iteration,
        numOuterSamples=n_optimiser_iterations,
        num_dims=n_dimensions,
        learning_time_limit=time_budget,
    )
    if not skip_run:
        alt_sampler.run(
            test_x=test_x,
            test_y=test_y,
            live_inference=infer_while_learning,
            sacred_run=sacred_run
        )
    return alt_sampler
