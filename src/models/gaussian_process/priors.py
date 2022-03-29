from typing import Sequence, Mapping
import torch
from torch import Tensor
from torch.nn import Module as TModule
from torch.distributions import Dirichlet
from gpytorch.priors import Prior, UniformPrior, NormalPrior, LogNormalPrior
from gpytorch.priors.utils import _bufferize_attributes


class DirichletPrior(Prior, Dirichlet):
    """
    Dirichlet prior.
    """

    def __init__(self, concentration, validate_args=None, transform=None):
        TModule.__init__(self)
        Dirichlet.__init__(self, concentration, validate_args=validate_args)
        _bufferize_attributes(self, ("concentration",))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return DirichletPrior(self.concentration.expand(batch_shape))


class TruncatedNormalPrior(NormalPrior):
    """
    Normal distribution truncated at zero. Samples will be correct,
    nothing else is.

    Mean is fixed at zero. Uncomment the code in sample to support
    non-zero means. It is currently commented out since it is very slow
    in higher dimensions; there is probably a better way to implement
    it.
    """

    def sample(self, sample_shape):
        # n_target_samples = sample_shape[0]
        # samples = []
        # n_samples = 0
        # while n_samples < n_target_samples:
        #     s = super().sample(sample_shape)
        #     mask = s > 0
        #     while mask.ndimension() > 1:
        #         mask = mask.all(dim=-1)
        #     s_ = s[mask]
        #     samples.append(s_)
        #     n_samples += s_.size(0)
        # samples = torch.cat(samples, dim=0)
        # return samples[0:n_target_samples]
        return super().sample(sample_shape).abs_()

    def log_prob(self, x):
        raise NotImplementedError


def automatic_prior_specification(
        dimensions: int = 1,
        n_mixtures: int = 1,
        nyquist_frequencies: Tensor = None
) -> Mapping:
    if nyquist_frequencies is None:
        raise RuntimeError('Nyquist frequencies must be provided.')
    specification = {
        'weights': {
            'type': 'dirichlet',
            'parameters': {
                'concentration': [1] * n_mixtures
            }
        },
        'means': {
            'type': 'normal',
            'parameters': {
                'loc': [0] * dimensions * n_mixtures,
                'scale': (nyquist_frequencies / 5).tolist() * n_mixtures
            }
        },
        'scales': {
            'type': 'lognormal',
            'parameters': {
                'loc': [0] * dimensions * n_mixtures,
                'scale': [0.35] * dimensions * n_mixtures
            }
        }
    }
    return specification


def build_prior(
        type: str = 'normal',
        parameters: Mapping[str, Sequence[int]] = None,
        device: torch.device = torch.device('cpu')
) -> Prior:
    if type == 'uniform':
        prior = UniformPrior(
            torch.tensor(parameters['lower']).to(
                device=device, dtype=torch.get_default_dtype()
            ),
            torch.tensor(parameters['upper']).to(
                device=device, dtype=torch.get_default_dtype()
            )
        )
    elif type == 'normal':
        prior = TruncatedNormalPrior(
            torch.tensor(parameters['loc']).to(
                device=device, dtype=torch.get_default_dtype()
            ),
            torch.tensor(parameters['scale']).to(
                device=device, dtype=torch.get_default_dtype()
            )
        )
    elif type == 'lognormal':
        prior = LogNormalPrior(
            torch.tensor(parameters['loc']).to(
                device=device, dtype=torch.get_default_dtype()
            ),
            torch.tensor(parameters['scale']).to(
                device=device, dtype=torch.get_default_dtype()
            )
        )
    elif type == 'dirichlet':
        prior = DirichletPrior(
            torch.tensor(parameters['concentration']).to(
                device=device, dtype=torch.get_default_dtype()
            )
        )
    else:
        raise NotImplementedError(f'{type} prior not implemented')

    return prior
