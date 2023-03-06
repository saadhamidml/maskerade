from typing import Sequence, Mapping
import math
import numpy as np
from scipy.stats import truncnorm
from scipy.stats.qmc import Sobol
import torch
from torch import Tensor, ceil
from torch.nn import Module as TModule
from torch.distributions import (
    Dirichlet,
    ExponentialFamily,
    MultivariateNormal,
    Normal,
    TransformedDistribution
)
from torch.distributions.transforms import AbsTransform
from gpytorch.priors import (
    Prior, UniformPrior, NormalPrior, LogNormalPrior as GLogNormalPrior
)
from gpytorch.priors.utils import _bufferize_attributes


class InverseWishartPrior(Prior, ExponentialFamily):
    def __init__(self, df, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        ExponentialFamily.__init__(self, validate_args=validate_args)
        self.df = df
        self.dim = scale.size(0)  # i.e. this class is a prior over dim x dim matrices.
        self.scale = scale
        self.scale_tril = torch.cholesky(scale)
        self.mvn = MultivariateNormal(
            loc=torch.zeros(scale.size(0)),
            precision_matrix=scale
        )
        self._transform = transform

    def sample(self, sample_shape=torch.Size()):
        """Returns cholesky decompositions of samples."""
        # Draw samples from corresponding Wishart and then invert.
        mvn_samples = self.mvn.sample(sample_shape=(*sample_shape, self.df))
        outer_prods = mvn_samples.unsqueeze(-1) @ mvn_samples.unsqueeze(-2)
        wishart_samples = outer_prods.sum(-3)
        chols = torch.cholesky(wishart_samples)
        tril_indices = torch.tril_indices(row=self.dim, col=self.dim)
        return torch.cholesky(torch.cholesky_inverse(chols))[..., tril_indices[0], tril_indices[1]]

    def low_discrepancy_sample(self, sample_shape=torch.Size()):
        raise NotImplementedError
        sobol_dim = self.dim * (self.dim + 1) / 2
        sobol_seqs = Sobol(d=sobol_dim).random_base2(
            m=int(math.log(sample_shape.prod().item(), 2))
        )
        dfs = [self.df - i + 2 for i in range(self.dim)]
        chi2_ppf = lambda i, df: chi2_ppf(sobol_seq[:, i], df)
        chi2_samples = map(chi2_ppf, enumerate(dfs))
        normal_samples = Normal(0, 1).icdf(sobol_seq[:, self.dim:])

        chols = torch.cholesky(wishart_samples)
        tril_indices = torch.tril_indices(row=self.scale.size(0), col=self.scale.size(1))
        return torch.cholesky(torch.cholesky_inverse(chols))[..., tril_indices[0], tril_indices[1]]

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return InverseWishartPrior(
            self.df.expand(batch_shape), self.scale.expand(batch_shape)
        )


class DirichletPrior(Prior, Dirichlet):
    """
    Dirichlet prior.
    """

    def __init__(self, concentration, validate_args=None, transform=None):
        TModule.__init__(self)
        Dirichlet.__init__(self, concentration, validate_args=validate_args)
        _bufferize_attributes(self, ("concentration",))
        self._transform = transform

    def low_discrepancy_sample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape=sample_shape)

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return DirichletPrior(self.concentration.expand(batch_shape))


class TruncatedNormal(TransformedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super(TruncatedNormal, self).__init__(
            base_dist, AbsTransform(), validate_args=validate_args
        )

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    def log_prob(self, x):
        assert (self.loc == 0).all()
        return self.base_dist.log_prob(x) + torch.tensor(2.).log()


class TruncatedNormalPrior(Prior, TruncatedNormal):
    """
    Log Normal prior.
    """

    def __init__(self, loc, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        TruncatedNormal.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        self._transform = transform

    def low_discrepancy_sample(self, sample_shape=torch.Size()):
        n_dimensions = torch.tensor(self.batch_shape).prod().item()
        n_samples = torch.tensor(sample_shape).prod().item()
        sobol_seq = Sobol(d=n_dimensions).random_base2(
            m=math.ceil(math.log2(float(n_samples)))
        )[:n_samples].reshape(
            *sample_shape, *self.batch_shape, *self.event_shape
        )
        # return self.icdf(torch.tensor(sobol_seq)).view(*sample_shape, -1)
        return torch.tensor(
            truncnorm.ppf(
                sobol_seq,
                np.zeros(n_dimensions).reshape(
                    *self.batch_shape, *self.event_shape
                ),
                10 * self.scale.cpu().numpy(),
                loc=self.loc.cpu().numpy(),
                scale=self.scale.cpu().numpy()
            )
        )

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return TruncatedNormalPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))

    def log_prob(self, x):
        return TruncatedNormal.log_prob(self, x)


class LogNormalPrior(GLogNormalPrior):
    def __init__(self, *args, **kwargs):
        super(LogNormalPrior, self).__init__(*args, **kwargs)

    def low_discrepancy_sample(self, sample_shape=torch.Size()):
        n_dimensions = torch.tensor(self.batch_shape).prod().item()
        n_samples = torch.tensor(sample_shape).prod().item()
        sobol_seq = Sobol(d=n_dimensions).random_base2(
            m=math.ceil(math.log2(float(n_samples)))
        )[:n_samples].reshape(
            *sample_shape, *self.batch_shape, *self.event_shape
        )
        return self.icdf(torch.tensor(sobol_seq))


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
