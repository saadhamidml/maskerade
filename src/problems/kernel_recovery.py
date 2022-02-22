"""Code specific to setting up synthetic function problems."""
from typing import Union, Sequence, Mapping
import torch
import pandas as pd
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.distributions import MultivariateNormal

# Use standard prepare and segregate functions
from .general import prepare, segregate


def ingest(
        mixture_weights: Sequence[Sequence[float]],
        mixture_means: Sequence[Sequence[float]],
        mixture_scales: Sequence[Sequence[float]],
        bounds: Mapping[str, float],
        density: int,
        noise: float,
        **kwargs
):
    """Generate data."""
    X = torch.linspace(bounds['lower'], bounds['upper'], density).unsqueeze_(1)
    X = X.roll(shifts=int(density / 3), dims=0)

    y = 0
    for w, m, s in zip(mixture_weights, mixture_means, mixture_scales):
        kernel = SpectralMixtureKernel(num_mixtures=len(w))
        kernel.mixture_weights = torch.tensor(w)
        kernel.mixture_means = torch.tensor(m)
        kernel.mixture_scales = torch.tensor(s)

        prior = MultivariateNormal(torch.zeros_like(X.squeeze()), kernel(X))
        y += prior.sample().view_as(X)

    y += torch.randn(X.size()) * noise

    return pd.DataFrame(torch.cat((X, y), dim=1).detach().cpu().numpy())
