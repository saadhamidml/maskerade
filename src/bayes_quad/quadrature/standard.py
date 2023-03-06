"""Provides models for Bayesian Quadrature"""
from typing import Tuple
import warnings
from multimethod import multimethod
import numpy as np
from numpy import newaxis
from math import sqrt, pi
from scipy.special import erf
from scipy.linalg import cho_solve
import torch
from torch.distributions import MultivariateNormal
from gpytorch.models import GP
from gpytorch.priors import Prior, UniformPrior, NormalPrior
from gpytorch.means import Mean, ConstantMean
from gpytorch.kernels import Kernel, RBFKernel

from bayes_quad.surrogates import GPModel, WsabiLGPModel


class IntegrandModel:
    """Combination of GPyTorch prior and surrogate model."""
    def __init__(
            self,
            prior: Tuple[Prior] = (UniformPrior,),
            surrogate: GP = WsabiLGPModel,
            **kwargs
    ):
        self.prior = prior
        self.surrogate = surrogate

    def integral_mean(self) -> float:
        """Compute integral mean."""
        return _compute_mean(
            self.prior,
            self.surrogate,
            self.surrogate.mean_module,
            self.surrogate.covar_module.base_kernel
        )

    def integral_variance(self) -> float:
        """Compute integral variance."""
        return _compute_variance(
            self.prior,
            self.surrogate,
            self.surrogate.covar_module.base_kernel
        )


@multimethod
def _compute_mean(
        prior: Tuple[Prior],
        surrogate: GPModel,
        mean: Mean,
        kernel: Kernel
) -> float:
    """Compute the mean of the integral for the given prior, warped
    GP, and kernel. This method will delegate to other methods of
    the same name defined in this module, based on the type of the
    arguments. If no implementation is found for the provided types,
    this default implementation will raise an error.
    """
    raise NotImplementedError(
        'Integration is not supported for this combination of prior,'
        'warping, mean function and kernel.\n\n'
        f'Prior was of type {prior}.\n'
        f'Warped GP was of type {surrogate}.\n'
        f'Mean function was of type {mean}.\n'
        f'Kernel was of type {kernel}.'
    )

@multimethod
def _compute_variance(
        prior: Tuple[Prior],
        surrogate: GPModel,
        kernel: Kernel
) -> float:
    """Compute the variance of the integral for the given prior, GP,
    and kernel. This method will delegate to other methods of
    the same name defined in this module, based on the type of the
    arguments. If no implementation is found for the provided types,
    this default implementation will raise an error.
    """
    raise NotImplementedError(
        'Integration is not supported for this combination of prior,'
        'warping, mean function and kernel.\n\n'
        f'Prior was of type {prior}.\n'
        f'Warped GP was of type {surrogate}.\n'
        f'Kernel was of type {kernel}.'
    )
