from typing import Union, Tuple
import torch
from torch import Tensor
from gpytorch import settings
from gpytorch.lazy import LazyEvaluatedKernelTensor, lazify
from gpytorch.lazy.keops_lazy_tensor import KeOpsLazyTensor
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.kernels.keops.keops_kernel import KeOpsKernel
from pykeops.torch import LazyTensor as KEOLazyTensor


class InterDomainKernel(Kernel):
    """Can handle inputs with different numbers of dimensions."""
    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        """Same as super.__call__ but doesn't raise exception if inputs
        are not the same shape.
        """

        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        if x2_ is None:
            x2_ = x1_

        if x1_.ndimension() < 2 or x2_.ndimension() < 2:
            raise RuntimeError('Kernel inputs must be at least 2D')

        # Check that ard_num_dims matches the supplied number of dimensions
        if settings.debug.on():
            if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
                raise RuntimeError(
                    "Expected the input to have {} dimensionality "
                    "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims,
                                                                           x1_.size(-1))
                )

        if diag:
            res = super(Kernel, self).__call__(x1_, x2_, diag=True,
                                               last_dim_is_batch=last_dim_is_batch, **params)
            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size(
                        (x1_.size(-2), x2_.size(-2))):
                    res = res.diag()
            return res

        else:
            if settings.lazily_evaluate_kernels.on():
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self,
                                                last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = lazify(
                    super(Kernel, self).__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch,
                                                 **params))
            return res


class InterDomainScaleKernel(InterDomainKernel, ScaleKernel):
    """Overrides standard __call__ function to allow inputs of different
    shapes. __init__ method of ScaleKernel will be used.
    """


def mixing_measure(
        gmm: Tensor,
        num_components: int,
        dimensions: int,
        concatenate_means_scales: bool = True
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Converts GMM in format (B, N(1 + 2D)) to mixing measure (B, N)
    and locations (B, N, 2D)
    """
    weights = gmm[:, :num_components]
    mid_index = -int(num_components * dimensions)
    means = gmm[
        :, num_components:mid_index
    ].view(gmm.size(0), num_components, dimensions)
    scales = gmm[
        :, mid_index:
    ].view(gmm.size(0), num_components, dimensions)
    if concatenate_means_scales:
        locations = torch.cat((means, scales), dim=2)
        return weights, locations
    else:
        return weights, means, scales


class MMDKernel(InterDomainKernel):
    """Gibbs kernel based on the Sinkhorn Distance for GMMs with
    diagonal covariance structure.
    """

    has_lengthscale = True

    def __init__(
            self,
            dimensions=1,
            ard_num_dims=None,
            batch_shape=torch.Size([]),
            active_dims=None,
            lengthscale_prior=None,
            lengthscale_constraint=None,
            eps=1e-6,
            **kwargs,
    ):
        # TODO: Use ard_num_dims instead of dimensions.
        self.dimensions = dimensions
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            eps=eps,
            **kwargs
        )

    def compute_distances(
            self,
            x1,
            x1_num_components,
            x2=None,
            x2_num_components=None,
            diag=False
    ):
        # Separate out the weights, locations and scales
        x1_weights, x1_locations = mixing_measure(
            x1,
            x1_num_components,
            self.dimensions,
            concatenate_means_scales=True
        )
        if x2 is not None and x2_num_components is not None:
            x2_weights, x2_locations = mixing_measure(
                x2,
                x2_num_components,
                self.dimensions,
                concatenate_means_scales=True
            )
        else:
            x2_weights = x1_weights
            x2_locations = x1_locations

        # TODO: If batch size is too large then use torch dataloader.
        location_distances = self.covar_dist(
            x1_locations, x2_locations, diag=diag
        )
        distances = torch.einsum(
            'ij,ijk,ik->i', x1_weights, location_distances, x2_weights
        )

        return distances

    def forward(self, x1, x2, diag=False, **params):
        if x1.size() == x2.size() and torch.allclose(x1, x2):
            identical = True
        else:
            identical = False

        # Work out the number of components in each set of GMMs
        denominator = 1 + 2 * self.dimensions
        x1_num_components = int(x1.size(1) / denominator)
        x2_num_components = int(x2.size(1) / denominator)

        if identical:
            # Copy the data appropriately so that each pair of x1_, x2_ can
            # be used to compute an element of the upper triange of the
            # covariance matrix
            indices = torch.triu_indices(row=x1.size(0), col=x2.size(0))
            x1_ = x1[indices[0]]
            x2_ = x2[indices[1]]
        else:
            x1_ = torch.repeat_interleave(x1, repeats=x2.size(0), dim=0)
            x2_ = x2.repeat(x1.size(0), 1)

        distances = self.compute_distances(
            x1_, x1_num_components, x2_, x2_num_components, diag=diag
        )

        if identical:
            # Reshape divergences
            distances_ = torch.empty(x1.size(0), x2.size(0))
            mask = torch.triu(torch.ones(x1.size(0), x2.size(0), dtype=torch.bool))
            distances_[mask] = distances
            distances_ = (
                torch.triu(distances_)
                + torch.triu(
                    distances_, diagonal=1
                ).transpose(dim0=0, dim1=1)
            )
        else:
            distances_ = distances.view(x1.size(0), x2.size(0))

        if identical:
            diagonal = distances_.diag()
            distances_ -= (diagonal.view(-1, 1) + diagonal.view(1, -1)) / 2
        else:
            x1_diagonal = self.compute_distances(x1, x1_num_components)
            x2_diagonal = self.compute_distances(x2, x2_num_components)
            distances_ -= (
                x1_diagonal.view(-1, 1) + x2_diagonal.view(1, -1)
            ) / 2

        return distances_.div_(-self.lengthscale).exp_()
