import torch
from torch import Tensor

def cholesky_update(
        cholesky: Tensor, cross_covariance: Tensor, self_covariance: Tensor
) -> Tensor:
    """Update lower cholesky decomposition with a single new entry.
    cross_covariance is the covariance between the new entry and all
    previous data. self_covariance is the variance of the new
    entry.
    """
    cross_covariance = cross_covariance.view(cholesky.size(0), 1)
    self_covariance = self_covariance.view(1, 1)

    new_row = torch.solve(cross_covariance, cholesky)[0].transpose(0, 1)
    new_element = torch.sqrt(
        self_covariance
        - cross_covariance.transpose(0, 1)
        @ torch.cholesky_solve(cross_covariance, cholesky)
    )

    updated_top = torch.cat(
        (cholesky, torch.zeros_like(cross_covariance)), dim=1
    )
    updated_bottom = torch.cat((new_row, new_element), dim=1)
    return torch.cat((updated_top, updated_bottom), dim=0)


def max_distances(train_inputs):
    if train_inputs.ndimension() == 1:
        train_inputs = train_inputs.unsqueeze(1)
    sorted = train_inputs.sort(dim=0)[0]
    return sorted[-1, :] - sorted[0, :]

def nyquist_frequencies(train_inputs: Tensor, max_distances: Tensor, assume_even: bool = True):
    """Helper method to compute Nyquist frequencies of each dimension.

    assume_even assumes train_inputs are evenly spread across domain,
    and that the input data has been normalised to [0, 1] in every
    dimension.
    """
    sorted = train_inputs.sort(dim=0)[0]
    differences = sorted[1:] - sorted[:-1]
    max = differences.max()
    dezeroed = torch.where(
        differences == 0, max * torch.ones_like(differences), differences
    )
    minimum_distances = dezeroed.min(dim=0)[0]
    if assume_even:
        if train_inputs.ndimension() == 1:
            train_inputs = train_inputs.unsqueeze(1)
        n_train_inputs = torch.tensor([
            torch.unique(train_inputs[:, i]).size(0)
            for i in range(train_inputs.size(1))
        ]).to(train_inputs)
        even_distances = max_distances / n_train_inputs
    else:
        even_distances = minimum_distances
    return 0.5 / torch.max(minimum_distances, even_distances)
