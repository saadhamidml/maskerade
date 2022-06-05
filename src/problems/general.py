"""Problem agnostic code for setting up problems."""
from pathlib import Path
from typing import Tuple
from math import ceil
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle
import matplotlib.pyplot as plt


def attribute_target_split(
        dataframe: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split Dataframe into attributes and targets, assuming that
    targets are the last column of the dataframe.
    """

    attributes = dataframe.iloc[:, :-1]
    targets = dataframe.iloc[:, -1]
    return attributes, targets


def dataframe_to_tensor(dataframe: pd.DataFrame) -> torch.Tensor:
    """Returns a torch tensor from a pandas dataframe."""
    return torch.tensor(
        dataframe.values
    ).to(torch.get_default_dtype()).squeeze()


def prepare(dataframe: pd.DataFrame, *args, **kwargs):
    """Prepare dataset and define training data preprocessing steps."""
    return MinMaxScaler(), StandardScaler()
    # return StandardScaler(), StandardScaler()


def segregate(
        dataframe: pd.DataFrame,
        feature_preprocessor=None,
        target_preprocessor=None,
        test_size: float = 0.2,
        shuffle: bool = False,
        cross_validation: int = None,
        use_cuda: bool = True,
        **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Dataframe into train and test sets, and apply preprocessor.

    preprocessor.fit_transform is applied to train_y and
    preprocessor.transform to test_y.

    cross_validation is assumed to be index from 1 to k.
    """
    if cross_validation is not None:
        if shuffle:
            dataframe = pd.DataFrame(sklearn_shuffle(dataframe.to_numpy()))
        # ceil to be consistent with train_test_split
        n_test_points = ceil(len(dataframe) * test_size)
        test_indices = np.arange(
            (cross_validation - 1) * n_test_points,
            cross_validation * n_test_points
        )
        train_indices = dataframe.index.drop(
            test_indices.tolist(), errors='ignore'
        ).to_numpy()
        reindexing = np.concatenate((train_indices, test_indices)).tolist()
        dataframe = dataframe.reindex(reindexing).dropna()
        train_set, test_set = train_test_split(
            dataframe,
            test_size=test_size,
            shuffle=False
        )
    else:
        train_set, test_set = train_test_split(
            dataframe,
            test_size=test_size,
            shuffle=shuffle
        )
    n_features = len(dataframe.columns) - 1

    train_x, train_y = attribute_target_split(train_set)
    test_x, test_y = attribute_target_split(test_set)
    if feature_preprocessor is not None:
        train_x = pd.DataFrame(
            feature_preprocessor.fit_transform(
                train_x.to_numpy().reshape(-1, n_features)
            )
        )
        test_x = pd.DataFrame(
            feature_preprocessor.transform(
                test_x.to_numpy().reshape(-1, n_features)
            )
        )
    if target_preprocessor is not None:
        train_y = pd.DataFrame(
            target_preprocessor.fit_transform(train_y.to_numpy().reshape(-1, 1))
        )
        test_y = pd.DataFrame(
            target_preprocessor.transform(test_y.to_numpy().reshape(-1, 1))
        )
    if use_cuda:
        map_function = lambda i: dataframe_to_tensor(i).cuda()
    else:
        map_function = dataframe_to_tensor
    (train_x, train_y, test_x, test_y) = map(
        map_function,
        (train_x, train_y, test_x, test_y)
    )
    return train_x, train_y, test_x, test_y


def visualise(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    log_dir: Path('./')
):
    """Plot train and test sets.
    
    :param train_inputs: Train inputs, shape [N, D].
    :param train_targets: Train targets, shape [N, D].
    :param test_inputs: Test inputs, shape [M, D].
    :param test_targets: Test targets, shape [M, D].
    """
    if train_inputs.ndimension() == 1:
        train_inputs = train_inputs.unsqueeze(1)
        test_inputs = test_inputs.unsqueeze(1)
    for d in range(train_inputs.size(1)):
        fig, ax = plt.subplots(1, 1)
        ax.plot(train_inputs[:, d].cpu().numpy(), train_targets.cpu().numpy(), 'b.')
        ax.plot(test_inputs[:, d].cpu().numpy(), test_targets.cpu().numpy(), 'r.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.tight_layout()
        fig.savefig(log_dir / f'data_{d}.pdf')
        plt.close(fig)
