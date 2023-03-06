"""Set up challenger dataset."""
from typing import Union
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

# Use standard prepare and segregate functions
from .general import segregate

def prepare(dataframe: pd.DataFrame, *args, **kwargs):
    """Prepare dataset and define training data preprocessing steps."""
    return MinMaxScaler(), StandardScaler()


def ingest(data_dir: Union[Path, str] = Path('./'), **kwargs):
    """Generate data."""
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    dataset_dir = data_dir / 'yacht'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / 'yacht.csv'
    if not dataset_path.exists():
        content = requests.get(data_url).content
        open(dataset_path, 'wb').write(content)

    dataframe = pd.read_csv(dataset_path, delim_whitespace=True, header=None)
    # data type setting so it plays nicely with PyTorch and GPyTorch
    default_type = torch.get_default_dtype()
    if default_type == torch.float32:
        np_type = np.float32
    elif default_type == torch.float64:
        np_type = np.float64
    else:
        raise NotImplementedError('Set an appropriate PyTorch default dtype.')
    return dataframe.astype(np_type)
