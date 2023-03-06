"""Code specific to setting up synthetic function problems."""
from typing import Union
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import torch

# Use standard prepare and segregate functions
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

from .general import segregate

def prepare(dataframe: pd.DataFrame, *args, **kwargs):
    """Prepare dataset and define training data preprocessing steps."""
    return StandardScaler(), FunctionTransformer()


def ingest(data_dir: Union[Path, str] = Path('./'), **kwargs):
    """Generate data."""
    data_url = 'https://www.quandl.com/api/v3/datasets/BOE/XUDLBK82.csv?api_key=4pFBn9H7L5sPoeyKMGEe'
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    dataset_dir = data_dir / 'sterling'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / 'sterling.csv'
    if not dataset_path.exists():
        content = requests.get(data_url).content
        open(dataset_path, 'wb').write(content)

    dataframe = pd.read_csv(dataset_path)
    dataframe['Date'] = np.arange(len(dataframe.index))
    dataframe = dataframe.dropna()
    # data type setting so it plays nicely with PyTorch and GPyTorch
    default_type = torch.get_default_dtype()
    if default_type == torch.float32:
        np_type = np.float32
    elif default_type == torch.float64:
        np_type = np.float64
    else:
        raise NotImplementedError('Set an appropriate PyTorch default dtype.')
    return dataframe.iloc[::-1].astype(np_type)
