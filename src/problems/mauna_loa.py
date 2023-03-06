"""Code specific to setting up synthetic function problems."""
from typing import Union
from pathlib import Path
import shutil
import urllib.request as request
from contextlib import closing
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Use standard prepare and segregate functions
from .general import segregate


def ingest(
        data_dir: Union[Path, str] = Path('./'),
        remove_linear_trend: bool = True,
        **kwargs
):
    """Generate data."""
    data_url = 'ftp://aftp.cmdl.noaa.gov/ccg/co2/trends/co2_mm_gl.csv'
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    dataset_dir = data_dir / 'mauna_loa'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / 'mauna_loa.csv'
    if not dataset_path.exists():
        with closing(request.urlopen(data_url)) as r:
            with open(dataset_path, 'wb') as f:
                shutil.copyfileobj(r, f)
    # set dtype so it plays nicely with PyTorch and GPyTorch
    default_type = torch.get_default_dtype()
    if default_type == torch.float32:
        np_type = np.float32
    elif default_type == torch.float64:
        np_type = np.float64
    else:
        raise NotImplementedError('Set an appropriate PyTorch default dtype.')
    dataframe = pd.read_csv(
        dataset_path,
        skiprows=56,
        usecols=['decimal', 'average'],
        dtype=np_type
    )
    if remove_linear_trend:
        dataframe['average'] = dataframe['average'].diff()
    dataframe = dataframe.dropna()
    return dataframe


def prepare(dataframe: pd.DataFrame, *args, **kwargs):
    """Prepare dataset and define training data preprocessing steps."""
    # Scale features so that nyquist frequency is 1.
    # Assume evenly spaced, warp so spacing is 0.5 units
    # max_value = 161
    max_value = 0.66
    return MinMaxScaler(feature_range=(0, max_value)), StandardScaler()
