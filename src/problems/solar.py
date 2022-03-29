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
from .general import prepare, segregate


def ingest(
        data_dir: Union[Path, str] = Path('./'),
        remove_linear_trend: bool = False,
        **kwargs
):
    """Generate data."""
    data_url = 'ftp://ftp.ncdc.noaa.gov/pub/data/paleo/climate_forcing/solar_variability/lean2000_irradiance.txt'
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    dataset_dir = data_dir / 'solar'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / 'solar.csv'
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
        skiprows=69,
        usecols=[0, 2],
        delim_whitespace=True,
        dtype=np_type
    )
    if remove_linear_trend:
        dataframe['11yrCYCLE'] = dataframe['11yrCYCLE'].diff()
    dataframe = dataframe.dropna()
    arange_twenty = np.arange(20)
    test_indices = np.concatenate((
        arange_twenty + 16,
        arange_twenty + 92,
        arange_twenty + 166,
        arange_twenty + 242,
        arange_twenty + 316,
    ))
    train_indices = dataframe.index.drop(test_indices.tolist()).to_numpy()
    reindexing = np.concatenate((train_indices, test_indices)).tolist()
    return dataframe.reindex(reindexing)
