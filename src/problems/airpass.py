"""Code specific to setting up synthetic function problems."""
from typing import Union
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Use standard prepare and segregate functions
from .general import segregate


def ingest(
        data_dir: Union[Path, str] = Path('./'),
        remove_linear_trend: bool = True,
        **kwargs
):
    """Generate data."""
    data_url = 'https://github.com/robjhyndman/fma/blob/master/data/airpass.rda?raw=true'
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    dataset_dir = data_dir / 'airpass'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / 'airpass.rda'
    if not dataset_path.exists():
        content = requests.get(data_url).content
        open(dataset_path, 'wb').write(content)
    robjects.r['load'](str(dataset_path))
    with localconverter(robjects.default_converter + pandas2ri.converter):
        y_data = robjects.conversion.rpy2py(robjects.r['airpass'])
    x_data = np.arange(len(y_data)).reshape(-1, 1)
    if remove_linear_trend:
        # y_data = np.diff(y_data)
        model = LinearRegression()
        model.fit(x_data, y_data)
        # calculate trend
        trend = model.predict(x_data)
        y_data = y_data - trend
    y_data = y_data.reshape(-1, 1)
    # np.float32 so it plays nicely with PyTorch and GPyTorch
    default_type = torch.get_default_dtype()
    if default_type == torch.float32:
        np_type = np.float32
    elif default_type == torch.float64:
        np_type = np.float64
    else:
        raise NotImplementedError('Set an appropriate PyTorch default dtype.')
    return pd.DataFrame(
        np.concatenate((x_data, y_data), axis=1),
        dtype=np_type
    )


def prepare(dataframe: pd.DataFrame, *args, **kwargs):
    """Prepare dataset and define training data preprocessing steps."""
    # Scale features so that nyquist frequency is 1.
    # Assume evenly spaced, warp so spacing is 0.5 units
    max_value = 48
    return MinMaxScaler(feature_range=(0, max_value)), StandardScaler()
