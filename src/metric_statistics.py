import argparse
from pathlib import Path
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Set up argument parser
parser = argparse.ArgumentParser(description='Parse input directories')
parser.add_argument('collection_dir')
args = parser.parse_args()
collection_dir = Path('../logs') / args.collection_dir
collections = (collection_dir,)

log_dir = Path('../logs')


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def training_curves(collection_dir, summary_statistics=False):
    ids = pd.read_csv(collection_dir / 'run_ids.txt', header=None).values.flatten().tolist()
    lls = []
    rmses = []
    for id in ids:
        metrics = load_json(str(log_dir / str(id) / 'metrics.json'))
        try:
            lls.append(np.array(metrics['test_data_log_marginal_likelihood']['values']))
            rmses.append(np.array(metrics['test_rmse']['values']))
        except:
            pass
    lls = np.concatenate(lls, axis=0)
    rmses = np.concatenate(rmses, axis=0)
    mses = rmses**2
    if summary_statistics:
        return (
            lls.mean(axis=0),
            lls.std(axis=0),
            rmses.mean(axis=0),
            rmses.std(axis=0),
            mses.mean(axis=0),
            mses.std(axis=0)
        )
    else:
        return lls, rmses


def save_collection_metrics(
        collection_dir, ll_means, ll_stddevs, rmse_means, rmse_stddevs, mse_means, mse_stddevs
):
    file = open(collection_dir / 'metrics.txt', 'w')
    file.write(
        f'Test Set Log Likelihood: {ll_means} $\pm$ {ll_stddevs}\n'
    )
    file.write(
        f'Test Set RMSE: {rmse_means} $\pm$ {rmse_stddevs}\n'
    )
    file.write(
        f'Test Set MSE: {mse_means} $\pm$ {mse_stddevs}\n'
    )
    file.close()


for collection in collections:
    ll_means, ll_stddevs, rmse_means, rmse_stddevs, mse_means, mse_stddevs = training_curves(
        collection, summary_statistics=True
    )
    save_collection_metrics(
        collection, ll_means, ll_stddevs, rmse_means, rmse_stddevs, mse_means, mse_stddevs
    )
