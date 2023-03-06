import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


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
        print(id)
        metrics = load_json(str(log_dir / str(id) / 'metrics.json'))
        lls.append(np.array(metrics['test_data_log_marginal_likelihood']['values']).reshape((-1,1)))
        rmses.append(np.array(metrics['test_rmse']['values']).reshape((-1,1)))
    max_len = max([res.size for res in lls])
    lls = [np.vstack((res, np.ones((max(0, max_len - res.size),1)) * res[-1]))  for res in lls]
    lls = np.concatenate(lls, axis=1)
    rmses = [np.vstack((res, np.ones((max(0, max_len - res.size),1)) * res[-1])) for res in rmses]
    rmses = np.concatenate(rmses, axis=1)
    mses = rmses**2
    if summary_statistics:
        return (
            lls.mean(axis=1),
            lls.std(axis=1),
            rmses.mean(axis=1),
            rmses.std(axis=1),
            mses.mean(axis=1),
            mses.std(axis=1)
        )
    else:
        return lls, rmses


def save_collection_metrics(
        collection_dir, ll_means, ll_stddevs, rmse_means, rmse_stddevs, mse_means, mse_stddevs
):
    file = open(collection_dir / 'metrics.txt', 'w')
    file.write(
        f'Test Set Log Likelihood:\n {pd.DataFrame([ll_means,ll_stddevs], index=["LL-mean", "LL-std"]).T} \n\n'
    )
    file.write(
        f'Test Set RMSE:\n {pd.DataFrame([rmse_means,rmse_stddevs], index=["RMSE-mean", "RMSE-std"]).T} \n\n'
    )
    file.write(
        f'Test Set MSE:\n {pd.DataFrame([mse_means,mse_stddevs], index=["MSE-mean", "MSE-std"]).T}\n\n'
    )
    file.close()


for collection in collections:
    ll_means, ll_stddevs, rmse_means, rmse_stddevs, mse_means, mse_stddevs = training_curves(
        collection, summary_statistics=True
    )
    save_collection_metrics(
        collection, ll_means, ll_stddevs, rmse_means, rmse_stddevs, mse_means, mse_stddevs
    )
