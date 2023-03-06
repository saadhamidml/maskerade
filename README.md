# MASKERADE

## Installation
Run the following commands from the project root:

docker build -t maskerade .

docker run -it -u $(id -u):$(id -g) -v $(pwd):/maskerade --gpus=all maskerade


## Running experiments
The working directory should be /maskerade/src/.

For a single run the command should look like "python -u main.py with configs/{config}.yaml --force".

To repeat runs use repeat_runs.sh. The command should look like "./repeat_runs.sh -c configs/{config}.yaml -r {num_repeats}". If you wish to cross validate then add the "-x" flag.

All configurations are specified in yaml files organised in the form configs/{problem}/{method}.yaml.

Experiment results will be saved in logs/{experiment_id}/metrics.json. Summary statistics and IDs for repeated experiments will be saved to logs/collection\_{problem}\_{method}\_{datetime}/.
