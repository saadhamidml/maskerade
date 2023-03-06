"""Collect configuration information and run experiment."""
import importlib
import os
from pathlib import Path
from typing import Union
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch

torch.set_printoptions(linewidth=200)

# Create Sacred experiment.
ex = Experiment()
# Turn off stdout capturing; apparently MongoDB record size limit is 16MB.
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

# Default configuration
@ex.config
def configuration():
    experiment = {
        'directories': {'data': '../data', 'logs': '../logs'},
        'default_data_type': 'float64',
        'disable_cuda': False,
        'cuda_device': 0,
        'smoke_test': False,
    }


# Add observer to which experiment configuration and results will be added.
@ex.capture(prefix='experiment.directories')
def add_observer(logs: Union[Path, str] = '../logs'):
    if not isinstance(logs, Path):
        logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    ex.observers.append(FileStorageObserver(logs))


add_observer()


@ex.automain
def run_experiment(_config=None, _run=None):
    """Experiment runner. _run is a Sacred run object. This is automatically
    filled in by sacred.
    """
    # Set up logging.
    log_dir = (
        Path(_config['experiment']['directories']['logs']) / str(_run._id)
    )
    # If running repeats.
    if _config.get('collection', None) is not None:
        collection_dir = (
            Path(_config['experiment']['directories']['logs'])
            / str(_config['collection'])
        )
        collection_dir.mkdir(parents=True, exist_ok=True)
        filename = collection_dir / 'run_ids.txt'
        append_write = 'a' if os.path.exists(filename) else 'w'
        file = open(filename, append_write)
        file.write(f'{_run._id}\n')
        file.close()
        seeds_filename = collection_dir / 'seeds.txt'
        seeds_file = open(seeds_filename, append_write)
        seeds_file.write(f"{_config['seed']}\n")
        seeds_file.close()

    # Set up torch
    if _config['experiment']['default_data_type'] == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available() and not _config['experiment']['disable_cuda']:
        use_cuda = True
        default_data_type = torch.get_default_dtype()
        if default_data_type == torch.float64:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(_config['experiment']['cuda_device'])
    else:
        use_cuda = False

    # Download, prepare and segregate data.
    problem_name = _config['problem']['name']
    problem = importlib.import_module('problems.' + problem_name)
    dataframe = problem.ingest(
        data_dir=_config['experiment']['directories']['data'],
        **_config['problem']
    )
    feature_preprocessor, target_preprocessor = problem.prepare(dataframe)
    train_x, train_y, test_x, test_y = problem.segregate(
        dataframe,
        feature_preprocessor=feature_preprocessor,
        target_preprocessor=target_preprocessor,
        cross_validation=_config.get('cross_validate', None),
        use_cuda=use_cuda,
        **_config['problem']
    )

    # Get model components.
    model_type = _config['model']['type']
    model_module = importlib.import_module('models.' + model_type)
    model_components = model_module.build_model(
        train_inputs=train_x,
        train_targets=train_y,
        sacred_run=_run,
        use_cuda=use_cuda,
        **_config['model']
    )

    # Learn or load learnt model
    learn_output = model_module.learn(
        **model_components,
        train_inputs=train_x,
        train_targets=train_y,
        test_inputs=test_x,
        test_targets=test_y,
        log_dir=log_dir,
        sacred_run=_run,
        use_cuda=use_cuda,
        smoke_test=_config['experiment']['smoke_test'],
        numerics=_config['numerics']
    )

    # Do inference in the model
    # if not _config['numerics']['infer_while_learning']:
    infer_output = model_module.infer(
        **model_components,
        learn_output=learn_output,
        test_inputs=test_x,
        test_targets=test_y,
        numerics=_config['numerics'],
        sacred_run=_run,
        log_dir=log_dir,
        use_cuda=use_cuda
    )
