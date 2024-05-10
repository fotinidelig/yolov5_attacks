from pathlib import Path
import configparser
import mlflow
import torch
import os

FILE_PATH = Path(os.path.abspath(__file__)).parent
CONFIG_FILE = FILE_PATH / ".." / ".." / "attack_config.ini"
if not os.path.exists(CONFIG_FILE):
    raise RuntimeError("config file not found")


def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def info(msg: str):
    print(f"[INFO]--- {msg}\n")


def read_config(file: str):
    config = configparser.ConfigParser()
    config.read(file)
    return config


def run_experiment(experiment_func, experiment_kwargs, experiment_name="test", run_name="test_run"):
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    info(f"Experiment ID: {experiment_id}")
    with mlflow.start_run(run_name=run_name):
        if experiment_kwargs:
            experiment_func(experiment_kwargs)
        else:
            experiment_func()
    return experiment_id


def cuda_info(msg: str=''):
    info(msg+"Memory stats: allocated / reserved memory: " + str(torch.cuda.memory_allocated() / 1024 ** 2) +
         "/ " + str(torch.cuda.memory_reserved() / 1024 ** 2) + "(MB)")