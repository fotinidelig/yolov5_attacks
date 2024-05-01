from pathlib import Path
import mlflow
import os

FILE_PATH = Path(os.path.abspath(__file__)).parent
CONFIG_FILE = FILE_PATH / ".." / ".." / "attack_config.ini"

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def info(msg):
    print(f"[INFO]--- {msg}\n")


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