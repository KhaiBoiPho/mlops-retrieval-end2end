from dotenv import load_dotenv
import os
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

_INITIALIZED = False

def init_mlflow(experiment_name: str | None = None):
    global _INITIALIZED
    if _INITIALIZED:
        return
    
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    _INITIALIZED = True

def get_mlflow_client(experiment_name: str | None = None) -> MlflowClient:
    init_mlflow(experiment_name=experiment_name)
    return MlflowClient()