from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Config file paths
CONFIG_DIR = PROJECT_ROOT / "config"

DATA_CONFIG_FILE_PATH = CONFIG_DIR / "data.yaml"

# Bi-encoder configs
BI_ENCODER_TRAIN_CONFIG_PATH = CONFIG_DIR / "bi-encoder" / "train.yaml"
BI_ENCODER_EVAL_CONFIG_PATH = CONFIG_DIR / "bi-encoder" / "eval.yaml"
BI_ENCODER_SERVE_CONFIG_PATH = CONFIG_DIR / "bi-encoder" / "serve.yaml"

# Cross-encoder configs
CROSS_ENCODER_TRAIN_CONFIG_PATH = CONFIG_DIR / "cross-encoder" / "train.yaml"
CROSS_ENCODER_EVAL_CONFIG_PATH = CONFIG_DIR / "cross-encoder" / "eval.yaml"
CROSS_ENCODER_SERVE_CONFIG_PATH = CONFIG_DIR / "cross-encoder" / "serve.yaml"

# Other configs
CORPUS_EMBEDDINGS_CONFIG_PATH = CONFIG_DIR / "corpus_embeddings.yaml"
HARD_NEGATIVE_MINING_CONFIG_PATH = CONFIG_DIR / "hard_negative_mining.yaml"
MONITORING_CONFIG_PATH = CONFIG_DIR / "monitoring.yaml"
ROLLBACK_CONFIG_PATH = CONFIG_DIR / "rollback.yaml"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
USE_DATA_DIR = DATA_DIR / "use"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# MLflow
MLFLOW_TRACKING_URI = "https://dagshub.com/KhaiBoiPho/legal-retrieval-mlops.mlflow"