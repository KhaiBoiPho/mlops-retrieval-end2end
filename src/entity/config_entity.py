from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# ==================== Data Pipeline ====================

@dataclass
class DataIngestionConfig:
    """Config for downloading RAW data from S3 (only raw data)"""
    s3_bucket: str
    s3_region: str
    
    # S3 keys - ONLY for raw data
    corpus_s3_key: str
    train_s3_key: str
    
    # Local directories
    raw_data_dir: Path
    processed_data_dir: Path
    use_data_dir: Path


@dataclass
class DataPreprocessConfig:
    """Config for data preprocessing"""
    raw_data_dir: Path
    processed_data_dir: Path
    use_data_dir: Path
    
    # Preprocessing params
    remove_duplicates: bool
    min_text_length: int
    max_text_length: int
    
    # Split ratios
    train_size: float
    val_size: float
    test_size: float
    random_state: int
    
    # Tokenization
    n_jobs: int
    batch_size: int


# ==================== Bi-Encoder ====================

@dataclass
class BiEncoderTrainingConfig:
    # Model
    model_name: str
    pooling: str
    normalize: bool
    
    # Paths
    output_dir: Path
    logging_dir: Path
    train_data_path: Path
    val_data_path: Path
    
    # Training params
    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_seq_length: int
    
    # Loss & optimization
    loss_type: str
    loss_params: Dict
    use_fp16: bool
    
    # Checkpointing - keep all
    save_strategy: str
    save_steps: int
    save_total_limit: int  # -1 = keep all
    cleanup_checkpoints: bool
    
    # Evaluation
    eval_strategy: str
    eval_steps: int
    metric_for_best_model: str
    load_best_model_at_end: bool
    
    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    register_model: bool = True
    model_registry_name: str = "bi-encoder-legal-retrieval"
    
    # S3 - upload all
    s3_enabled: bool = False
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    s3_region: Optional[str] = None
    upload_all: bool = True  # Upload everything
    upload_strategy: str = "incremental"  # or "final"


@dataclass
class CorpusEmbeddingsConfig:
    """Config for building corpus embeddings - loads model from MLflow"""
    # MLflow model reference
    mlflow_tracking_uri: str
    mlflow_model_name: str
    mlflow_model_stage: str  # "Production", "Staging", or version number
    mlflow_run_id: Optional[str]  # Alternative to stage
    
    # Data
    corpus_path: Path
    output_dir: Path
    embeddings_file: str
    ids_file: str
    
    # Processing
    batch_size: int
    max_seq_length: int
    device: str
    show_progress: bool = True


@dataclass
class BiEncoderEvalConfig:
    """Config for bi-encoder evaluation"""
    # MLflow model reference (thay vì local model_path)
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
    mlflow_model_stage: str  # "Production", "Staging", "latest", or version number
    
    # Data
    test_path: Path
    corpus_path: Path
    corpus_embeddings_path: Path
    corpus_ids_path: Path
    
    # Metrics
    recall_k_values: list
    precision_k_values: list
    ndcg_k_values: list
    compute_mrr: bool
    compute_map: bool
    compute_hit_rate: bool
    
    # Processing
    batch_size: int
    max_seq_length: int
    device: str
    
    # Output
    results_dir: Path
    save_predictions: bool
    save_failed_cases: bool
    save_rankings: bool
    metrics_file: str
    predictions_file: str
    failed_cases_file: str

    mlflow_run_id: Optional[str] = None  # Alternative to stage

@dataclass
class BiEncoderServeConfig:
    """Config for bi-encoder serving"""
    # Model S3 configuration
    s3_bucket: str
    model_id: str  # Specific model ID or "latest"
    model_type: str  # "bi-encoder"
    local_path: Path
    
    # Server
    host: str
    port: int
    workers: int
    reload: bool
    
    # Inference
    batch_size: int
    max_seq_length: int
    device: str
    
    # Cache (optional)
    cache_enable: bool = False
    cache_ttl: int = 3600
    cache_max_size: int = 10000


# ==================== Hard Negative Mining ====================

@dataclass
class HardNegativeMiningConfig:
    """Config for hard negative mining"""
    # MLflow model reference
    mlflow_tracking_uri: str
    mlflow_model_name: str
    mlflow_model_stage: str
    
    # Input/Output paths
    input_path: Path
    output_path: Path
    
    # Corpus data
    embeddings_path: Path
    corpus_data_path: Path
    
    # Mining params
    top_k_candidates: int
    num_negatives_per_query: int
    min_score_threshold: float
    max_score_threshold: float
    batch_size: int
    max_seq_length: int
    device: str
    show_progress: bool
    
    # Optional
    mlflow_run_id: Optional[str] = None

# ==================== Cross-Encoder ====================

@dataclass
class CrossEncoderTrainingConfig:
    """Config for Cross-Encoder training"""
    # Model
    model_name: str
    num_labels: int
    
    # Data paths
    train_data_path: Path
    val_data_path: Path
    corpus_path: Path
    
    # Training params
    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_seq_length: int
    loss_type: str
    use_fp16: bool
    
    # Output
    output_dir: Path
    logging_dir: Path
    
    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    
    # S3 backup
    s3_enabled: bool = True
    s3_bucket: str = ""
    s3_prefix: str = "models/cross-encoder"
    s3_region: str = "ap-southeast-1"  # ✅ Thêm region
    upload_strategy: str = "final"
    
    # Model Registry (add these)
    register_model: bool = False
    model_registry_name: str = "cross-encoder-legal-retrieval"
    
    # Optional
    max_samples: Optional[int] = None


@dataclass
class CrossEncoderEvalConfig:
    """Config for cross-encoder evaluation"""
    # MLflow model reference
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
    mlflow_model_stage: str
    
    # Data
    test_path: Path
    corpus_path: Path
    
    # Processing
    batch_size: int
    max_seq_length: int
    device: str
    
    # Metrics
    threshold: float
    
    # Output
    results_dir: Path
    save_predictions: bool
    save_failed_cases: bool
    
    # Optional
    mlflow_run_id: Optional[str] = None
    candidates_path: Optional[Path] = None
    top_k_candidates: int = 100


@dataclass
class CrossEncoderServeConfig:
    """Config for cross-encoder serving"""
    # MLflow model reference
    mlflow_tracking_uri: str
    mlflow_model_name: str
    mlflow_model_stage: str
    
    # Corpus (optional)
    corpus_path: Path
    
    # Server
    host: str
    port: int
    workers: int
    reload: bool
    
    # Inference
    batch_size: int
    max_seq_length: int
    device: str
    top_n: int
    
    # Optional
    mlflow_run_id: Optional[str] = None


# ==================== Monitoring ====================

@dataclass
class MonitoringConfig:
    """Config for model monitoring"""
    prometheus_port: int
    metrics_path: str
    grafana_url: str
    dashboard_refresh_interval: int
    alert_rules_path: Path
    enable_slack_alerts: bool
    slack_webhook_url: Optional[str]
    enable_drift_detection: bool
    drift_check_interval: int
    embedding_drift_threshold: float
    performance_drop_threshold: float