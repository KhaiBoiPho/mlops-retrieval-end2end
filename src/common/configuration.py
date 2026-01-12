from pathlib import Path
from typing import Dict, Any
from src.common.constants import (
    DATA_CONFIG_FILE_PATH,
    BI_ENCODER_TRAIN_CONFIG_PATH,
    BI_ENCODER_EVAL_CONFIG_PATH,
    BI_ENCODER_SERVE_CONFIG_PATH,
    CROSS_ENCODER_TRAIN_CONFIG_PATH,
    CROSS_ENCODER_EVAL_CONFIG_PATH,
    CROSS_ENCODER_SERVE_CONFIG_PATH,
    CORPUS_EMBEDDINGS_CONFIG_PATH,
    HARD_NEGATIVE_MINING_CONFIG_PATH,
    MONITORING_CONFIG_PATH
)
from src.common.utils import (
    read_yaml, 
    create_directories, 
    convert_to_dict, 
    safe_float, 
    safe_int, 
    safe_bool
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessConfig,
    BiEncoderTrainingConfig,
    CorpusEmbeddingsConfig,
    CrossEncoderTrainingConfig,
    BiEncoderEvalConfig,
    CrossEncoderEvalConfig,
    BiEncoderServeConfig,
    CrossEncoderServeConfig,
    HardNegativeMiningConfig,
    MonitoringConfig
)


class ConfigurationManager:
    """
    Centralized configuration manager for DVC-based pipeline.
    No more manual S3 upload configs - DVC handles everything.
    """
    
    def __init__(
        self,
        data_config_path: Path = DATA_CONFIG_FILE_PATH,
        bi_encoder_train_config_path: Path = BI_ENCODER_TRAIN_CONFIG_PATH,
        corpus_embeddings_config_path: Path = CORPUS_EMBEDDINGS_CONFIG_PATH,
        bi_encoder_eval_config_path: Path = BI_ENCODER_EVAL_CONFIG_PATH,
        bi_encoder_serve_config_path: Path = BI_ENCODER_SERVE_CONFIG_PATH,
        cross_encoder_train_config_path: Path = CROSS_ENCODER_TRAIN_CONFIG_PATH,
        cross_encoder_eval_config_path: Path = CROSS_ENCODER_EVAL_CONFIG_PATH,
        cross_encoder_serve_config_path: Path = CROSS_ENCODER_SERVE_CONFIG_PATH,
        hard_negative_config_path: Path = HARD_NEGATIVE_MINING_CONFIG_PATH,
        monitoring_config_path: Path = MONITORING_CONFIG_PATH,
    ):
        # Load all configs
        self.data_config = read_yaml(data_config_path)
        self.bi_encoder_train_config = read_yaml(bi_encoder_train_config_path)
        self.corpus_embeddings_config = read_yaml(corpus_embeddings_config_path)
        self.bi_encoder_eval_config = read_yaml(bi_encoder_eval_config_path)
        self.bi_encoder_serve_config = read_yaml(bi_encoder_serve_config_path)
        self.cross_encoder_train_config = read_yaml(cross_encoder_train_config_path)
        self.cross_encoder_eval_config = read_yaml(cross_encoder_eval_config_path)
        self.cross_encoder_serve_config = read_yaml(cross_encoder_serve_config_path)
        self.hard_negative_config = read_yaml(hard_negative_config_path)
        self.monitoring_config = read_yaml(monitoring_config_path)
        
        # Create root directories (DVC-managed paths)
        create_directories([
            Path("data/raw"),
            Path("data/processed"),
            Path("data/use"),
            Path("artifacts"),
            Path("reports"),
            Path("logs")
        ])

    # Data pipeline configs
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Config for downloading RAW data from S3 (only once)"""
        s3 = self.data_config.s3
        local = self.data_config.local
        
        create_directories([
            Path(local.raw_data_dir),
            Path(local.processed_data_dir),
            Path(local.use_data_dir)
        ])
        
        return DataIngestionConfig(
            s3_bucket=s3.bucket,
            s3_region=s3.region,
            
            # S3 keys - ONLY for raw data
            corpus_s3_key=s3.raw_data.corpus_key,
            train_s3_key=s3.raw_data.train_key,
            
            # Local directories
            raw_data_dir=Path(local.raw_data_dir),
            processed_data_dir=Path(local.processed_data_dir),
            use_data_dir=Path(local.use_data_dir)
        )
    
    def get_data_preprocess_config(self) -> DataPreprocessConfig:
        """Config for data preprocessing (cleaning, tokenization, splitting)"""
        local = self.data_config.local
        preprocessing = self.data_config.preprocessing
        split = self.data_config.split
        tokenization = self.data_config.tokenization
        
        return DataPreprocessConfig(
            raw_data_dir=Path(local.raw_data_dir),
            processed_data_dir=Path(local.processed_data_dir),
            use_data_dir=Path(local.use_data_dir),
            
            # Preprocessing
            remove_duplicates=safe_bool(preprocessing.get('remove_duplicates', True)),
            min_text_length=safe_int(preprocessing.get('min_text_length', 10)),
            max_text_length=safe_int(preprocessing.get('max_text_length', 5000)),
            
            # Split ratios
            train_size=safe_float(split.train_size),
            val_size=safe_float(split.val_size),
            test_size=safe_float(split.test_size),
            random_state=safe_int(split.random_state),
            
            # Tokenization
            n_jobs=safe_int(tokenization.get('n_jobs', -1)),
            batch_size=safe_int(tokenization.get('batch_size', 100))
        )

    # Bi-encoder configs
    def get_bi_encoder_training_config(self) -> BiEncoderTrainingConfig:
        """Load bi-encoder training configuration"""
        config = self.bi_encoder_train_config
        params = config.training_params
        
        create_directories([
            Path(config.output_dir),
            Path(config.logging_dir)
        ])
        
        loss_params = convert_to_dict(params.loss.params) if hasattr(params.loss.params, '__dict__') else params.loss.params
        
        # S3 config
        s3_config = config.get('s3', {})
        
        return BiEncoderTrainingConfig(
            # Model
            model_name=config.model.name,
            pooling=config.model.get('pooling', 'mean'),
            normalize=safe_bool(config.model.get('normalize', True)),
            
            # Paths
            output_dir=Path(config.output_dir),
            logging_dir=Path(config.logging_dir),
            train_data_path=Path(config.data.train_path),
            val_data_path=Path(config.data.val_path),
            
            # Training params
            num_epochs=safe_int(params.num_epochs),
            batch_size=safe_int(params.batch_size),
            learning_rate=safe_float(params.learning_rate),
            warmup_ratio=safe_float(params.warmup_ratio),
            weight_decay=safe_float(params.weight_decay),
            max_seq_length=safe_int(params.max_seq_length),
            
            # Loss
            loss_type=params.loss.type,
            loss_params=loss_params,
            use_fp16=safe_bool(params.use_fp16),
            
            # Checkpointing - keep all
            save_strategy=params.save_strategy,
            save_steps=safe_int(params.save_steps),
            save_total_limit=safe_int(params.get('save_total_limit', -1)),  # -1 = keep all
            cleanup_checkpoints=safe_bool(params.get('cleanup_checkpoints', False)),
            
            # Evaluation
            eval_strategy=params.eval_strategy,
            eval_steps=safe_int(params.eval_steps),
            metric_for_best_model=params.metric_for_best_model,
            load_best_model_at_end=safe_bool(params.load_best_model_at_end),
            
            # MLflow
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_experiment_name=config.mlflow.experiment_name,
            register_model=safe_bool(config.mlflow.get('register_model', True)),
            model_registry_name=config.mlflow.get('registry_name', 'bi-encoder-legal-retrieval'),
            
            # S3 - upload all
            s3_enabled=safe_bool(s3_config.get('enabled', False)),
            s3_bucket=s3_config.get('bucket'),
            s3_prefix=s3_config.get('prefix', 'models/bi-encoder'),
            s3_region=s3_config.get('region', 'ap-southeast-1'),
            upload_all=safe_bool(s3_config.get('upload_all', True)),
            upload_strategy=s3_config.get('upload_strategy', 'final')  # 'incremental' or 'final'
        )
    
    def get_corpus_embeddings_config(self) -> CorpusEmbeddingsConfig:
        """Config for building corpus embeddings - loads model from MLflow"""
        config = self.corpus_embeddings_config
        
        # Create output directory
        output_dir = Path(config.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return CorpusEmbeddingsConfig(
            # MLflow model reference
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_model_name=config.mlflow.model_name,
            mlflow_model_stage=config.mlflow.get('model_stage', 'Production'),
            mlflow_run_id=config.mlflow.get('run_id', None),
            
            # Data
            corpus_path=Path(config.data.corpus_path),
            output_dir=output_dir,
            embeddings_file=config.data.embeddings_file,
            ids_file=config.data.ids_file,
            
            # Processing
            batch_size=safe_int(config.processing.batch_size),
            max_seq_length=safe_int(config.processing.max_seq_length),
            device=str(config.processing.device),
            show_progress=safe_bool(config.processing.get('show_progress', True))
        )
    
    def get_bi_encoder_eval_config(self) -> BiEncoderEvalConfig:
        """Config for bi-encoder evaluation"""
        config = self.bi_encoder_eval_config
        
        # Create results directory
        results_dir = Path(config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle nested mlflow config (support both structures)
        if hasattr(config, 'mlflow') and hasattr(config.mlflow, 'tracking_uri'):
            # Structure: mlflow: { tracking_uri: ... }
            mlflow_tracking_uri = str(config.mlflow.tracking_uri)
            mlflow_experiment_name = str(config.mlflow.experiment_name)
            mlflow_model_stage = str(config.mlflow.model_stage)
            mlflow_run_id = config.mlflow.get('run_id', None)
        elif hasattr(config.model, 'mlflow'):
            # Structure: model: { mlflow: { tracking_uri: ... } }
            mlflow_tracking_uri = str(config.model.mlflow.tracking_uri)
            mlflow_experiment_name = str(config.model.mlflow.experiment_name)
            mlflow_model_stage = str(config.model.mlflow.model_stage)
            mlflow_run_id = config.model.mlflow.get('run_id', None)
        else:
            raise ValueError("MLflow config not found in expected locations")
        
        return BiEncoderEvalConfig(
            # MLflow model reference
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_model_name=str(config.model.mlflow_model_name),
            mlflow_model_stage=mlflow_model_stage,
            mlflow_run_id=mlflow_run_id,
            
            # Data
            test_path=Path(config.data.test_path),
            corpus_path=Path(config.data.corpus_path),
            corpus_embeddings_path=Path(config.data.corpus_embeddings_path),
            corpus_ids_path=Path(config.data.corpus_ids_path),
            
            # Metrics
            recall_k_values=list(config.metrics.recall_k_values),
            precision_k_values=list(config.metrics.precision_k_values),
            ndcg_k_values=list(config.metrics.ndcg_k_values),
            compute_mrr=safe_bool(config.metrics.compute_mrr),
            compute_map=safe_bool(config.metrics.compute_map),
            compute_hit_rate=safe_bool(config.metrics.compute_hit_rate),
            
            # Processing
            batch_size=safe_int(config.processing.batch_size),
            max_seq_length=safe_int(config.processing.max_seq_length),
            device=str(config.processing.device),
            
            # Output
            results_dir=results_dir,
            save_predictions=safe_bool(config.output.save_predictions),
            save_failed_cases=safe_bool(config.output.save_failed_cases),
            save_rankings=safe_bool(config.output.save_rankings),
            metrics_file=config.output.get('metrics_file', 'metrics.json'),
            predictions_file=config.output.get('predictions_file', 'predictions.csv'),
            failed_cases_file=config.output.get('failed_cases_file', 'failed_cases.csv')
        )
    
    def get_bi_encoder_serve_config(self) -> BiEncoderServeConfig:
            """Config for bi-encoder serving"""
            config = self.bi_encoder_serve_config
            
            # Get model configuration
            model_config = config.model
            s3_bucket = str(model_config.s3_bucket)
            model_id = str(model_config.model_id)
            model_type = str(model_config.model_type)
            local_path = Path(model_config.local_path)
            
            # Get server configuration
            server_config = config.server
            host = str(server_config.host)
            port = safe_int(server_config.port)
            workers = safe_int(server_config.workers)
            reload = safe_bool(server_config.reload)
            
            # Get inference configuration
            inference_config = config.inference
            batch_size = safe_int(inference_config.batch_size)
            max_seq_length = safe_int(inference_config.max_seq_length)
            device = str(inference_config.device)
            
            # Get cache configuration (optional)
            cache_config = config.get('cache', {})
            cache_enable = safe_bool(cache_config.get('enable', False))
            cache_ttl = safe_int(cache_config.get('ttl', 3600))
            cache_max_size = safe_int(cache_config.get('max_size', 10000))
            
            return BiEncoderServeConfig(
                # Model S3
                s3_bucket=s3_bucket,
                model_id=model_id,
                model_type=model_type,
                local_path=local_path,
                
                # Server
                host=host,
                port=port,
                workers=workers,
                reload=reload,
                
                # Inference
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                device=device,
                
                # Cache
                cache_enable=cache_enable,
                cache_ttl=cache_ttl,
                cache_max_size=cache_max_size
            )

    # Hard negative mining
    def get_hard_negative_mining_config(self, split: str = 'train') -> HardNegativeMiningConfig:
        """Config for hard negative mining
        
        Args:
            split: 'train', 'val', or 'test'
        """
        config = self.hard_negative_config
        
        # Create output directory
        output_dir = Path(config.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select appropriate paths based on split
        if split == 'train':
            input_path = Path(config.data.train_path)
            output_file = config.data.train_output_file
        elif split == 'val':
            input_path = Path(config.data.val_path)
            output_file = config.data.val_output_file
        elif split == 'test':
            input_path = Path(config.data.get('test_path', 'data/use/test_split.csv'))
            output_file = config.data.get('test_output_file', 'test_with_hard_negatives.csv')
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        return HardNegativeMiningConfig(
            # MLflow model reference
            mlflow_tracking_uri=str(config.mlflow.tracking_uri),
            mlflow_model_name=str(config.mlflow.model_name),
            mlflow_model_stage=str(config.mlflow.model_stage),
            mlflow_run_id=config.mlflow.get('run_id', None),
            
            # Data paths
            embeddings_path=Path(config.corpus.embeddings_path),
            corpus_data_path=Path(config.corpus.data_path),
            input_path=input_path,
            output_path=output_dir / output_file,  # ← THÊM DÒNG NÀY
            
            # Mining params
            top_k_candidates=safe_int(config.mining.top_k_candidates),
            num_negatives_per_query=safe_int(config.mining.num_negatives_per_query),
            min_score_threshold=safe_float(config.mining.min_score_threshold),
            max_score_threshold=safe_float(config.mining.get('max_score_threshold', 0.9)),
            batch_size=safe_int(config.mining.batch_size),
            max_seq_length=safe_int(config.mining.max_seq_length),
            device=str(config.mining.device),
            show_progress=safe_bool(config.mining.get('show_progress', True))
        )

    # Cross-encoder configs    
    def get_cross_encoder_training_config(self) -> CrossEncoderTrainingConfig:
        """Config for cross-encoder training"""
        config = self.cross_encoder_train_config
        
        # Create directories
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging_dir = Path(config.logging_dir)
        logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Safely get s3 and registry configs with defaults
        s3_config = config.get('s3', {})
        registry_config = config.get('registry', {})
        
        return CrossEncoderTrainingConfig(
            # Model
            model_name=str(config.model.name),
            num_labels=safe_int(config.model.num_labels),
            
            # Data
            train_data_path=Path(config.data.train_path),
            val_data_path=Path(config.data.val_path),
            corpus_path=Path(config.data.corpus_path),
            max_samples=config.data.get('max_samples', None),
            
            # Training
            num_epochs=safe_int(config.training_params.num_epochs),
            batch_size=safe_int(config.training_params.batch_size),
            learning_rate=safe_float(config.training_params.learning_rate),
            warmup_ratio=safe_float(config.training_params.warmup_ratio),
            weight_decay=safe_float(config.training_params.weight_decay),
            max_seq_length=safe_int(config.training_params.max_seq_length),
            loss_type=str(config.training_params.loss_type),
            use_fp16=safe_bool(config.training_params.use_fp16),
            
            # Output
            output_dir=output_dir,
            logging_dir=logging_dir,
            
            # MLflow
            mlflow_tracking_uri=str(config.mlflow.tracking_uri),
            mlflow_experiment_name=str(config.mlflow.experiment_name),
            
            # S3 (with safe defaults)
            s3_enabled=safe_bool(s3_config.get('enabled', False)),
            s3_bucket=str(s3_config.get('bucket', '')),
            s3_prefix=str(s3_config.get('prefix', 'models/cross-encoder')),
            s3_region=str(s3_config.get('region', 'ap-southeast-1')),
            upload_strategy=str(s3_config.get('upload_strategy', 'final')),
        
            
            # Registry (with safe defaults)
            register_model=safe_bool(registry_config.get('enabled', False)),
            model_registry_name=str(registry_config.get('model_name', 'cross-encoder-legal-retrieval'))
        )
    
    def get_cross_encoder_eval_config(self) -> CrossEncoderEvalConfig:
        """Config for cross-encoder evaluation"""
        config = self.cross_encoder_eval_config
        
        # Create results directory
        results_dir = Path(config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Safely get model config
        model_config = config.get('model', {})
        
        return CrossEncoderEvalConfig(
            # MLflow
            mlflow_tracking_uri=str(config.mlflow.tracking_uri),
            mlflow_experiment_name=str(config.mlflow.experiment_name),
            mlflow_model_name=str(model_config.get('mlflow_model_name', 'cross-encoder-legal-retrieval')),
            mlflow_model_stage=str(model_config.get('mlflow_model_stage', 'latest')),
            mlflow_run_id=model_config.get('mlflow_run_id', None),
            
            # Data
            test_path=Path(config.data.test_path),
            corpus_path=Path(config.data.corpus_path),
            candidates_path=Path(config.data.candidates_path) if config.data.get('candidates_path') else None,
            
            # Processing
            batch_size=safe_int(config.processing.batch_size),
            max_seq_length=safe_int(config.processing.max_seq_length),
            device=str(config.processing.device),
            top_k_candidates=safe_int(config.processing.get('top_k_candidates', 100)),
            
            # Metrics
            threshold=safe_float(config.metrics.get('threshold', 0.5)),
            
            # Output
            results_dir=results_dir,
            save_predictions=safe_bool(config.output.save_predictions),
            save_failed_cases=safe_bool(config.output.save_failed_cases)
        )

    def get_cross_encoder_serve_config(self) -> CrossEncoderServeConfig:
        """Config for cross-encoder serving"""
        config = self.cross_encoder_serve_config
        
        return CrossEncoderServeConfig(
            # MLflow
            mlflow_tracking_uri=str(config.model.mlflow_tracking_uri),
            mlflow_model_name=str(config.model.mlflow_model_name),
            mlflow_model_stage=str(config.model.mlflow_model_stage),
            mlflow_run_id=config.model.get('mlflow_run_id', None),
            
            # Corpus
            corpus_path=Path(config.corpus.path),
            
            # Server
            host=str(config.server.host),
            port=int(config.server.port),
            workers=int(config.server.workers),
            reload=bool(config.server.reload),
            
            # Inference
            batch_size=int(config.inference.batch_size),
            max_seq_length=int(config.inference.max_seq_length),
            device=str(config.inference.device),
            top_n=int(config.inference.top_n)
        )

    # ==================== MONITORING CONFIG ====================
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Config for model monitoring"""
        config = self.monitoring_config
        
        return MonitoringConfig(
            # Prometheus
            prometheus_port=safe_int(config.prometheus.port),
            metrics_path=config.prometheus.metrics_path,
            
            # Grafana
            grafana_url=config.grafana.url,
            dashboard_refresh_interval=safe_int(config.grafana.dashboard_refresh_interval),
            
            # Alerts
            alert_rules_path=Path(config.alerts.rules_path),
            enable_slack_alerts=safe_bool(config.alerts.slack.enable),
            slack_webhook_url=config.alerts.slack.get('webhook_url'),
            
            # Drift detection
            enable_drift_detection=safe_bool(config.drift_detection.enable),
            drift_check_interval=safe_int(config.drift_detection.check_interval),
            embedding_drift_threshold=safe_float(config.drift_detection.embedding_drift_threshold),
            performance_drop_threshold=safe_float(config.drift_detection.performance_drop_threshold)
        )
    
    # ==================== UTILITY METHODS ====================
    
    def get_all_params(self) -> Dict[str, Any]:
        """Get all parameters as dict for logging"""
        return {
            "data_config": dict(self.data_config),
            "bi_encoder_train": dict(self.bi_encoder_train_config),
            "cross_encoder_train": dict(self.cross_encoder_train_config),
            "hard_negative_mining": dict(self.hard_negative_config),
            "monitoring": dict(self.monitoring_config)
        }