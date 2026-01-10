import torch
from pathlib import Path
from typing import Tuple, Dict
from transformers import AutoTokenizer
import mlflow
import json

from src.common.configuration import BiEncoderServeConfig
from src.models.bi_encoder.model import BiEncoder
from src.common.logging_config import get_logger
from src.common.s3_utils import S3Client

logger = get_logger(__name__)


class BiEncoderModelLoader:
    """Load bi-encoder model and corpus embeddings"""

    def __init__(self, config: BiEncoderServeConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

    def load_model(self) -> Tuple[BiEncoder, AutoTokenizer]:
        """Load bi-encoder model from MLflow"""
        logger.info("="*80)
        logger.info("LOADING BI-ENCODER MODEL FROM MLFLOW")
        logger.info("="*80)
        logger.info(f"Model name: {self.config.mlflow_model_name}")
        logger.info(f"Model stage: {self.config.mlflow_model_stage}")
        
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        # Build model URI
        if self.config.mlflow_run_id:
            model_uri = f"runs:/{self.config.mlflow_run_id}/best_model"
            logger.info(f"Loading from run ID: {self.config.mlflow_run_id}")
        elif self.config.mlflow_model_stage.lower() == "latest":
            # Get latest version from registry
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                f"name='{self.config.mlflow_model_name}'"
            )
            
            if not versions:
                raise ValueError(f"No versions found for model: {self.config.mlflow_model_name}")
            
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            logger.info(f"Latest version found: {latest_version.version}")
            
            model_uri = f"models:/{self.config.mlflow_model_name}/{latest_version.version}"
            logger.info(f"Loading from registry: {model_uri}")
        else:
            model_uri = f"models:/{self.config.mlflow_model_name}/{self.config.mlflow_model_stage}"
            logger.info(f"Loading from registry: {model_uri}")
        
        # Download model artifacts from MLflow
        logger.info("Downloading model artifacts...")
        model_path = Path(mlflow.artifacts.download_artifacts(model_uri))
        
        # Try different possible locations for checkpoint
        possible_paths = [
            model_path / "pytorch_model.bin",
            model_path / "best_model.pt",
            model_path / "model.pt",
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                logger.info(f"Found checkpoint at: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No model checkpoint found in {model_path}. "
                f"Tried: {[p.name for p in possible_paths]}"
            )
        
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_config = checkpoint.get('config')
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_config = checkpoint.get('config')
            state_dict = checkpoint['state_dict']
        else:
            # Direct state dict
            state_dict = checkpoint
            model_config = None
        
        if model_config is None:
            # Try to load from model_config.json
            config_path = model_path / "model_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    model_config_dict = json.load(f)
                
                class SimpleConfig:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)
                
                model_config = SimpleConfig(model_config_dict)
            else:
                raise ValueError("Model config not found")
        
        # Initialize model
        model = BiEncoder(
            model_name=model_config.model_name,
            pooling=getattr(model_config, 'pooling', 'mean'),
            normalize=getattr(model_config, 'normalize', True)
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model.encoder.config._name_or_path
        )
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Model: {model_config.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Pooling: {model.pooling}")
        logger.info(f"  Normalize: {model.normalize}")
        
        return model, tokenizer
    
    def load_corpus_embeddings(self) -> Dict:
        """Load pre-computed corpus embeddings"""
        logger.info("="*80)
        logger.info("LOADING CORPUS EMBEDDINGS")
        logger.info("="*80)

        # Download from S3 if needed
        if self.config.corpus_use_s3:
            logger.info("Downloading corpus embeddings from S3...")
            self._download_corpus_from_s3()

        embeddings_path = self.config.corpus_embeddings_path

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Corpus embeddings not found: {embeddings_path}")

        # Load embeddings
        logger.info(f"Loading embeddings from: {embeddings_path}")
        data = torch.load(
            embeddings_path,
            map_location=self.device,
            weights_only=False
        )

        # Move embeddings to device
        data['embeddings'] = data['embeddings'].to(self.device)

        logger.info("✓ Corpus embeddings loaded successfully")
        logger.info(f"  Documents: {len(data['cids']):,}")
        logger.info(f"  Embedding dimension: {data['embeddings'].shape[1]}")
        logger.info(f"  Memory: {data['embeddings'].element_size() * data['embeddings'].nelement() / 1024 / 1024:.2f} MB")
        
        return data
    
    def _download_corpus_from_s3(self):
        """Download corpus embeddings from S3"""
        s3_client = S3Client()
        
        local_path = self.config.corpus_embeddings_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = s3_client.download_file(
            s3_bucket=self.config.corpus_s3_bucket,
            s3_key=self.config.corpus_s3_key,
            local_path=local_path,
            show_progress=True
        )
        
        if not success:
            raise RuntimeError("Failed to download corpus embeddings from S3")
        
        logger.info(f"✓ Corpus embeddings downloaded to: {local_path}")