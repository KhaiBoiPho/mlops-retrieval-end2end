import torch
from pathlib import Path
from transformers import AutoTokenizer
from typing import Tuple
import mlflow
import json

from src.models.cross_encoder.model import CrossEncoder
from src.common.logging_config import get_logger
from src.entity.config_entity import CrossEncoderServeConfig

logger = get_logger(__name__)


class CrossEncoderModelLoader:
    """Load cross-encoder model for reranking from MLflow"""

    def __init__(self, config: CrossEncoderServeConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

    def load_model(self) -> Tuple[CrossEncoder, AutoTokenizer]:
        """Load cross-encoder model from MLflow"""
        logger.info("="*80)
        logger.info("LOADING CROSS-ENCODER MODEL FROM MLFLOW")
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
        model = CrossEncoder(
            model_name=model_config.model_name,
            num_labels=getattr(model_config, 'num_labels', 1)
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
        logger.info(f"  Num labels: {model.num_labels}")
        
        return model, tokenizer