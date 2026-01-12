import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.models.cross_encoder.model import CrossEncoder
from src.common.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderModelLoader:
    """Load cross-encoder from S3 or local path"""
    
    def __init__(self, config=None):
        self.config = config
        self.model_path = Path(os.getenv("MODEL_PATH", "/app/models/cross-encoder"))
        
    def download_from_s3(self):
        """Download model from S3 if not exists locally"""
        if self.model_path.exists() and (self.model_path / "pytorch_model.bin").exists():
            logger.info(f"✓ Model already exists at {self.model_path}")
            return
        
        logger.info("📥 Downloading model from S3...")
        
        import subprocess
        
        s3_bucket = os.getenv("S3_BUCKET")
        s3_model_path = os.getenv("S3_MODEL_PATH", "models/cross-encoder/latest")
        
        if not s3_bucket:
            raise ValueError("S3_BUCKET environment variable not set")
        
        # Create directory
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Download with aws cli
        cmd = [
            "aws", "s3", "sync",
            f"s3://{s3_bucket}/{s3_model_path}/best_model",
            str(self.model_path),
            "--quiet"
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"S3 download failed: {result.stderr}")
            raise RuntimeError(f"Failed to download model from S3: {result.stderr}")
        
        logger.info("✓ Model downloaded from S3")
        
    def load_model(self):
        """Load model from local path (download from S3 if needed)"""
        logger.info("="*80)
        logger.info("LOADING CROSS-ENCODER MODEL")
        logger.info("="*80)
        
        # Download from S3 if S3_BUCKET is set
        if os.getenv("S3_BUCKET"):
            self.download_from_s3()
        
        logger.info(f"Model path: {self.model_path}")
        
        # Verify model exists
        if not (self.model_path / "pytorch_model.bin").exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}/pytorch_model.bin"
            )
        
        # Load config
        config_path = self.model_path / "model_config.json"
        with open(config_path) as f:
            model_config = json.load(f)
        
        # Initialize model
        model = CrossEncoder(
            model_name=model_config["model_name"],
            num_labels=model_config.get("num_labels", 1)
        )
        
        # Load weights
        logger.info("Loading model weights...")
        state_dict = torch.load(
            self.model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=False
        )
        model.load_state_dict(state_dict)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        logger.info("✓ Model loaded successfully")
        logger.info("="*80)
        
        return model, tokenizer
