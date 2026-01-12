import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.models.bi_encoder.model import BiEncoder
from src.common.logging_config import get_logger
from src.entity.config_entity import BiEncoderServeConfig

logger = get_logger(__name__)


class BiEncoderModelLoader:
    """Load bi-encoder from S3 or local path"""
    
    def __init__(self, config: BiEncoderServeConfig):
        self.config = config
        self.model_path = Path(config.local_path)
        
    def download_from_s3(self):
        """Download model from S3 if not exists locally"""
        if self.model_path.exists() and (self.model_path / "pytorch_model.bin").exists():
            logger.info(f"✓ Model already exists at {self.model_path}")
            return
        
        logger.info("📥 Downloading bi-encoder model from S3...")
        
        import subprocess
        
        s3_bucket = self.config.s3_bucket
        model_id = self.config.model_id
        
        # Determine model ID
        if model_id == "latest":
            # Get latest model ID from S3
            model_id = self._get_latest_model_id(s3_bucket, "bi-encoder")
            logger.info(f"Using latest bi-encoder model: {model_id}")
        
        s3_prefix = f"models/bi-encoder/{model_id}"
        
        # Create directory
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Download best_model folder
        cmd = [
            "aws", "s3", "sync",
            f"s3://{s3_bucket}/{s3_prefix}/best_model/",
            str(self.model_path),
            "--quiet"
        ]
        
        logger.info(f"Downloading from: s3://{s3_bucket}/{s3_prefix}/best_model/")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"S3 download failed: {result.stderr}")
            raise RuntimeError(f"Failed to download model from S3: {result.stderr}")
        
        logger.info("✓ Bi-encoder model downloaded from S3")
    
    def _get_latest_model_id(self, s3_bucket: str, model_type: str) -> str:
        """Get latest model ID from S3 by sorting folders by modification time"""
        import subprocess
        
        cmd = [
            "aws", "s3", "ls",
            f"s3://{s3_bucket}/models/{model_type}/",
            "--recursive"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to list S3 models: {result.stderr}")
        
        # Parse output and find latest
        lines = result.stdout.strip().split('\n')
        if not lines:
            raise RuntimeError(f"No models found in s3://{s3_bucket}/models/{model_type}/")
        
        # Extract model IDs from paths like "models/bi-encoder/0f0eae5cd35744caa55b4f784ec5559e/"
        model_ids = set()
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]  # The S3 key
                if f"models/{model_type}/" in path:
                    model_id = path.split(f"models/{model_type}/")[1].split('/')[0]
                    if model_id:
                        model_ids.add(model_id)
        
        if not model_ids:
            raise RuntimeError("No valid model IDs found")
        
        # Return the last one when sorted (alphabetically)
        # You might want to implement better logic here (e.g., by timestamp)
        latest_id = sorted(model_ids)[-1]
        return latest_id
        
    def load_model(self):
        """Load model from local path (download from S3 if needed)"""
        logger.info("="*80)
        logger.info("LOADING BI-ENCODER MODEL")
        logger.info("="*80)
        
        # Download from S3 if S3 bucket is configured
        if self.config.s3_bucket:
            self.download_from_s3()
        
        logger.info(f"Model path: {self.model_path}")
        
        # Verify model exists
        if not (self.model_path / "pytorch_model.bin").exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}/pytorch_model.bin\n"
                f"Please ensure:\n"
                f"1. Model is uploaded to S3 at s3://{self.config.s3_bucket}/models/bi-encoder/{self.config.model_id}/best_model/, or\n"
                f"2. Model exists at local path {self.model_path}"
            )
        
        # Load config
        config_path = self.model_path / "model_config.json"
        with open(config_path) as f:
            model_config = json.load(f)
        
        logger.info(f"Model: {model_config['model_name']}")
        logger.info(f"Pooling: {model_config.get('pooling', 'mean')}")
        logger.info(f"Normalize: {model_config.get('normalize', True)}")
        
        # Initialize model
        model = BiEncoder(
            model_name=model_config["model_name"],
            pooling=model_config.get("pooling", "mean"),
            normalize=model_config.get("normalize", True)
        )
        
        # Load weights
        logger.info("Loading model weights...")
        state_dict = torch.load(
            self.model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=False
        )
        model.load_state_dict(state_dict)
        
        # Move to device
        device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded on device: {device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        logger.info("✓ Bi-encoder model loaded successfully")
        logger.info("="*80)
        
        return model, tokenizer