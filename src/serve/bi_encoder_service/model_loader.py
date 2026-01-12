# import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
import boto3
from botocore.exceptions import ClientError

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
        """Download model from S3 using boto3"""
        if self.model_path.exists() and (self.model_path / "pytorch_model.bin").exists():
            logger.info(f"✓ Model already exists at {self.model_path}")
            return
        
        logger.info("📥 Downloading bi-encoder model from S3...")
        
        s3_bucket = self.config.s3_bucket
        model_id = self.config.model_id
        
        # Determine model ID
        if model_id == "latest":
            model_id = self._get_latest_model_id(s3_bucket, "bi-encoder")
            logger.info(f"Using latest bi-encoder model: {model_id}")
        
        s3_prefix = f"models/bi-encoder/{model_id}/best_model"
        
        logger.info(f"Downloading from: s3://{s3_bucket}/{s3_prefix}/")
        
        # Create directory
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Download using boto3
        try:
            s3_client = boto3.client('s3')
            
            # List all objects in prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
            
            downloaded_files = 0
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Skip if it's just the directory marker
                    if s3_key.endswith('/'):
                        continue
                    
                    # Get relative path (remove prefix)
                    relative_path = s3_key.replace(f"{s3_prefix}/", "")
                    local_file = self.model_path / relative_path
                    
                    # Create parent directories
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    logger.info(f"Downloading: {relative_path}")
                    s3_client.download_file(s3_bucket, s3_key, str(local_file))
                    downloaded_files += 1
            
            if downloaded_files == 0:
                raise RuntimeError(f"No files found at s3://{s3_bucket}/{s3_prefix}/")
            
            logger.info(f"✓ Downloaded {downloaded_files} files from S3")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise RuntimeError(f"Model not found at s3://{s3_bucket}/{s3_prefix}/")
            elif error_code == '403' or error_code == 'AccessDenied':
                raise RuntimeError("Access denied to S3. Check AWS credentials and IAM permissions.")
            else:
                raise RuntimeError(f"S3 download failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from S3: {e}")
    
    def _get_latest_model_id(self, s3_bucket: str, model_type: str) -> str:
        """Get latest model ID from S3"""
        try:
            s3_client = boto3.client('s3')
            
            prefix = f"models/{model_type}/"
            
            # List all model directories
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=prefix, Delimiter='/')
            
            model_ids = set()
            for page in pages:
                if 'CommonPrefixes' in page:
                    for obj in page['CommonPrefixes']:
                        # Extract model ID from prefix like "models/bi-encoder/xxx/"
                        model_id = obj['Prefix'].replace(prefix, '').rstrip('/')
                        if model_id:
                            model_ids.add(model_id)
            
            if not model_ids:
                raise RuntimeError(f"No models found in s3://{s3_bucket}/{prefix}")
            
            # Return the last one when sorted
            latest_id = sorted(model_ids)[-1]
            return latest_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to get latest model ID: {e}")
        
    def load_model(self):
        """Load model from local path (download from S3 if needed)"""
        logger.info("="*80)
        logger.info("LOADING BI-ENCODER MODEL")
        logger.info("="*80)
        
        # Download from S3 if configured
        if self.config.s3_bucket:
            self.download_from_s3()
        
        logger.info(f"Model path: {self.model_path}")
        
        # Verify model exists
        if not (self.model_path / "pytorch_model.bin").exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}/pytorch_model.bin\n"
                f"Please ensure model is uploaded to S3 or exists locally"
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
        device = torch.device(
            self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
        )
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded on device: {device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        logger.info("✓ Bi-encoder model loaded successfully")
        logger.info("="*80)
        
        return model, tokenizer