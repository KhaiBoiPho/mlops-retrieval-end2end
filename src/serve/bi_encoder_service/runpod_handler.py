"""
RunPod Serverless Handler for Bi-Encoder
Handles: /embed and /batch_embed endpoints
"""
import os
import runpod
# import torch
import time
from typing import Dict, Any

from src.serve.bi_encoder_service.model_loader import BiEncoderModelLoader
from src.serve.bi_encoder_service.embedding import BiEncoderEmbedder
from src.common.configuration import ConfigurationManager
from src.common.logging_config import get_logger
from src.monitoring.metrics_exporter import MetricsExporter

logger = get_logger(__name__)

# Global state
embedder: BiEncoderEmbedder = None
config = None

# Initialize metrics exporter
metrics_exporter = MetricsExporter(endpoint_name='bi-encoder')


def load_model():
    """Load model on cold start"""
    global embedder, config

    # Track cold start
    metrics_exporter.track_cold_start()

    logger.info("="*80)
    logger.info("INITIALIZING BI-ENCODER ON RUNPOD SERVERLESS")
    logger.info("="*80)

    try:
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.get_bi_encoder_serve_config()

        if os.getenv("S3_BUCKET"):
            config.s3_bucket = os.getenv("S3_BUCKET")
            logger.info(f"Override S3_BUCKET from env: {config.s3_bucket}")

        if os.getenv("BI_ENCODER_MODEL_ID"):
            config.model_id = os.getenv("BI_ENCODER_MODEL_ID")
            logger.info(f"Override BI_ENCODER_MODEL_ID from env: {config.model_id}")

        if os.getenv("DEVICE"):
            config.device = os.getenv("DEVICE")
            logger.info(f"Override DEVICE from env: {config.device}")

        logger.info("Configuration:")
        logger.info(f"  S3 Bucket: {config.s3_bucket}")
        logger.info(f"  Model ID: {config.model_id}")
        logger.info(f"  Device: {config.device}")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  Max Seq Length: {config.max_seq_length}")

        # Load model from S3
        loader = BiEncoderModelLoader(config=config)
        model, tokenizer = loader.load_model()

        # Initialize embedder
        embedder = BiEncoderEmbedder(
            model=model,
            tokenizer=tokenizer,
            config=config
        )

        logger.info("✓ BI-ENCODER READY ON RUNPOD")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    
    Input format:
    {
        "input": {
            "action": "embed" | "batch_embed",
            "text": str,           # for embed
            "texts": List[str]     # for batch_embed
        }
    }
    
    Output format:
    {
        "embedding": List[float],      # for embed
        "embeddings": List[List[float]], # for batch_embed
        "dimension": int,
        "encode_time_ms": float,
        "count": int  # for batch_embed
    }
    """

    start_time = time.time()
    action = event.get("input", {}).get("action", "embed")
    status = "success"
    
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "embed")
        
        if action == "embed":
            text = input_data.get("text")
            if not text:
                status = "error"
                return {"error": "Missing 'text' parameter"}
            
            # Track inference time
            inference_start = time.time()
            embedding, encode_time = embedder.encode_single(text)
            inference_time = time.time() - inference_start
            
            # Track metrics
            metrics_exporter.track_inference(action, inference_time)
            metrics_exporter.track_embedding(len(embedding))
            metrics_exporter.track_batch_size(1)
            
            return {
                "embedding": embedding.tolist(),
                "dimension": len(embedding),
                "encode_time_ms": encode_time
            }
        
        elif action == "batch_embed":
            texts = input_data.get("texts")
            if not texts or not isinstance(texts, list):
                status = "error"
                return {"error": "Missing or invalid 'texts' parameter"}
            
            # Track inference time
            inference_start = time.time()
            embeddings, encode_time = embedder.encode_batch(texts)
            inference_time = time.time() - inference_start
            
            # Track metrics
            metrics_exporter.track_inference(action, inference_time)
            metrics_exporter.track_embedding(len(embeddings[0]) if embeddings else 0)
            metrics_exporter.track_batch_size(len(texts))
            
            return {
                "embeddings": [emb.tolist() for emb in embeddings],
                "count": len(embeddings),
                "dimension": len(embeddings[0]) if embeddings else 0,
                "encode_time_ms": encode_time
            }
        
        else:
            status = "error"
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        status = "error"
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}
    
    finally:
        # Track request
        duration = time.time() - start_time
        metrics_exporter.track_request(action, status, duration)

if __name__ == "__main__":
    # Load model on startup
    load_model()

    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})