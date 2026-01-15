"""
RunPod Serverless Handler for Cross-Encoder
"""
import runpod
# import torch
# import time
import os
import time
from typing import Dict, Any
# from pathlib import Path

from src.serve.cross_encoder_service.model_loader import CrossEncoderModelLoader
from src.serve.cross_encoder_service.reranker import CrossEncoderReranker
from src.common.configuration import ConfigurationManager
# from src.entity.config_entity import CrossEncoderServeConfig
from src.common.logging_config import get_logger
from src.monitoring.metrics_exporter import MetricsExporter

logger = get_logger(__name__)

# Global state
reranker: CrossEncoderReranker = None
config = None

# Initialize metrics exporter
metrics_exporter = MetricsExporter(endpoint_name='cross-encoder')


def load_model():
    """Load model on cold start"""
    global reranker, config, metrics_exporter

    # Track cold start
    metrics_exporter.track_cold_start()
    
    logger.info("="*80)
    logger.info("INITIALIZING CROSS-ENCODER ON RUNPOD SERVERLESS")
    logger.info("="*80)
    
    try:
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.get_cross_encoder_serve_config()
        
        # Override with env vars if provided
        if os.getenv("S3_BUCKET"):
            config.s3_bucket = os.getenv("S3_BUCKET")
            logger.info(f"Override S3_BUCKET: {config.s3_bucket}")
        
        if os.getenv("MODEL_ID"):
            config.model_id = os.getenv("MODEL_ID")
            logger.info(f"Override MODEL_ID: {config.model_id}")
        
        if os.getenv("DEVICE"):
            config.device = os.getenv("DEVICE")
            logger.info(f"Override DEVICE: {config.device}")
        
        logger.info("Configuration:")
        logger.info(f"  S3 Bucket: {config.s3_bucket}")
        logger.info(f"  Model ID: {config.model_id}")
        logger.info(f"  Device: {config.device}")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  Max Seq Length: {config.max_seq_length}")
        
        # Load model
        loader = CrossEncoderModelLoader(config=config)
        model, tokenizer = loader.load_model()
        
        # Initialize reranker
        reranker = CrossEncoderReranker(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        logger.info("✓ CROSS-ENCODER READY ON RUNPOD")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler
    
    Input:
    {
        "input": {
            "action": "rerank" | "score_pair" | "batch_score",
            "query": str,
            "documents": [{"id": "1", "text": "..."}, ...],  # for rerank
            "document": str,  # for score_pair
            "top_n": int  # optional
        }
    }
    """

    """RunPod handler with metrics"""
    start_time = time.time()
    action = event.get("input", {}).get("action", "rerank")
    status = "success"
    
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "rerank")
        query = input_data.get("query")
        
        if not query:
            status = "error"
            return {"error": "Missing 'query' parameter"}
        
        if action == "rerank":
            documents = input_data.get("documents")
            if not documents:
                status = "error"
                return {"error": "Missing 'documents' parameter"}
            
            top_n = input_data.get("top_n", len(documents))
            
            # Prepare documents
            docs = [
                {
                    "id": doc.get("id", str(i)),
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata")
                }
                for i, doc in enumerate(documents)
            ]
            
            # Track inference time
            inference_start = time.time()
            reranked_results, rerank_time = reranker.rerank(
                query=query,
                documents=docs,
                top_n=top_n
            )
            inference_time = time.time() - inference_start
            
            # Extract scores
            scores = [r['score'] for r in reranked_results]
            
            # Track metrics
            metrics_exporter.track_inference(action, inference_time)
            metrics_exporter.track_scores(scores)
            metrics_exporter.track_batch_size(len(documents))
            
            return {
                "results": reranked_results,
                "total": len(reranked_results),
                "rerank_time_ms": rerank_time
            }
        
        elif action == "score_pair":
            document = input_data.get("document")
            if not document:
                status = "error"
                return {"error": "Missing 'document' parameter"}
            
            inference_start = time.time()
            score, score_time = reranker.score_pair(query=query, document=document)
            inference_time = time.time() - inference_start
            
            # Track metrics
            metrics_exporter.track_inference(action, inference_time)
            metrics_exporter.track_scores([score])
            
            return {
                "score": score,
                "score_time_ms": score_time
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
    load_model()
    runpod.serverless.start({"handler": handler})