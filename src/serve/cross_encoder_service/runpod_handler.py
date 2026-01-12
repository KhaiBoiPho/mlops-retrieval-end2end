"""
RunPod Serverless Handler for Cross-Encoder
"""
import runpod
# import torch
# import time
import os
from typing import Dict, Any
from pathlib import Path

from src.serve.cross_encoder_service.model_loader import CrossEncoderModelLoader
from src.serve.cross_encoder_service.reranker import CrossEncoderReranker
from src.common.configuration import ConfigurationManager
from src.entity.config_entity import CrossEncoderServeConfig
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Global state
reranker: CrossEncoderReranker = None
config = None


def load_model():
    """Load model on cold start"""
    global reranker, config
    
    logger.info("="*80)
    logger.info("INITIALIZING CROSS-ENCODER ON RUNPOD SERVERLESS")
    logger.info("="*80)
    
    try:
        # Load configuration
        config_manager = ConfigurationManager()
        raw_config = config_manager.cross_encoder_serve_config
        
        # Parse to CrossEncoderServeConfig
        config = CrossEncoderServeConfig(
            s3_bucket=str(raw_config.model.s3_bucket),
            model_id=str(raw_config.model.model_id),
            model_type=str(raw_config.model.model_type),
            local_path=Path(raw_config.model.local_path),
            host=str(raw_config.server.host),
            port=int(raw_config.server.port),
            workers=int(raw_config.server.workers),
            reload=bool(raw_config.server.reload),
            batch_size=int(raw_config.inference.batch_size),
            max_seq_length=int(raw_config.inference.max_seq_length),
            device=str(raw_config.inference.device),
            cache_enable=bool(raw_config.get('cache', {}).get('enable', False)),
            cache_ttl=int(raw_config.get('cache', {}).get('ttl', 3600)),
            cache_max_size=int(raw_config.get('cache', {}).get('max_size', 10000))
        )
        
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
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "rerank")
        query = input_data.get("query")
        
        if not query:
            return {"error": "Missing 'query' parameter"}
        
        if action == "rerank":
            documents = input_data.get("documents")
            if not documents or not isinstance(documents, list):
                return {"error": "Missing or invalid 'documents' parameter"}
            
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
            
            # Rerank
            reranked_results, rerank_time = reranker.rerank(
                query=query,
                documents=docs,
                top_n=top_n
            )
            
            return {
                "results": reranked_results,
                "total": len(reranked_results),
                "rerank_time_ms": rerank_time
            }
        
        elif action == "score_pair":
            document = input_data.get("document")
            if not document:
                return {"error": "Missing 'document' parameter"}
            
            score, score_time = reranker.score_pair(
                query=query,
                document=document
            )
            
            return {
                "score": score,
                "score_time_ms": score_time
            }
        
        elif action == "batch_score":
            documents = input_data.get("documents")
            if not documents or not isinstance(documents, list):
                return {"error": "Missing or invalid 'documents' parameter"}
            
            scores, score_time = reranker.score_batch(
                query=query,
                documents=documents
            )
            
            return {
                "scores": scores,
                "count": len(scores),
                "score_time_ms": score_time
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})