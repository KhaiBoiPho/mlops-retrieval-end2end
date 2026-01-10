"""
RunPod Serverless Handler for Cross-Encoder Reranking Service

This handler wraps the reranking logic to work with RunPod's serverless platform.
"""
import runpod
from typing import Dict, Any

from src.serve.cross_encoder_service.model_loader import CrossEncoderModelLoader
from src.serve.cross_encoder_service.reranker import CrossEncoderReranker
from src.common.configuration import ConfigurationManager
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Global state
reranker: CrossEncoderReranker = None
config = None


def initialize_model():
    """Initialize model on cold start"""
    global reranker, config
    
    if reranker is not None:
        return  # Already initialized
    
    logger.info("="*80)
    logger.info("INITIALIZING CROSS-ENCODER FOR RUNPOD SERVERLESS")
    logger.info("="*80)
    
    try:
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.get_cross_encoder_serve_config()
        
        # Load model
        loader = CrossEncoderModelLoader(config)
        model, tokenizer = loader.load_model()
        
        # Initialize reranker
        reranker = CrossEncoderReranker(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        logger.info("✓ Model initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    
    Expected event format:
    {
        "input": {
            "action": "rerank" | "score_pair" | "batch_score",
            "query": "user query",
            "documents": [{"id": "1", "text": "doc1"}, ...] (for rerank),
            "document": "single doc" (for score_pair),
            "top_n": 10 (optional, for rerank)
        }
    }
    
    Returns:
    {
        "output": {
            "results": [...] or "score": float,
            "total": int,
            "rerank_time_ms": float
        }
    }
    """
    try:
        # Initialize model if not done yet
        if reranker is None:
            initialize_model()
        
        # Extract input
        input_data = event.get("input", {})
        action = input_data.get("action", "rerank")
        query = input_data.get("query")
        
        if not query:
            return {"error": "Missing 'query' in input"}
        
        if action == "rerank":
            # Rerank documents
            documents = input_data.get("documents")
            if not documents:
                return {"error": "Missing 'documents' in input"}
            
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
                "output": {
                    "query": query,
                    "results": reranked_results,
                    "total": len(reranked_results),
                    "rerank_time_ms": rerank_time
                }
            }
        
        elif action == "score_pair":
            # Score single pair
            document = input_data.get("document")
            if not document:
                return {"error": "Missing 'document' in input"}
            
            score, score_time = reranker.score_pair(
                query=query,
                document=document
            )
            
            return {
                "output": {
                    "query": query,
                    "document": document[:200] + "..." if len(document) > 200 else document,
                    "score": score,
                    "score_time_ms": score_time
                }
            }
        
        elif action == "batch_score":
            # Score multiple documents
            documents = input_data.get("documents")
            if not documents:
                return {"error": "Missing 'documents' in input"}
            
            scores, score_time = reranker.score_batch(
                query=query,
                documents=documents
            )
            
            return {
                "output": {
                    "query": query,
                    "scores": scores,
                    "count": len(scores),
                    "score_time_ms": score_time
                }
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler for Cross-Encoder...")
    runpod.serverless.start({"handler": handler})