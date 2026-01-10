"""
RunPod Serverless Handler for Bi-Encoder Embedding Service

This handler wraps the FastAPI app to work with RunPod's serverless platform.
It supports both /embed and /batch_embed endpoints.
"""
import runpod
from typing import Dict, Any

# Import your FastAPI app components
from src.serve.bi_encoder_service.model_loader import BiEncoderModelLoader
from src.serve.bi_encoder_service.embedding import BiEncoderEmbedder
from src.common.configuration import ConfigurationManager
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Global state (loaded once on cold start)
embedder: BiEncoderEmbedder = None
config = None


def initialize_model():
    """Initialize model on cold start"""
    global embedder, config
    
    if embedder is not None:
        return  # Already initialized
    
    logger.info("="*80)
    logger.info("INITIALIZING BI-ENCODER FOR RUNPOD SERVERLESS")
    logger.info("="*80)
    
    try:
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.get_bi_encoder_serve_config()
        
        # Load model
        loader = BiEncoderModelLoader(config=config)
        model, tokenizer = loader.load_model()
        
        # Initialize embedder
        embedder = BiEncoderEmbedder(
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
            "action": "embed" | "batch_embed",
            "text": "single text" (for embed),
            "texts": ["text1", "text2"] (for batch_embed)
        }
    }
    
    Returns:
    {
        "output": {
            "embedding": [...] or "embeddings": [[...], [...]],
            "dimension": int,
            "encode_time_ms": float
        }
    }
    """
    try:
        # Initialize model if not done yet
        if embedder is None:
            initialize_model()
        
        # Extract input
        input_data = event.get("input", {})
        action = input_data.get("action", "embed")
        
        if action == "embed":
            # Single text embedding
            text = input_data.get("text")
            if not text:
                return {"error": "Missing 'text' in input"}
            
            embedding, encode_time = embedder.encode_single(text)
            
            return {
                "output": {
                    "embedding": embedding.tolist(),
                    "dimension": len(embedding),
                    "encode_time_ms": encode_time
                }
            }
        
        elif action == "batch_embed":
            # Batch embedding
            texts = input_data.get("texts")
            if not texts or not isinstance(texts, list):
                return {"error": "Missing 'texts' list in input"}
            
            embeddings, encode_time = embedder.encode_batch(texts)
            
            return {
                "output": {
                    "embeddings": [emb.tolist() for emb in embeddings],
                    "count": len(embeddings),
                    "dimension": len(embeddings[0]) if embeddings else 0,
                    "encode_time_ms": encode_time
                }
            }
        
        else:
            return {"error": f"Unknown action: {action}. Use 'embed' or 'batch_embed'"}
    
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    # Start RunPod serverless
    logger.info("Starting RunPod serverless handler for Bi-Encoder...")
    runpod.serverless.start({"handler": handler})