from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from src.serve.bi_encoder_service.model_loader import BiEncoderModelLoader
from src.serve.bi_encoder_service.embedding import BiEncoderEmbedder
from src.serve.bi_encoder_service.schema import (
    EmbedRequest,
    EmbedResponse,
    BatchEmbedRequest,
    BatchEmbedResponse,
    HealthResponse
)
from src.common.configuration import ConfigurationManager
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Global state
embedder: BiEncoderEmbedder = None
config = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global embedder, config, startup_time

    logger.info("="*80)
    logger.info("STARTING BI-ENCODER EMBEDDING SERVICE")
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

        startup_time = time.time()

        logger.info("="*80)
        logger.info("✓ EMBEDDING SERVICE READY")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

    yield

    logger.info("Shutting down Bi-Encoder service...")


app = FastAPI(
    title="Bi-encoder Embedding Service",
    description="Legal document embedding service using fine-tuned bi-encoder",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Bi-Encoder Embedding Service",
        "description": "Fine-tuned on Vietnamese legal documents",
        "version": "1.0.0",
        "endpoints": {
            "embed": "POST /embed - Embed single text",
            "batch_embed": "POST /batch_embed - Embed multiple texts",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        if embedder is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - embedder is None"
            )
        
        if config is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - config is None"
            )
        
        uptime = time.time() - startup_time if startup_time else 0

        return HealthResponse(
            status="healthy",
            model_version="v1.0",
            device=str(config.device),
            uptime_seconds=float(uptime)
        )
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/embed", response_model=EmbedResponse, tags=["Embedding"])
async def embed_text(request: EmbedRequest):
    """
    Embed single text
    
    Use cases:
    - Embed user query for search
    - Embed single document
    
    Returns:
        Vector embedding (dense representation)
    """
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        embedding, encode_time = embedder.encode_single(request.text)

        return EmbedResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding),
            encode_time_ms=encode_time
        )
    
    except Exception as e:
        logger.error(f"Embed error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encoding failed: {str(e)}"
        )
    
@app.post("/batch_embed", response_model=BatchEmbedResponse, tags=["Embedding"])
async def batch_embed_texts(request: BatchEmbedRequest):
    """
    Embed multiple texts in batch (more efficient)
    
    Use cases:
    - Embed document chunks for indexing
    - Bulk embedding operations
    
    Returns:
        List of vector embeddings
    """
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        embeddings, encode_time = embedder.encode_batch(request.texts)
        
        return BatchEmbedResponse(
            embeddings=[emb.tolist() for emb in embeddings],
            count=len(embeddings),
            dimension=len(embeddings[0]) if embeddings else 0,
            encode_time_ms=encode_time
        )
        
    except Exception as e:
        logger.error(f"Batch embed error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch encoding failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serve.bi_encoder_service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )