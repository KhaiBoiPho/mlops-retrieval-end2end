# src/serve/cross_encoder_service/app.py

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from src.serve.cross_encoder_service.model_loader import CrossEncoderModelLoader
from src.serve.cross_encoder_service.reranker import CrossEncoderReranker
from src.serve.cross_encoder_service.schema import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    ScorePairRequest,
    ScorePairResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse
)
from src.common.configuration import ConfigurationManager
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Global state
reranker: CrossEncoderReranker = None
config = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global reranker, config, startup_time
    
    logger.info("="*80)
    logger.info("STARTING CROSS-ENCODER RERANKING SERVICE")
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
        
        startup_time = time.time()
        
        logger.info("="*80)
        logger.info("✓ RERANKING SERVICE READY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Cross-Encoder service...")


app = FastAPI(
    title="Cross-Encoder Reranking Service",
    description="Rerank documents using fine-tuned cross-encoder for legal RAG",
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
        "service": "Cross-Encoder Reranking Service",
        "description": "Fine-tuned on Vietnamese legal Q&A pairs",
        "version": "1.0.0",
        "endpoints": {
            "rerank": "POST /rerank - Rerank document candidates",
            "score_pair": "POST /score_pair - Score single query-document pair",
            "batch_score": "POST /batch_score - Score multiple documents",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        if reranker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
        
        uptime = time.time() - startup_time if startup_time else 0
        
        return HealthResponse(
            status="healthy",
            model_version="v1.0",  # ✅ Hardcode hoặc từ config
            device=str(config.device) if config else "unknown",
            uptime_seconds=float(uptime)
        )
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/rerank", response_model=RerankResponse, tags=["Reranking"])
async def rerank_documents(request: RerankRequest):
    """
    Rerank candidate documents
    
    Use case:
    - After initial retrieval (bi-encoder or BM25)
    - Get top-K candidates → Rerank → Return top-N
    
    Returns:
        Reranked documents with scores
    """
    if reranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        # Prepare documents
        documents = [
            {
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata
            }
            for doc in request.documents
        ]
        
        # Rerank
        reranked_results, rerank_time = reranker.rerank(
            query=request.query,
            documents=documents,
            top_n=request.top_n
        )
        
        # Format response
        results = [
            RerankResult(
                id=r['id'],
                text=r['text'],
                score=r['score'],
                rank=r['rank'],
                metadata=r.get('metadata')
            )
            for r in reranked_results
        ]
        
        return RerankResponse(
            query=request.query,
            results=results,
            total=len(results),
            rerank_time_ms=rerank_time
        )
        
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reranking failed: {str(e)}"
        )


@app.post("/score_pair", response_model=ScorePairResponse, tags=["Scoring"])
async def score_pair(request: ScorePairRequest):
    """
    Score a single query-document pair
    
    Use case:
    - Quick relevance check
    - Testing model behavior
    
    Returns:
        Relevance score (0-1)
    """
    if reranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        score, score_time = reranker.score_pair(
            query=request.query,
            document=request.document
        )
        
        return ScorePairResponse(
            query=request.query,
            document=request.document[:200] + "..." if len(request.document) > 200 else request.document,
            score=score,
            score_time_ms=score_time
        )
        
    except Exception as e:
        logger.error(f"Score pair error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {str(e)}"
        )


@app.post("/batch_score", response_model=BatchScoreResponse, tags=["Scoring"])
async def batch_score(request: BatchScoreRequest):
    """
    Score multiple documents for a single query
    
    Use case:
    - Bulk scoring without full reranking
    - Get scores for all candidates
    
    Returns:
        List of relevance scores
    """
    if reranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        scores, score_time = reranker.score_batch(
            query=request.query,
            documents=request.documents
        )
        
        return BatchScoreResponse(
            query=request.query,
            scores=scores,
            count=len(scores),
            score_time_ms=score_time
        )
        
    except Exception as e:
        logger.error(f"Batch score error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serve.cross_encoder_service.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )