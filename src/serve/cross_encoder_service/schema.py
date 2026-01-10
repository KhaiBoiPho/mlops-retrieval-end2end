from pydantic import BaseModel, Field
from typing import List, Optional

class Document(BaseModel):
    """Document for reranking - FLEXIBLE ID"""
    id: str = Field(
        ..., 
        description="Any unique identifier: chunk_id, doc_id, URL, UUID, etc."
    )
    text: str = Field(..., description="Document text content")
    metadata: Optional[dict] = Field(None, description="Optional metadata")
    
    class Config:
        json_schema_extra = {
            "examples": [
                # Example 1: Chunk-based RAG
                {
                    "id": "chunk_abc123",
                    "text": "Điều 51. Điều kiện ly hôn...",
                    "metadata": {
                        "doc_id": "civil_code_2015",
                        "chunk_index": 5,
                        "page": 23
                    }
                },
                # Example 2: Document-based
                {
                    "id": "doc_456",
                    "text": "Full article text...",
                    "metadata": {
                        "source": "legal_code",
                        "article": 51
                    }
                },
                # Example 3: Database record
                {
                    "id": "db_record_789",
                    "text": "Document content...",
                    "metadata": {
                        "db_table": "legal_articles",
                        "primary_key": 789
                    }
                },
                # Example 4: URL-based
                {
                    "id": "https://example.com/article/51",
                    "text": "Article content...",
                    "metadata": None
                }
            ]
        }

class RerankRequest(BaseModel):
    """Request for reranking documents"""
    query: str = Field(..., description="Search query", min_length=1)
    documents: List[Document] = Field(
        ...,
        description="List of candidate documents to rerank",
        min_length=1,
        max_length=1000
    )
    top_n: int = Field(
        10,
        description="Number of top results to return after reranking",
        ge=1,
        le=100
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Điều kiện để ly hôn là gì?",
                "documents": [
                    {"id": "doc_1", "text": "Điều 51. Điều kiện ly hôn..."},
                    {"id": "doc_2", "text": "Điều 52. Quyền yêu cầu ly hôn..."}
                ],
                "top_n": 5
            }
        }

class RerankResult(BaseModel):
    """Single reranked result"""
    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score from cross-encoder")
    rank: int = Field(..., description="New rank after reranking")
    metadata: Optional[dict] = Field(None, description="Optional metadata")

class RerankResponse(BaseModel):
    """Response with reranked documents"""
    query: str = Field(..., description="Original query")
    results: List[RerankResult] = Field(..., description="Reranked documents")
    total: int = Field(..., description="Number of results returned")
    rerank_time_ms: float = Field(..., description="Reranking time in milliseconds")

class ScorePairRequest(BaseModel):
    """Request to score a single query-document pair"""
    query: str = Field(..., description="Search query", min_length=1)
    document: str = Field(..., description="Document text", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Điều kiện ly hôn",
                "document": "Điều 51. Điều kiện ly hôn: Vợ chồng có quyền ly hôn..."
            }
        }

class ScorePairResponse(BaseModel):
    """Response with relevance score"""
    query: str = Field(..., description="Query")
    document: str = Field(..., description="Document text (truncated)")
    score: float = Field(..., description="Relevance score (0-1)")
    score_time_ms: float = Field(..., description="Scoring time in milliseconds")

class BatchScoreRequest(BaseModel):
    """Request to score multiple documents for one query"""
    query: str = Field(..., description="Search query", min_length=1)
    documents: List[str] = Field(
        ...,
        description="List of document texts",
        min_length=1,
        max_length=1000
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Điều kiện ly hôn",
                "documents": [
                    "Điều 51. Điều kiện ly hôn...",
                    "Điều 52. Quyền yêu cầu ly hôn..."
                ]
            }
        }

class BatchScoreResponse(BaseModel):
    """Response with multiple scores"""
    query: str = Field(..., description="Query")
    scores: List[float] = Field(..., description="Relevance scores (0-1)")
    count: int = Field(..., description="Number of documents scored")
    score_time_ms: float = Field(..., description="Total scoring time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Model version")
    device: str = Field(..., description="Device (cuda/cpu)")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")