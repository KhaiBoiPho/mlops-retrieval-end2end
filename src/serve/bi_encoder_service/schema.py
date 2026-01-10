from pydantic import BaseModel, Field
from typing import List

class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Điều kiện để ly hôn theo quy định của pháp luật Việt Nam"
            }
        }

class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Dense vector embedding")
    dimension: int = Field(..., description="Embedding dimension")
    encode_time_ms: float = Field(..., description="Encoding time in milliseconds")

class BatchEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_length=1, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Điều kiện để ly hôn",
                    "Quyền và nghĩa vụ của cha mẹ",
                    "Thủ tục kết hôn"
                ]
            }
        }

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embeddings")
    count: int = Field(..., description="Number of embeddings")
    dimension: int = Field(..., description="Embedding dimension")
    encode_time_ms: float = Field(..., description="Total encoding time in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Model version")
    device: str = Field(..., description="Device (cuda/cpu)")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")