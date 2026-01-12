import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from src.models.bi_encoder.pooling import apply_pooling
from src.common.logging_config import get_logger

logger = get_logger(__name__)


class BiEncoder(nn.Module):
    """
    Bi-Encoder model for semantic similarity
    Uses mean pooling over transformer outputs
    """

    def __init__(
        self,
        model_name: str,
        pooling: str,
        normalize: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.normalize = normalize

        # Load pretrained transformer
        logger.info(f"Loading model: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.config.hidden_size

        logger.info(f"Model loaded with hidden_size={self.hidden_size}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        # Get transformer outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get token embeddings
        token_embeddings = outputs.last_hidden_state
        
        # Pooling
        if self.pooling == "mean" or "cls":
            embeddings = apply_pooling(
                token_embeddings=token_embeddings,
                attention_mask=attention_mask,
                pooling=self.pooling
            )
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Normalize embeddings
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode(self, **inputs) -> torch.Tensor:
        """Convenience method for encoding"""
        return self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )