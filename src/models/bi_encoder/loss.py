import torch
import torch.nn as nn
from src.common.logging_config import get_logger

logger = get_logger(__name__)


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss for contrastive learning
    Uses in-batch negatives
    
    For each (query, positive) pair, all other positives in the batch
    are treated as negatives.
    """
    
    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            query_embeddings: [batch_size, hidden_size]
            positive_embeddings: [batch_size, hidden_size]
            
        Returns:
            loss: scalar
        """
        # Compute similarity scores
        # [batch_size, batch_size]
        scores = torch.matmul(query_embeddings, positive_embeddings.T) * self.scale
        
        # Labels are diagonal (each query matches with its positive)
        labels = torch.arange(scores.size(0), device=scores.device)
        
        # Compute cross entropy loss
        loss = self.cross_entropy(scores, labels)
        
        return loss


class CachedMultipleNegativesRankingLoss(nn.Module):
    """
    Cached version of Multiple Negatives Ranking Loss
    Accumulates embeddings over multiple mini-batches before computing loss
    
    This allows for more negatives without increasing GPU memory for forward pass
    """
    
    def __init__(
        self,
        scale: float = 20.0,
        mini_batch_size: int = 32,
        cache_size: int = 1024
    ):
        super().__init__()
        self.scale = scale
        self.mini_batch_size = mini_batch_size
        self.cache_size = cache_size
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Cache for embeddings
        self.query_cache = []
        self.positive_cache = []
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with caching
        
        Args:
            query_embeddings: [batch_size, hidden_size]
            positive_embeddings: [batch_size, hidden_size]
            
        Returns:
            loss: scalar
        """
        batch_size = query_embeddings.size(0)  # noqa: F841
        
        # Add to cache
        self.query_cache.append(query_embeddings.detach())
        self.positive_cache.append(positive_embeddings.detach())
        
        # If cache is full, compute loss
        if sum(q.size(0) for q in self.query_cache) >= self.cache_size:
            # Concatenate all cached embeddings
            all_queries = torch.cat(self.query_cache, dim=0)
            all_positives = torch.cat(self.positive_cache, dim=0)
            
            # Compute similarity scores
            scores = torch.matmul(all_queries, all_positives.T) * self.scale
            
            # Labels are diagonal
            labels = torch.arange(scores.size(0), device=scores.device)
            
            # Compute loss
            loss = self.cross_entropy(scores, labels)
            
            # Clear cache
            self.query_cache = []
            self.positive_cache = []
            
            return loss
        else:
            # Return zero loss if cache not full yet
            return torch.tensor(0.0, device=query_embeddings.device, requires_grad=True)