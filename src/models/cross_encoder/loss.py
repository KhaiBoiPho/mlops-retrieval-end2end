import torch
import torch.nn as nn


class CrossEncoderLoss(nn.Module):
    """
    Loss function for cross-encoder training
    Binary Cross Entropy with Logits Loss
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            logits: [batch_size, 1] or [batch_size] model predictions
            labels: [batch_size] ground truth (0 or 1)
            
        Returns:
            loss: scalar
        """
        # Ensure correct shapes
        if logits.dim() == 2:
            logits = logits.squeeze(-1)  # [batch_size]
        
        labels = labels.float()  # Ensure float type
        
        loss = self.criterion(logits, labels)
        return loss


class MarginRankingLoss(nn.Module):
    """
    Margin-based ranking loss for triplet training
    Encourages positive pairs to have higher scores than negatives
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=margin)
    
    def forward(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin ranking loss
        
        Args:
            positive_logits: [batch_size] scores for positive pairs
            negative_logits: [batch_size] scores for negative pairs
            
        Returns:
            loss: scalar
        """
        # Target: positives should rank higher than negatives
        target = torch.ones_like(positive_logits)
        
        loss = self.criterion(positive_logits, negative_logits, target)
        return loss