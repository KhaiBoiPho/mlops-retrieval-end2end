import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.common.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoder(nn.Module):
    """
    Cross-Encoder for document reranking
    
    Input: [CLS] query [SEP] document [SEP]
    Output: Relevance score (0-1)
    
    Architecture:
    - BERT/RoBERTa encoder
    - Pooling (CLS token)
    - Classification head (linear layer)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)

        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Dropout
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        logger.info(f"CrossEncoder initialized: {model_name}")
        logger.info(f"  Hidden size: {self.config.hidden_size}")
        logger.info(f"  Num labels: {num_labels}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)
            
        Returns:
            logits: [batch_size, num_labels] relevance scores
        """

        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Pool - use [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        return logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict relevance scores (with sigmoid)
        
        Returns:
            scores: [batch_size] relevance scores (0-1)
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        scores = torch.sigmoid(logits).squeeze(-1)  # [batch_size]
        return scores
    
    def save_pretrained(self, save_directory: str):
        """Save model and config"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(save_directory)
        
        # Save classifier weights
        torch.save({
            'classifier': self.classifier.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels
        }, os.path.join(save_directory, 'classifier.pt'))
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from directory"""
        import os
        
        # Load classifier config
        classifier_path = os.path.join(load_directory, 'classifier.pt')
        checkpoint = torch.load(classifier_path)
        
        # Initialize model
        model = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels']
        )
        
        # Load encoder
        model.encoder = AutoModel.from_pretrained(load_directory)
        
        # Load classifier weights
        model.classifier.load_state_dict(checkpoint['classifier'])
        
        return model