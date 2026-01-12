import torch
import numpy as np
from typing import List, Tuple
import time

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False

from src.common.logging_config import get_logger
from src.entity.config_entity import BiEncoderServeConfig

logger = get_logger(__name__)

class BiEncoderEmbedder:
    """Encode texts into embeddings using fine-tuned bi-encoder"""

    def __init__(
        self,
        model,
        tokenizer,
        config: BiEncoderServeConfig,
        auto_tokenize: bool = True
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(self.config.device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.auto_tokenize = auto_tokenize
        
        if self.auto_tokenize and not PYVI_AVAILABLE:
            logger.warning("⚠️  pyvi not available, Vietnamese tokenization disabled")
            self.auto_tokenize = False
        
        logger.info(
            "BiEncoderEmbedder initialized | "
            f"max_length={self.config.max_seq_length}, "
            f"batch_size={self.config.batch_size}, "
            f"device={self.device}, "
            f"auto_tokenize={self.auto_tokenize}"
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess Vietnamese text with word segmentation"""
        if not text or not text.strip():
            return text
            
        if self.auto_tokenize and PYVI_AVAILABLE:
            try:
                # Vietnamese word segmentation
                tokenized = ViTokenizer.tokenize(text)
                return tokenized
            except Exception as e:
                logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
                return text
        
        return text

    @torch.no_grad()
    def encode_single(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Encode single text
        
        Args:
            text: Raw Vietnamese text (will be tokenized automatically)
        
        Returns:
            embedding: numpy array
            encode_time: time in milliseconds
        """
        start_time = time.time()
        
        # Auto tokenize Vietnamese text
        text = self._preprocess_text(text)
        
        # Tokenize with transformer tokenizer
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        embedding = self.model.encode(**inputs)
        embedding_np = embedding.cpu().numpy()[0]

        encode_time = (time.time() - start_time) * 1000
        return embedding_np, encode_time
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], float]:
        """
        Encode multiple texts in batches
        
        Args:
            texts: List of raw Vietnamese texts (will be tokenized automatically)
        
        Returns:
            embeddings: list of numpy arrays
            encode_time: total time in milliseconds
        """
        start_time = time.time()
        
        # Auto tokenize all texts
        texts = [self._preprocess_text(t) for t in texts]
        
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            embeddings = self.model.encode(**inputs)
            all_embeddings.extend(embeddings.cpu().numpy())

        encode_time = (time.time() - start_time) * 1000
        return all_embeddings, encode_time