import torch
import numpy as np
from typing import List, Tuple
import time

from src.common.logging_config import get_logger
from src.entity.config_entity import BiEncoderServeConfig

logger = get_logger(__name__)

class BiEncoderEmbedder:
    """Encode texts into embeddings using fine-tuned bi-encoder"""

    def __init__(
        self,
        model,
        tokenizer,
        config: BiEncoderServeConfig
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(self.config.device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        logger.info(
            "BiEncoderEmbedder initialized | "
            f"max_length={self.config.max_seq_length}, batch_size={self.config.batch_size}, device={self.device}"
        )

    @torch.no_grad()
    def encode_single(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Encode single text
        
        Returns:
            embedding: numpy array
            encode_time: time in milliseconds
        """
        start_time = time.time()

        # Tokenize
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

        # Convert to numpy
        embedding_np = embedding.cpu().numpy()[0]

        encode_time = (time.time() - start_time) * 1000

        return embedding_np, encode_time
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], float]:
        """
        Encode multiple texts in batches
        
        Returns:
            embeddings: list of numpy arrays
            encode_time: total time in milliseconds
        """
        start_time = time.time()

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]

            # Tokenize in batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode batch
            embeddings = self.model.encode(**inputs)

            # Convert to numpy and collect
            all_embeddings.extend(embeddings.cpu().numpy())

        encode_time = (time.time() - start_time) * 1000

        return all_embeddings, encode_time