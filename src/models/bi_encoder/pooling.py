import torch

def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean pooling over token embeddings

    Args:
        token_embeddings: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]

    Returns:
        pooled: [batch_size, hidden_size]
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(
    token_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    CLS token pooling

    Args:
        token_embeddings: [batch_size, seq_len, hidden_size]

    Returns:
        pooled: [batch_size, hidden_size]
    """
    return token_embeddings[:, 0, :]


def apply_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str
) -> torch.Tensor:
    if pooling == "mean":
        return mean_pooling(token_embeddings, attention_mask)
    elif pooling == "cls":
        return cls_pooling(token_embeddings)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
