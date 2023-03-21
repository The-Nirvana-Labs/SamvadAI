import os
from typing import List
import torch
from sentence_transformers import SentenceTransformer


def generate_embeddings(text: str) -> torch.Tensor:
    """
    Generates an embedding for a given text using the all-MiniLM-L6-v2 model from SentenceTransformers.

    Args:
    - text (str): A string representing text to generate an embedding for.

    Returns:
    - embedding (torch.Tensor): A tensor representing the embedding for the input text.
    """
    # Load all-MiniLM-L6-v2 model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embedding
    embedding = model.encode(text, show_progress_bar=True)

    return embedding
