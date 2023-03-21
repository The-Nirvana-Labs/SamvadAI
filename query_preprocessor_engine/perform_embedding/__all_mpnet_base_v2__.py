import torch
from sentence_transformers import SentenceTransformer


def generate_embedding(sentence: str) -> torch.Tensor:
    """
    Generates an embedding for a sentence using the all-mpnet-base-v2 model from SentenceTransformers.

    Args:
    - sentence (str): A string representing the sentence to generate an embedding for.

    Returns:
    - embedding (torch.Tensor): A torch Tensor representing the embedding for the input sentence.
    """
    # Load all-mpnet-base-v2 model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Generate embedding
    embedding = model.encode(sentence, show_progress_bar=True)

    return embedding
