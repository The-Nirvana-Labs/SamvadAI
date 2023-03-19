from hnswlib import Index
from typing import List


def embedding_to_indexes(embeddings: List[List[float]]) -> Index:
    """
    Builds a HNSWLIB index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (List[List[float]]): The input embeddings to build the HNSWLIB index for.

    Returns:
    - An instance of `Index` representing the HNSWLIB index.
    """
    # Set hyperparameters
    num_elements = len(embeddings)
    dim = len(embeddings[0])
    max_elements = num_elements * 2

    # Build the index
    index = Index(dim, "cosine")
    index.init_index(max_elements, ef_construction=100, M=64)
    index.add_items(embeddings)

    return index
