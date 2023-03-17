from annoy import AnnoyIndex
from typing import List


def embedding_to_indexes(embeddings: List[List[float]]) -> AnnoyIndex:
    """
    Builds an Annoy index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (List[List[float]]): The input embeddings to build the Annoy index for.

    Returns:
    - An instance of `AnnoyIndex` representing the Annoy index.
    """
    # Set hyperparameters
    n_trees = 100
    metric = 'angular'
    embedding_dim = len(embeddings[0])

    # Build the index
    index = AnnoyIndex(embedding_dim, metric=metric)
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(n_trees)

    return index
