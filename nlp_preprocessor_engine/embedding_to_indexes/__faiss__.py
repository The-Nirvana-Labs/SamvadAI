import faiss
import numpy as np


def embedding_to_indexes(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the FAISS index for.

    Returns:
    - An instance of `faiss.IndexFlatIP` representing the FAISS index.
    """
    # Set hyperparameters
    index_type = faiss.IndexFlatIP
    metric = faiss.METRIC_INNER_PRODUCT

    # Build the index
    index = index_type(embeddings.shape[1], metric)
    index.add(embeddings)

    return index
