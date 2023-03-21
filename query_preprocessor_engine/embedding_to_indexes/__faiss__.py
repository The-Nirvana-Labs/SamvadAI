import faiss
import numpy as np


def embedding_to_indexes(embeddings: np.ndarray) -> np.ndarray:
    """
    Builds a FAISS index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the FAISS index for.
                               Shape: (num_embeddings, embedding_dim)

    Returns:
    - An np.ndarray representing the FAISS index.
      Shape: (num_embeddings, )
    """
    # Set hyperparameters
    index_type = faiss.IndexFlatIP
    metric = faiss.METRIC_INNER_PRODUCT

    # Build the index
    index = index_type(embeddings.shape[1], metric)
    index.add(embeddings)

    # Generate the index for all embeddings
    _, index_array = index.search(embeddings, 1)

    return index_array
