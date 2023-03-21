import hnswlib
import numpy as np


def embedding_to_indexes(embeddings: np.ndarray) -> np.ndarray:
    """
    Builds a HNSWLIB index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the HNSWLIB index for.
                               Shape: (num_embeddings, embedding_dim)

    Returns:
    - An np.ndarray representing the HNSWLIB index.
      Shape: (num_embeddings, )
    """
    # Set hyperparameters
    num_elements = embeddings.shape[0]
    dim = embeddings.shape[1]
    max_elements = num_elements * 2

    # Build the index
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements, ef_construction=100, M=64)
    index.add_items(embeddings)

    # Generate the index for all embeddings
    index_array = np.zeros((num_elements,), dtype=int)
    for i, embedding in enumerate(embeddings):
        index_array[i] = index.knn_query(embedding, k=1)[0][0]

    return index_array
