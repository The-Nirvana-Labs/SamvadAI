import numpy as np
from annoy import AnnoyIndex


def embedding_to_indexes(embeddings: np.ndarray) -> np.ndarray:
    """
    Builds an Annoy index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the Annoy index for.
                               Shape: (num_embeddings, embedding_dim)

    Returns:
    - An np.ndarray representing the Annoy index.
      Shape: (num_embeddings, )
    """
    # Set hyperparameters
    n_trees = 100
    metric = 'angular'
    embedding_dim = embeddings.shape[1]

    # Build the index
    index = AnnoyIndex(embedding_dim, metric=metric)
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding.tolist())
    index.build(n_trees)

    # Generate the index for all embeddings
    index_array = np.zeros((embeddings.shape[0],), dtype=int)
    for i, embedding in enumerate(embeddings):
        index_array[i] = index.get_nns_by_vector(embedding.tolist(), 1)[0]

    return index_array
