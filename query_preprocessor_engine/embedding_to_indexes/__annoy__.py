from annoy import AnnoyIndex
import torch


def embedding_to_indexes(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Builds an Annoy index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (torch.Tensor): The input embeddings to build the Annoy index for.
                               Shape: (num_embeddings, embedding_dim)

    Returns:
    - A torch.Tensor representing the Annoy index.
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
    index_tensor = torch.zeros((embeddings.shape[0],), dtype=torch.long)
    for i, embedding in enumerate(embeddings):
        index_tensor[i] = index.get_nns_by_vector(embedding.tolist(), 1)[0]

    return index_tensor
