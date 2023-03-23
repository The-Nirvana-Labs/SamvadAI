from annoy import AnnoyIndex
import faiss
import numpy as np
import hnswlib
from milvus import Milvus, IndexType, MetricType

# Set hyper parameters
index_type = faiss.IndexFlatIP
metric_faiss = faiss.METRIC_INNER_PRODUCT


def embedding_to_indexes_faiss(embeddings: np.ndarray) -> np.ndarray:
    """
    Builds a FAISS index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the FAISS index for.
                               Shape: (num_embeddings, embedding_dim)

    Returns:
    - An np.ndarray representing the FAISS index.
      Shape: (num_embeddings, )
    """

    # Build the index
    index = index_type(embeddings.shape[1], metric_faiss)
    index.add(embeddings)

    # Generate the index for all embeddings
    _, index_array = index.search(embeddings, 1)

    return index_array


# Set hyper parameters
n_trees = 100
metric_annoy = 'angular'


def embedding_to_indexes_annoy(embeddings: np.ndarray) -> np.ndarray:
    """
    Builds an Annoy index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the Annoy index for.
                               Shape: (num_embeddings, embedding_dim)

    Returns:
    - An np.ndarray representing the Annoy index.
      Shape: (num_embeddings, )
    """
    embedding_dim = embeddings.shape[1]

    # Build the index
    index = AnnoyIndex(embedding_dim, metric=metric_annoy)
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding.tolist())
    index.build(n_trees)

    # Generate the index for all embeddings
    index_array = np.zeros((embeddings.shape[0],), dtype=int)
    for i, embedding in enumerate(embeddings):
        index_array[i] = index.get_nns_by_vector(embedding.tolist(), 1)[0]

    return index_array


def embedding_to_indexes_hnswlib(embeddings: np.ndarray) -> np.ndarray:
    """
    Builds a HNSWLIB index for the input embeddings using the given hyperparameters.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the HNSWLIB index for.
                               Shape: (num_embeddings, embedding_dim)

    Returns:
    - An np.ndarray representing the HNSWLIB index.
      Shape: (num_embeddings, )
    """
    # Set hyper parameters
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


index_type_ivf = IndexType.IVF_FLAT
metric_type = MetricType.IP
search_param_milvus = {"metric_type": metric_type}


def embedding_to_indexes_milvus(embeddings: np.ndarray, collection_name: str) -> np.ndarray:
    """
    Builds a Milvus index for the input embeddings.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the Milvus index for.
                               Shape: (num_embeddings, embedding_dim)
    - collection_name (str): The name of the collection to create in Milvus.

    Returns:
    - An np.ndarray representing the Milvus index.
      Shape: (num_embeddings, )
    """

    # Create a connection to Milvus
    milvus = Milvus()

    # Create the collection
    milvus.create_collection(collection_name, {
        "fields": [{"name": "embedding", "type": "float", "params": {"dim": embeddings.shape[1]}}]})

    # Insert the embeddings into the collection
    milvus.insert(collection_name, embeddings.tolist())

    milvus.create_index(collection_name, index_type_ivf, {"m": 16})

    # Get the Milvus index
    results = milvus.search(collection_name, query_records=embeddings.tolist(), top_k=1, params=search_param_milvus)
    milvus_index = np.array([result[0].id for result in results])

    return milvus_index
