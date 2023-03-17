from milvus import Milvus, IndexType, MetricType
import numpy as np


def embedding_to_indexes(embeddings: np.ndarray, collection_name: str) -> Milvus:
    """
    Builds a Milvus index for the input embeddings.

    Args:
    - embeddings (np.ndarray): The input embeddings to build the Milvus index for.
    - collection_name (str): The name of the collection to create in Milvus.

    Returns:
    - An instance of `Milvus` representing the Milvus index.
    """

    # Create a connection to Milvus
    milvus = Milvus()

    # Create the collection
    milvus.create_collection(collection_name, {
        "fields": [{"name": "embedding", "type": "float", "params": {"dim": embeddings.shape[1]}}]})

    # Insert the embeddings into the collection
    ids = milvus.insert(collection_name, embeddings.tolist())["ids"]

    # Create an index for the collection
    index_type = IndexType.IVF_FLAT
    metric_type = MetricType.IP
    milvus.create_index(collection_name, index_type, {"m": 16})

    return milvus
