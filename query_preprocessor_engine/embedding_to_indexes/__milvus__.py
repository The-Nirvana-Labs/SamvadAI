from milvus import Milvus, IndexType, MetricType
import numpy as np


def embedding_to_indexes(embeddings: np.ndarray, collection_name: str) -> np.ndarray:
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
    ids = milvus.insert(collection_name, embeddings.tolist())["ids"]

    # Create an index for the collection
    index_type = IndexType.IVF_FLAT
    metric_type = MetricType.IP
    milvus.create_index(collection_name, index_type, {"m": 16})

    # Get the Milvus index
    search_param = {"metric_type": metric_type}
    results = milvus.search(collection_name, query_records=embeddings.tolist(), top_k=1, params=search_param)
    milvus_index = np.array([result[0].id for result in results])

    return milvus_index
