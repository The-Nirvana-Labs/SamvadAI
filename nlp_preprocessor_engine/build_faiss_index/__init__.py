import faiss
import openai


def build_faiss_index(text, openai_api_key):
    """
    Builds an FAISS index for the input text using OpenAI's text-davinci-002 embedding model.

    Args:
    - text (str): The input text to build the FAISS index for.
    - openai_api_key (str): The API key to use for OpenAI's API.

    Returns:
    - index (faiss index object): An FAISS index object for the input text.
    """
    openai.api_key = openai_api_key
    embedding = openai.Embedding("text-davinci-002")
    chunk_size = 4096
    vectors = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        vector = embedding.embed(chunk)
        vectors.append(vector)
    vectors = faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index
