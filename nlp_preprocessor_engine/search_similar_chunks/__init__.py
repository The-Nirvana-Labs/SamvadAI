from typing import List
import openai
import faiss
import os


def search_similar_chunks(query: str, faiss_index, extract_chunks_func, load_text_func, k: int = 3) -> List[str]:
    """
    Searches for the most similar chunks to the input query using an FAISS index built with OpenAI's text-davinci-002 embedding model.

    Args:
    - query (str): The query string to search for similar chunks.
    - faiss_index: An FAISS index built with OpenAI's text-davinci-002 embedding model.
    - extract_chunks_func: A function that takes a string as input and returns a list of noun chunks found in the input text, represented as strings.
    - load_text_func: A function that loads and returns the text to search for similar chunks in.
    - k (int, optional): The number of similar chunks to return. Default is 3.

    Returns:
    - list: A list of k similar chunks found in the input text, represented as strings.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embedding = openai.Embedding("text-davinci-002")
    query_vector = embedding.embed(query)
    query_vector = faiss.normalize_L2(query_vector)
    distances, indices = faiss_index.search(query_vector, k)
    similar_chunks = []
    text = load_text_func()
    chunks = extract_chunks_func(text)
    for index in indices[0]:
        similar_chunks.append(chunks[index])
    return similar_chunks
