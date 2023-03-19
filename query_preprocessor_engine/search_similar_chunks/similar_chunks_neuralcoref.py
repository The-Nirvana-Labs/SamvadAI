from typing import List
import openai
import faiss
import os
import spacy
import neuralcoref


def search_similar_chunks(query: str, k: int = 3) -> List[str]:
    """
    Searches for the most similar chunks to the input query using an FAISS index built with OpenAI's text-davinci-002 embedding model.

    Args:
    - query (str): The query string to search for similar chunks.
    - k (int, optional): The number of similar chunks to return. Default is 3.

    Returns:
    - list: A list of k similar chunks found in the input text, represented as strings.
    """
    # Load the OpenAI API key from the environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Load the FAISS index and vector embeddings
    index_path = "faiss.index"
    faiss_index = faiss.read_index(index_path)
    faiss_vectors = faiss.read_bvec("faiss.bvecs")

    # Define a function to extract noun chunks from text
    def extract_chunks(text):
        spacy_nlp = spacy.load('en_core_web_trf')
        neuralcoref.add_to_pipe(spacy_nlp)
        doc = spacy_nlp(text)
        chunks = []
        for chunk in doc.noun_chunks:
            # Use a binary classifier to filter out non-noun phrases
            if chunk.root.pos_ == 'NOUN':
                # Use neuralcoref to resolve co-references within noun chunks
                resolved_chunk = doc._.coref_resolved[chunk.start:chunk.end].strip()
                chunks.append(resolved_chunk)
        return chunks

    # Define a function to load the text to search for similar chunks in
    def load_text():
        with open("input.txt", "r") as f:
            text = f.read()
        return text

    # Embed the query and normalize the vector
    embedding = openai.Embedding("text-davinci-002")
    query_vector = embedding.embed(query)
    query_vector = faiss.normalize_L2(query_vector)

    # Use the FAISS index to find the most similar vectors
    distances, indices = faiss_index.search(query_vector, k)

    # Extract the similar chunks from the text
    similar_chunks = []
    text = load_text()
    chunks = extract_chunks(text)
    for index in indices[0]:
        similar_chunks.append(chunks[index])

    return similar_chunks
