from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import openai
import os


def build_annoy_index(text, model_name):
    """
    Builds an Annoy index for the input text using Sentence-Transformers and OpenAI's text-davinci-002 embedding model.

    Args:
    - text (str): The input text to build the Annoy index for.
    - model_name (str): The name of the Sentence-Transformers model to use for generating embeddings.

    Returns:
    - An instance of `AnnoyIndex` representing the Annoy index.
    """
    # Load the model
    model = SentenceTransformer(model_name)

    # Get the embeddings
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with openai.api_key:
        embeddings = model.encode(text, show_progress_bar=False)

    # Build the index
    index = AnnoyIndex(embeddings.shape[1], metric='angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(100)

    return index
