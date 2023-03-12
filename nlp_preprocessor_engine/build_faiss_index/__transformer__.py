from transformers import AutoTokenizer, AutoModel
import faiss
import openai
import os


def build_faiss_index(text, model_name) -> faiss.IndexFlatIP:
    """
    Builds an FAISS index for the input text using Hugging Face Transformers and OpenAI's text-davinci-002 embedding model.

    Args:
    - text (str): The input text to build the FAISS index for.
    - model_name (str): The name of the Hugging Face Transformers model to use for generating embeddings.

    Returns:
    - An instance of `faiss.IndexFlatIP` representing the FAISS index.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the text and get the embeddings
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with openai.api_key:
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.detach().numpy()

    # Normalize the embeddings and build the index
    embeddings = faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index
