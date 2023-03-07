import openai
import torch
from transformers import AutoTokenizer, AutoModel
import os


def generate_embeddings(text):
    """
    Generates embeddings for the input text using a pre-trained model.

    Args:
    - text (str): The input text to generate embeddings for.

    Returns:
    - list: A list of embeddings, where each embedding is a list of floats.
    """
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"  # or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
    return embeddings


def perform_embedding(embeddings):
    """
    Performs a task using the input embeddings and returns the result.

    Args:
    - embeddings (list): A list of embeddings, where each embedding is a list of floats.
    - api_key (str): The OpenAI API key to use for the task.

    Returns:
    - str: The result of the task.
    """

    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = "Perform a task using the following embeddings:\n" + str(embeddings)
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.5,
        max_tokens=256,
        n=1,
        stop=None,
    )
    return response.choices[0].text
