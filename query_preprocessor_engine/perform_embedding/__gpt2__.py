import openai
import torch
import os
from transformers import GPT2Tokenizer, GPT2Model


def generate_embeddings(text):
    """
    Returns the embeddings of the input text using the pre-trained GPT-2 model.

    Args:
    text (str): The input text to convert to embeddings.

    Returns:
    torch.Tensor: A tensor of shape (1, 768) containing the embeddings of the input text.
    """
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    # Tokenize input text and convert to tensor
    inputs = tokenizer(text, return_tensors='pt')

    # Get model outputs and extract the last hidden state
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    # Average the last hidden state over the sequence dimension to get a single embedding vector
    embeddings = torch.mean(last_hidden_state, dim=1)

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
