import openai
import os


def perform_embedding(text):
    """
    Performs text embedding using the OpenAI API.

    Args:
    - text (str): The input text to perform embedding on.

    Returns:
    - str: The resulting text embedding.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=text,
        temperature=0.5,
        max_tokens=50,
        n=1,
        stop=None,
    )
    return response.choices[0].text
