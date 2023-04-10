# Import the required libraries
from transformers import GPT2Tokenizer


# def count_tokens(string):
#     count = len(string)
#     result = count / 4
#     return result


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a given string input using the GPT-2 tokenizer.

    Parameters:
        text (str): The input string to tokenize.

    Returns:
        int: The number of tokens in the input string.
    """

    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode the input string using the tokenizer
    encoding = tokenizer.encode(text)

    # Return the number of tokens in the encoding
    return len(encoding)
