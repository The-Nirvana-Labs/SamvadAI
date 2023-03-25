import spacy
from typing import List
nlp = spacy.load('en_core_web_sm')


def remove_stopwords(text: str) -> str:
    """
    Removes stopwords from the input text and returns the filtered text as a string.

    Parameters:
    - text (str): The input text to filter.

    Returns:
    - str: The filtered text as a string, with stopwords removed and all words converted to lowercase.

    Raises:
    - TypeError: If the input text is not a string.

    Example:
    >>> sample = "This is a sample text that we will use to test the stopwords function."
    >>> stopwords(text)
    'sample text test stopwords function .'

    Performance:
    - Time complexity: O(n), where n is the number of tokens in the input text.
    - Space complexity: O(n), where n is the number of tokens in the input text.

    Dependencies:
    - This function requires the `spacy` library to be installed, which can be installed via pip.
    """
    # Check that input is a string
    if not isinstance(text, str):
        raise TypeError("Input text must be a string")

    # Tokenize text with spaCy
    doc = nlp(text)

    # Filter out stopwords and convert to lowercase
    filtered_tokens: List[str] = [token.text.lower() for token in doc if not token.is_stop]

    # Join filtered tokens back into a string and return
    filtered_text: str = ' '.join(filtered_tokens)
    return filtered_text
