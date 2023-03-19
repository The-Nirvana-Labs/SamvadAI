import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List


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
    >>> sample = "This is a sample text that we will use to test the remove_stopwords function."
    >>> remove_stopwords(text)
    'sample text use test remove_stopwords function .'

    Performance:
    - Time complexity: O(n), where n is the number of words in the input text.
    - Space complexity: O(n), where n is the number of words in the input text.

    Benchmarking: - To benchmark this function, you can use the `benchmark_remove_stopwords` function from the
    previous example, which returns a dictionary containing the benchmark results.

    Dependencies:
    - This function requires the `word_tokenize` library to be installed, which can be installed via pip.
    """
    # Check that input is a string
    if not isinstance(text, str):
        raise TypeError("Input text must be a string")

    # Load stopwords from NLTK library
    nltk.download('stopwords')
    stop_words: List[str] = set(stopwords.words('english'))

    # Tokenize text and filter out stopwords
    tokens: List[str] = word_tokenize(text)
    filtered_tokens: List[str] = [word.lower() for word in tokens if word.lower() not in stop_words]

    # Join filtered tokens back into a string and return
    filtered_text: str = ' '.join(filtered_tokens)
    return filtered_text

