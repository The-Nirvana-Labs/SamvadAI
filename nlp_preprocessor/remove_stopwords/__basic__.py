from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def remove_stopwords(text):
    """
    Removes stopwords from the input text and returns the filtered text as a string.

    Args:
    - text (str): The input text to filter.

    Returns:
    - str: The filtered text as a string, with stopwords removed and all words converted to lowercase.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)
