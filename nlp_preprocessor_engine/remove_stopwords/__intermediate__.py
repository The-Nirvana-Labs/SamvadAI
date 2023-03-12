import spacy

nlp = spacy.load('en_core_web_sm')


def remove_stopwords(text):
    """
    Removes stopwords from the input text and returns the filtered text as a string.

    Args:
    - text (str): The input text to filter.

    Returns:
    - str: The filtered text as a string, with stopwords removed and all words converted to lowercase.
    """
    doc = nlp(text)
    filtered_tokens = [token.text.lower() for token in doc if not token.is_stop]
    return ' '.join(filtered_tokens)
