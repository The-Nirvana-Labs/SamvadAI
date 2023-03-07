import spacy

nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    """
    Lemmatizes the words in the input text using spaCy and returns the lemmatized text as a string.

    Args:
    - text (str): The input text to lemmatize.

    Returns:
    - str: The lemmatized text as a string, with all words reduced to their base form.
    """
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)
