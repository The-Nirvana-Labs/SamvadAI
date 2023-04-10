import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('wordnet')


def lemmatize_text_ecws(text):
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


def lemmatize_text_wordnet(text):
    """
    Lemmatizes the words in the input text and returns the lemmatized text as a string.

    Args:
    - text (str): The input text to lemmatize.

    Returns:
    - str: The lemmatized text as a string, with all words reduced to their base form.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)
