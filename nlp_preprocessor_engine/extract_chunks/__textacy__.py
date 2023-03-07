import spacy
import textacy


def extract_chunks(text):
    """
    Extracts noun chunks from the input text using a pre-trained deep learning model and returns them as a list of strings.

    Args:
    - text (str): The input text to extract noun chunks from.

    Returns:
    - list: A list of noun chunks found in the input text, represented as strings.
    """
    spacy_nlp = spacy.load('en_core_web_sm')
    doc = spacy_nlp(text)
    chunks = list(textacy.extract.noun_chunks(doc))
    return chunks
