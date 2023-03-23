from typing import List

import spacy
spacy_nlp = spacy.load('en_core_web_sm')


def extract_chunks(text) -> List[str]:
    """
    Extracts noun chunks from the input text using a machine learning model and returns them as a list of strings.

    Args:
    - text (str): The input text to extract noun chunks from.

    Returns:
    - list: A list of noun chunks found in the input text, represented as strings.
    """
    doc = spacy_nlp(text)
    chunks = []
    for chunk in doc.noun_chunks:
        # Use a binary classifier to filter out non-noun phrases
        if chunk.root.pos_ == 'NOUN':
            chunks.append(chunk.text)
    return chunks
