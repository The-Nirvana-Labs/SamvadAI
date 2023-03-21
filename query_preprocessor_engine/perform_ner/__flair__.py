from flair.models import SequenceTagger
from flair.data import Sentence


def perform_ner(text):
    """
    Performs named entity recognition on the input text using Flair and wraps the identified entities with their respective XML tags.

    Args:
    - text (str): The input text to perform named entity recognition on.

    Returns:
    - str: The input text with identified entities wrapped in their respective XML tags.

    Example:
    >>> text = "My favorite book is To Kill a Mockingbird by Harper Lee."
    >>> perform_ner(text)
    'My favorite book is [BOOK]To Kill a Mockingbird[/BOOK] by [PERSON]Harper Lee[/PERSON].'
    """

    # load the named entity recognition model
    tagger = SequenceTagger.load('ner')

    # create a sentence object from the input text
    sentence = Sentence(text)

    # run named entity recognition on the sentence
    tagger.predict(sentence)

    # loop through the identified entities and wrap them in their respective XML tags
    for entity in sentence.get_spans('ner'):
        text = text.replace(entity.text, f"[{entity.tag.upper()}]{entity.text}[/{entity.tag.upper()}]")

    return text
