import spacy
from flair.models import SequenceTagger
from flair.data import Sentence

nlp = spacy.load("en_core_web_lg")


def perform_ner_ecwl(text) -> str:
    """
    Performs Named Entity Recognition (NER) on the input text and returns a modified version of the input text
    with named entities wrapped in XML tags of their respective entity types.

    Args:
    - text (str): The input text to perform NER on.

    Returns:
    - str: The input text with named entities wrapped in XML tags of their respective entity types.
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    xml_tags = {}
    for entity, entity_type in entities:
        if entity_type not in xml_tags:
            xml_tags[entity_type] = f"{entity_type.upper()}"
    for entity, entity_type in entities:
        xml_tag = xml_tags[entity_type]
        text = text.replace(entity, f"[{xml_tag}]{entity}[/{xml_tag}]")
    return text


def perform_ner_flair(text) -> str:
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
