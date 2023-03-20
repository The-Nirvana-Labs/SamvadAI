from typing import List, Dict

from flair.data import Sentence
from flair.models import SequenceTagger
import spacy


def perform_ner_ecwl(text) -> List[Dict[str, str]]:
    """
    Performs Named Entity Recognition (NER) on the input text and returns a list of named entities found.

    Args:
    - text (str): The input text to perform NER on.

    Returns:
    - list: A list of named entities found in the input text, represented as tuples of the form (entity_text, entity_type).
    """
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    entities = []
    for entity in doc.ents:
        entities.append({
            'text': entity.text,
            'type': entity.label_
        })
    return entities


def perform_ner_flair(text: str, model_name: str = 'ner') -> List[Dict[str, str]]:
    """
    Perform Named Entity Recognition (NER) on input text using Flair.

    Args:
        text (str): Input text to perform NER on.
        model_name (str, optional): Name of the Flair NER model to use. Default is 'ner'.

    Returns:
        dict: A dictionary containing the recognized named entities and their types.

    """
    # Load the Flair NER model
    tagger = SequenceTagger.load(model_name)

    # Create a Flair Sentence object from the input text
    sentence = Sentence(text)

    # Run NER on the sentence
    tagger.predict(sentence)

    # Extract the named entities and their types from the sentence
    entities = []
    for entity in sentence.get_spans('ner'):
        entities.append({
            'text': entity.text,
            'type': entity.tag
        })

    # Return the named entities as a dictionary
    return entities
