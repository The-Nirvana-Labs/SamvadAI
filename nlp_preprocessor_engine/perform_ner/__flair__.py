from flair.data import Sentence
from flair.models import SequenceTagger


def flair_ner(text: str, model_name: str = 'ner') -> dict:
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
    return {'entities': entities}
