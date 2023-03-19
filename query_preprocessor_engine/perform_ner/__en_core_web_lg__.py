import spacy


def perform_ner(text):
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
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities
