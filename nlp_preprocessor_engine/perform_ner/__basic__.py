import stanfordnlp


def perform_ner(text):
    """
    Performs Named Entity Recognition (NER) on the input text and returns a list of named entities found.

    Args:
    - text (str): The input text to perform NER on.

    Returns:
    - list: A list of named entities found in the input text, represented as tuples of the form (entity_text, entity_type).
    """
    nlp = stanfordnlp.Pipeline()
    doc = nlp(text)
    entities = []
    for sentence in doc.sentences:
        for entity in sentence.ents:
            entities.append((entity.text, entity.type))
    return entities
