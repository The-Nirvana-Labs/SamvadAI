import spacy


def perform_ner(text):
    """
    Performs Named Entity Recognition (NER) on the input text and returns a modified version of the input text
    with named entities wrapped in XML tags of their respective entity types.

    Args:
    - text (str): The input text to perform NER on.

    Returns:
    - str: The input text with named entities wrapped in XML tags of their respective entity types.
    """
    nlp = spacy.load("en_core_web_lg")
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
