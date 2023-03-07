import stanfordnlp


def perform_ner(text, model_path=None, language='en', return_tokens=False):
    """
    Performs Named Entity Recognition (NER) on the input text and returns a list of named entities found.

    Args:
    - text (str): The input text to perform NER on.
    - model_path (str): Path to the Stanford NER model. If not provided, the default model for the specified language will be used.
    - language (str): The language of the input text. Defaults to 'en' (English).
    - return_tokens (bool): If True, returns the list of tokens with named entities tagged. Defaults to False.

    Returns:
    - list: A list of named entities found in the input text, represented as tuples of the form (entity_text, entity_type).
    """
    # Load the Stanford NER model
    if model_path is None:
        model_path = f'stanford-ner-2021-05-26/classifiers/english.muc.7class.distsim.crf.ser.gz'
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,ner', models_dir=model_path, lang=language)

    # Process the input text and extract named entities
    doc = nlp(text)
    entities = []
    for sentence in doc.sentences:
        for entity in sentence.ents:
            if return_tokens:
                tokens = [(token.text, token.ner) for token in entity.tokens]
                entities.append((entity.text, entity.type, tokens))
            else:
                entities.append((entity.text, entity.type))

    # Return the list of named entities
    return entities
