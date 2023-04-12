from transformers import pipeline


def transformer_based_chunking(sentence):
    """
    Perform chunking using a BERT-based model.

    Args:
        sentence (str): Input sentence to be chunked.

    Returns:
        chunks (list): List of extracted chunks.
    """
    model = pipeline('ner', model='bert-base-cased')
    chunks = []
    for entity in model(sentence):
        if entity['entity'] in ['B-NP', 'I-NP']:
            chunks.append(entity['word'])
    return chunks
