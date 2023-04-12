from transformers import AutoTokenizer, AutoModelForTokenClassification


def extract_chunks(document: str, max_chunk_size: int = 256) -> list:
    """
    Extracts chunks from a large document using a pre-trained model for named entity recognition.

    Args:
        document (str): The large document to be divided into chunks.
        max_chunk_size (int, optional): The maximum size of each chunk. Defaults to 512.

    Returns:
        list: A list of chunks where each chunk is a string.
    """
    # Load pre-trained model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Tokenize the document and get the tokens' labels
    tokens = tokenizer.tokenize(document)
    labels = []
    for i in range(0, len(tokens), max_chunk_size):
        chunk_tokens = tokens[i:i + max_chunk_size]
        inputs = tokenizer.encode(' '.join(chunk_tokens), return_tensors='pt')
        outputs = model(inputs).logits.argmax(-1)
        chunk_labels = [model.config.id2label[label_id] for label_id in outputs[0].tolist()]
        labels.extend(chunk_labels)

    # Extract chunks based on the labels
    chunks = []
    current_chunk = ""
    current_label = ""
    for token, label in zip(tokens, labels):
        if current_label != label and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += " " + token
        current_label = label
    chunks.append(current_chunk.strip())

    return chunks
