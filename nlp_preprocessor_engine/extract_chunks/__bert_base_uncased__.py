import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def extract_chunks(text):
    """
    Extracts noun chunks from the input text using a pre-trained transformer model and returns them as a list of strings.

    Args:
    - text (str): The input text to extract noun chunks from.

    Returns:
    - list: A list of noun chunks found in the input text, represented as strings.
    """
    # Load a pre-trained tokenizer and transformer model from the Hugging Face Transformers library
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Tokenize the input text using the tokenizer
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)

    # Process the input text using the transformer model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the noun chunks from the processed text
    labels = torch.argmax(outputs.logits, dim=2)
    chunks = []
    current_chunk = ''
    for i, token_label in enumerate(labels[0]):
        token = tokenizer.decode([inputs['input_ids'][0][i]])
        if token_label == 1:
            current_chunk += ' ' + token
        elif token_label == 2:
            current_chunk += token
            chunks.append(current_chunk.strip())
            current_chunk = ''

    # Return the list of noun chunks found in the text
    return chunks
