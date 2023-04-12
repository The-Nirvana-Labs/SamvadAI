import torch
from transformers import BertTokenizer, BertForTokenClassification


def chunk_document(document, max_chunk_length=128):
    """
    Chunk a document into smaller chunks using BERT.

    Args:
        document (str): Input document to be chunked.
        max_chunk_length (int): Maximum length of each chunk in number of words.

    Returns:
        chunks (list): List of extracted chunks.
    """
    # Load a pre-trained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForTokenClassification.from_pretrained('bert-base-cased')

    # Tokenize the input document
    tokens = tokenizer.tokenize(document)

    # Divide the tokens into chunks of maximum length
    chunks = [tokens[i:i+max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]

    # Convert the chunks to input features
    input_ids = []
    attention_masks = []
    for chunk in chunks:
        encoded_chunk = tokenizer.encode_plus(chunk, add_special_tokens=True, max_length=max_chunk_length,
                                              pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_chunk['input_ids'])
        attention_masks.append(encoded_chunk['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Predict the chunk labels using the trained BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

    # Convert the predicted labels to chunk boundaries
    boundaries = [i+1 for i, pred in enumerate(predictions[0]) if pred == 1]
    boundaries.insert(0, 0)
    boundaries.append(len(tokens))

    # Extract the chunks from the original document
    chunks = [document[boundaries[i-1]:boundaries[i]] for i in range(1, len(boundaries))]

    return chunks
