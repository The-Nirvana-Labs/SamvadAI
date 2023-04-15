import torch
from transformers import BertTokenizer, BertForTokenClassification


def chunk_document(document, max_chunk_length=256):
    """
    Chunk a document into smaller chunks using a pre-trained BERT model.

    Args:
        document (str): Input document to be chunked.
        max_chunk_length (int): Maximum length of each chunk in number of tokens.

    Returns:
        chunks (list): List of extracted chunks.
    """
    # Load a pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # Tokenize the input document
    tokens = tokenizer.tokenize(document)

    # Divide the tokens into chunks of maximum length
    chunks = [tokens[i:i + max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]

    # Convert the chunks to input features for the BERT model
    input_ids = []
    attention_masks = []
    for chunk in chunks:
        encoded_dict = tokenizer.encode_plus(
            chunk,  # Chunk to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_chunk_length,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Load a pre-trained BERT model
    model = BertForTokenClassification.from_pretrained('bert-base-cased')

    # Predict the token labels using the trained BERT model
    model.eval()
    with torch.no_grad():
        labels = []
        for input_id, attention_mask in zip(input_ids, attention_masks):
            outputs = model(input_id, attention_mask=attention_mask)
            logits = outputs[0]
            predicted_labels = torch.argmax(logits, dim=2)
            labels.append(predicted_labels)

    # Convert the predicted labels to chunk boundaries
    boundaries = []
    for label in labels:
        boundary = torch.where(label[0] == 1)[0]
        if len(boundary) > 0:
            boundaries.append(boundary[0])
        else:
            boundaries.append(len(label[0]))

    # Extract the chunks from the original document
    chunks = [document[boundaries[i - 1]:boundaries[i]] for i in range(1, len(boundaries))]

    return chunks
