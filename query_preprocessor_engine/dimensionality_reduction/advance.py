import torch
from transformers import AutoTokenizer, AutoModel
from umap import UMAP


def dimensionality_reduction(text_data, model_name="bert-base-uncased", num_components=2):
    """
    Use UMAP to reduce the dimensionality of text data.

    Args:
        text_data (list): A list of strings, where each string represents a document.
        model_name (str): The name of the pre-trained language model to use for encoding the text (default="bert-base-uncased").
        num_components (int): The number of dimensions to reduce the data to (default=2).

    Returns:
        numpy.ndarray: An array of shape (len(text_data), num_components) containing the reduced
        data.

    Example:
        text_data = ["This is the first document.", "This is the second document.", "And this is the third one."]
        reduced_data = reduce_text_dimensionality(text_data, model_name="bert-base-uncased", num_components=2)
    """
    # Load the pre-trained language model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Encode the text data using the language model
    encoded_data = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_data)

    # Use UMAP to reduce the dimensionality of the encoded data
    umap = UMAP(n_components=num_components)
    reduced_data = umap.fit_transform(model_output.last_hidden_state)

    return reduced_data
