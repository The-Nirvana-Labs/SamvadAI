import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")


def lemmatize_text(text):
    """
    Lemmatizes the words in the input text using a pre-trained language model and returns the lemmatized text as a string.

    Args:
    - text (str): The input text to lemmatize.

    Returns:
    - str: The lemmatized text as a string, with all words reduced to their base form.
    """
    text = "[CLS] " + text + " [SEP]"
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')

    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    output = model(input_ids)[0]
    mask_token_logits = output[0, mask_token_index, :]
    mask_token_logits = torch.nn.functional.softmax(mask_token_logits, dim=1)
    predicted_token_indexes = torch.argmax(mask_token_logits, dim=1)
    predicted_tokens = [tokenizer.convert_ids_to_tokens(int(idx)) for idx in predicted_token_indexes]

    lemmatized_tokens = [token.split('#')[0] for token in predicted_tokens]
    lemmatized_text = ' '.join(lemmatized_tokens[1:-1])
    return lemmatized_text
