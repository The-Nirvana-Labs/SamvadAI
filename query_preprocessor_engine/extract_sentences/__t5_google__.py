import transformers


def extract_sentences(text, num_sentences):
    """
    Extracts a number of sentences from the input text using the T5 model.

    Args:
    - text (str): The input text to extract sentences from.
    - num_sentences (int): The desired number of sentences to extract.

    Returns:
    - list: A list of top sentences extracted from the input text, as determined by the T5 model.
    """
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base')
    input_text = 'summarize: ' + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(input_ids, num_beams=4, max_length=num_sentences, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    sentences = summary.split('. ')
    return sentences
