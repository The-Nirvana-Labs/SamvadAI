from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer


def summarize_text(input_text):
    """
    Summarizes input text using ProphetNet model

    Args:
    input_text (str): The text to be summarized

    Returns:
    str: The summary of the input text
    """

    # Load the ProphetNet model and tokenizer
    model_name = 'microsoft/prophetnet-large-uncased'
    tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
    model = ProphetNetForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and encode the input text
    inputs = tokenizer(input_text, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, length_penalty=2.0, max_length=142, min_length=56,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
