from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer,\
                         PegasusForConditionalGeneration, PegasusTokenizer

# Load the ProphetNet model and tokenizer
model_prophetnet_name = 'microsoft/prophetnet-large-uncased'
tokenizer_prophetnet = ProphetNetTokenizer.from_pretrained(model_prophetnet_name)
model_prophetnet = ProphetNetForConditionalGeneration.from_pretrained(model_prophetnet_name)

# Load the Pegasus model and tokenizer
model_name_pegasus = 'google/pegasus-xsum'
tokenizer_pegasus = PegasusTokenizer.from_pretrained(model_name_pegasus)
model_pegasus = PegasusForConditionalGeneration.from_pretrained(model_name_pegasus)


def summarize_text_prophet(input_text: str) -> str:
    """
    Summarizes input text using ProphetNet model

    Args:
    input_text (str): The text to be summarized

    Returns:
    str: The summary of the input text
    """

    # Tokenize and encode the input text
    inputs = tokenizer_prophetnet(input_text, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')

    # Generate the summary
    summary_ids = model_prophetnet.generate(inputs['input_ids'], num_beams=4, length_penalty=2.0, max_length=142,
                                            min_length=56, early_stopping=True)

    summary = tokenizer_prophetnet.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def summarize_text_pegasus(input_text: str) -> str:
    """
    Summarizes input text using Pegasus model

    Args:
    input_text (str): The text to be summarized

    Returns:
    str: The summary of the input text
    """

    # Tokenize and encode the input text
    inputs = tokenizer_pegasus(input_text, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')

    # Generate the summary
    summary_ids = model_pegasus.generate(inputs['input_ids'], num_beams=4, length_penalty=2.0, max_length=142,
                                         min_length=56, early_stopping=True)

    summary = tokenizer_pegasus.decode(summary_ids[0], skip_special_tokens=True)

    return summary
