import langchain
from math import ceil

def extract_sentences(text, summarizer=None):
    """
    Extracts a number of sentences from the input text using the specified summarization model, based on the length of the text.

    Args:
    - text (str): The input text to extract sentences from.
    - summarizer: (object): An optional summarization model. Default is langchain.Summarizer('english').

    Returns:
    - list: A list of top sentences extracted from the input text, as determined by the specified summarization model.
    """
    if summarizer is None:
        summarizer = langchain.Summarizer('english')
    text_length = len(text)
    num_sentences = ceil(text_length / 500) + 1
    sentences = summarizer.extract_sentences(text, num_sentences=num_sentences)
    return sentences

text = 'The sun shone brightly overhead as the birds chirped and the leaves rustled in the gentle breeze. Sarah walked ' \
      'along the path, taking in the beauty of the surrounding nature. She felt at peace, her worries melting away ' \
      'with each step she took. It was moments like these that made her grateful to be alive and able to enjoy the ' \
      'simple pleasures of life. '
print(extract_sentences(text))
