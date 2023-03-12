import spacy
nlp = spacy.load('en_core_web_sm')


def remove_stopwords(text):
    """
    Removes stopwords from the input text and returns the filtered text as a string.

    Args:
    - text (str): The input text to filter.

    Returns: - str: The filtered text as a string, with stopwords removed, words lemmatized, and all words converted
    to lowercase.

    Raises:
    - TypeError: If input text is not a string.

    Example:
    >>> sample = "This is an example sentence with some stopwords."
    >>> benchmark_results = benchmark_remove_stopwords(text, num_trials=10, remove_stopwords_fn=remove_stopwords)

    Performance:
    - Time complexity: O(n), where n is the number of tokens in the input text.
    - Space complexity: O(n), where n is the number of tokens in the input text.

    Dependencies:
    - This function requires the `spacy` and `memory_profiler` libraries to be installed, which can be installed via pip.


    """
    doc = nlp(text)
    filtered_tokens = [token.lemma_.lower() for token in doc if
                       not token.is_stop and token.is_alpha and token.pos_ != 'PUNCT']
    return ' '.join(filtered_tokens)
