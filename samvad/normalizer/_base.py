import re


def normalizer(text):
    """
    Preprocesses text by converting it to lowercase and removing all punctuation marks.

    Parameters:
        text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text with all punctuation marks removed and converted to lowercase.
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove all punctuation marks
    text = re.sub(r'[^\w\s]', '', text)

    return text
