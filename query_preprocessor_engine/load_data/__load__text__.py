def load_text(text_file):
    """
    Opens and reads a text file and returns its contents as a string.

    Args:
    - text_file (str): The path to the text file.

    Returns:
    - text (str): The contents of the text file as a string, with line breaks replaced by spaces.
    """
    with open(text_file, 'r') as file:
        text = file.read().replace('\n', ' ')
    return text
