import re


def chunk_document(document):
    """
    Chunk a document into 256-sized chunks using a rule-based approach.

    Args:
        document (str): Input document to be chunked.

    Returns:
        chunks (list): List of extracted chunks.
    """
    # Define regular expression patterns for noun phrases and verb phrases
    noun_phrase_pattern = re.compile(r'\b(NN|NNS|NNP|NNPS)\b')
    verb_phrase_pattern = re.compile(r'\b(VB|VBD|VBG|VBN|VBP|VBZ)\b')

    # Split the input document into sentences
    sentences = re.split(r'\. |\? |\! |\n', document)

    # Initialize an empty list to store the extracted chunks
    chunks = []

    # Loop over each sentence in the document
    for sentence in sentences:
        # Split the sentence into words and part-of-speech tags
        words_and_tags = [word_and_tag.split('_') for word_and_tag in sentence.split()]

        # Initialize an empty list to store the current chunk
        current_chunk = []

        # Loop over each word and its part-of-speech tag
        for word, tag in words_and_tags:
            # If the word is a noun or verb, add it to the current chunk
            if noun_phrase_pattern.match(tag) or verb_phrase_pattern.match(tag):
                current_chunk.append(word)

            # If the current chunk is longer than 256 words, add it to the list of chunks
            if len(current_chunk) >= 256:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        # If the current chunk is not empty, add it to the list of chunks
        if current_chunk:
            chunks.append(' '.join(current_chunk))

    return chunks
