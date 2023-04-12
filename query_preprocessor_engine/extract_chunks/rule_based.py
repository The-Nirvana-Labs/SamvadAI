import spacy
import nltk

nlp = spacy.load('en_core_web_sm')
nltk.download('averaged_perceptron_tagger')


# Define a function to perform chunking using a rule-based method
def chunking(sentence):
    """
    Perform chunking using a rule-based method.

    Args:
        sentence (str): Input sentence to be chunked.

    Returns:
        chunks (list): List of extracted chunks.
    """
    grammar = r"""
        NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
        PP: {<IN><NP>}
    """
    chunk_parser = nltk.RegexpParser(grammar)
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    tree = chunk_parser.parse(pos_tags)
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() in ['NP', 'PP']:
            chunks.append(' '.join([token for token, pos in subtree.leaves()]))
    return chunks
