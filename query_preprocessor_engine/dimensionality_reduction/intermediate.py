import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


def dimensionality_reduction(text_data, num_components=2, min_count=5, window=5, sg=1):
    """
    Use t-SNE and Word2Vec to reduce the dimensionality of text data.

    Args:
        text_data (list): A list of strings, where each string represents a document.
        num_components (int): The number of dimensions to reduce the data to (default=2).
        min_count (int): The minimum frequency of a word in the corpus to be included in the Word2Vec model (default=5).
        window (int): The size of the window used for Word2Vec training (default=5).
        sg (int): The training algorithm used for Word2Vec: 1 for skip-gram, 0 for CBOW (default=1).

    Returns:
        numpy.ndarray: An array of shape (len(text_data), num_components) containing the reduced data.

    Example:
        text_data = ["This is the first document.", "This is the second document.", "And this is the third one."]
        reduced_data = reduce_text_dimensionality(text_data, num_components=2, min_count=5, window=5, sg=1)
    """
    # Use Word2Vec to extract features from the text data
    tokenized_data = [doc.lower().split() for doc in text_data]
    model = Word2Vec(tokenized_data, min_count=min_count, window=window, sg=sg)
    feature_vectors = np.array([model.wv[word] for doc in tokenized_data for word in doc])

    # Use t-SNE to reduce the dimensionality of the feature vectors
    tsne = TSNE(n_components=num_components)
    reduced_data = tsne.fit_transform(feature_vectors)

    return reduced_data
