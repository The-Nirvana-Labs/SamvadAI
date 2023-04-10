from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def dimensionality_reduction(text_data, num_components=2):
    """
    Use PCA to reduce the dimensionality of text data.

    Args:
        text_data (list): A list of strings, where each string represents a document.
        num_components (int): The number of dimensions to reduce the data to (default=2).

    Returns:
        numpy.ndarray: An array of shape (len(text_data), num_components) containing the reduced
        data.

    Example:
        text_data = ["This is the first document.", "This is the second document.", "And this is the third one."]
        reduced_data = reduce_text_dimensionality(text_data, num_components=2)
    """
    # Convert the text data into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(text_data)

    # Use PCA to reduce the dimensionality of the matrix
    pca = PCA(n_components=num_components)
    reduced_matrix = pca.fit_transform(tf_idf_matrix.toarray())

    return reduced_matrix
