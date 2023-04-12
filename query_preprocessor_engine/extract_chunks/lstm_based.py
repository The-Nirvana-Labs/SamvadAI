import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed


def chunk_document(document, max_chunk_length=128):
    """
    Chunk a document into smaller chunks using an LSTM network.

    Args:
        document (str): Input document to be chunked.
        max_chunk_length (int): Maximum length of each chunk in number of words.

    Returns:
        chunks (list): List of extracted chunks.
    """
    # Load a pre-trained tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([document])

    # Convert the input document to a sequence of word indices
    sequences = tokenizer.texts_to_sequences([document])[0]

    # Divide the sequences into chunks of maximum length
    chunks = [sequences[i:i+max_chunk_length] for i in range(0, len(sequences), max_chunk_length)]

    # Convert the chunks to padded sequences
    chunks = tf.keras.preprocessing.sequence.pad_sequences(chunks, maxlen=max_chunk_length)

    # Load a pre-trained LSTM network
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64),
        LSTM(units=64, return_sequences=True),
        TimeDistributed(Dense(units=1, activation='sigmoid'))
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('lstm_chunker.h5')

    # Predict the chunk labels using the trained LSTM network
    labels = model.predict(chunks)

    # Convert the predicted labels to chunk boundaries
    boundaries = np.where(labels > 0.5)[1]

    # Extract the chunks from the original document
    chunks = [document[boundaries[i-1]:boundaries[i]] for i in range(1, len(boundaries))]

    return chunks
