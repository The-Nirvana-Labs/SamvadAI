import spacy
import stanfordnlp
import openai
import faiss
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import langchain


class NLPPreprocess:
    def __init__(self, text_file):
        self.text_file = text_file
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.keyword_extractor = langchain.KeywordExtractor('english')
        self.spacy_nlp = spacy.load('en_core_web_lg')
        self.stanford_nlp = stanfordnlp.Pipeline()
        self.summarizer = langchain.Summarizer('english')
        self.openai_api_key = "Enter your API key here"
        self.faiss_index = None
        self.faiss_vectors = None

    def load_text(self):
        """
        Opens and reads a text file and returns its contents as a string.

        Returns:
        - text (str): The contents of the text file as a string, with line breaks replaced by spaces.
        """
        with open(self.text_file, 'r') as file:
            text = file.read().replace('\n', ' ')
        return text

    def remove_stopwords(self, text):
        """
        Removes stopwords from the input text and returns the filtered text as a string.

        Args:
        - text (str): The input text to filter.

        Returns:
        - str: The filtered text as a string, with stopwords removed and all words converted to lowercase.
        """
        tokens = word_tokenize(text)
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    def lemmatize_text(self, text):
        """
        Lemmatizes the words in the input text and returns the lemmatized text as a string.

        Args:
        - text (str): The input text to lemmatize.

        Returns:
        - str: The lemmatized text as a string, with all words reduced to their base form.
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def perform_ner(self, text):
        """
        Performs Named Entity Recognition (NER) on the input text and returns a list of named entities found.

        Args:
        - text (str): The input text to perform NER on.

        Returns:
        - list: A list of named entities found in the input text, represented as tuples of the form (entity_text, entity_type).
        """
        doc = self.stanford_nlp(text)
        entities = []
        for sentence in doc.sentences:
            for entity in sentence.ents:
                entities.append((entity.text, entity.type))
            return entities

    def extract_chunks(self, text):
        """
        Extracts noun chunks from the input text and returns them as a list of strings.

        Args:
        - text (str): The input text to extract noun chunks from.

        Returns:
        - list: A list of noun chunks found in the input text, represented as strings.
        """
        doc = self.spacy_nlp(text)
        chunks = []
        for chunk in doc.noun_chunks:
            chunks.append(chunk.text)
        return chunks

    def perform_embedding(self, text):
        openai.api_key = self.oopenai_api_key
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=text,
            temperature=0.5,
            max_tokens=50,
            n=1,
            stop=None,
        )
        return response.choices[0].text

    def build_faiss_index(self, text):
        """
        Builds an FAISS index for the input text using OpenAI's text-davinci-002 embedding model.

        Args:
        - text (str): The input text to build the FAISS index for.

        Returns:
        - None
        """
        openai.api_key = self.openai_api_key
        embedding = openai.Embedding("text-davinci-002")
        vectors = []
        for chunk in self.extract_chunks(text):
            vector = embedding.embed(chunk)
            vectors.append(vector)
        vectors = faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.faiss_index = index
        self.faiss_vectors = vectors

    def search_similar_chunks(self, query, k=3):
        """
        Searches for the most similar chunks to the input query using an FAISS index built with OpenAI's text-davinci-002 embedding model.

        Args:
        - query (str): The query string to search for similar chunks.
        - k (int, optional): The number of similar chunks to return. Default is 3.

        Returns:
        - list: A list of k similar chunks found in the input text, represented as strings.
        """
        embedding = openai.Embedding("text-davinci-002")
        query_vector = embedding.embed(query)
        query_vector = faiss.normalize_L2(query_vector)
        distances, indices = self.faiss_index.search(query_vector, k)
        similar_chunks = []
        for index in indices[0]:
            similar_chunks.append(self.extract_chunks(self.load_text())[index])
        return similar_chunks

    def summarize_text(self, text):
        """
        Summarizes the input text using the summarization model specified in the instance of the class.

        Args:
        - text (str): The input text to be summarized.

        Returns:
        - str: A summary of the input text, generated by the specified summarization model.
        """
        summarizer = self.summarizer
        summary = summarizer.summarize(text, num_sentences=5)
        return summary

    def extract_keywords(self, text):
        """
        Extracts keywords from the input text using the specified keyword extraction model.

        Args:
        - text (str): The input text to extract keywords from.

        Returns:
        - list: A list of the top keywords extracted from the input text, as determined by the specified keyword extraction model.
        """
        keyword_extractor = self.keyword_extractor
        keywords = keyword_extractor.extract(text, num_keywords=5)
        return keywords

    def extract_sentences(self, text, num_sentences=5):
        """
        Extracts a specified number of sentences from the input text using the specified summarization model.

        Args:
        - text (str): The input text to extract sentences from.
        - num_sentences (int): The number of sentences to extract. Default is 5.

        Returns:
        - list: A list of the specified number of top sentences extracted from the input text, as determined by the specified summarization model.
        """
        summarizer = self.summarizer
        sentences = summarizer.extract_sentences(text, num_sentences=num_sentences)
        return sentences

    def preprocess_text(self):
        """
        Preprocesses the text using a series of NLP techniques, including stopword removal, lemmatization, summarization, keyword extraction, and sentence extraction. Additionally, an FAISS index is built using the top sentences extracted from the text.

        Returns:
        - tuple: A tuple consisting of the preprocessed text, a list of the top 5 extracted keywords from the text, and a list of the top 5 extracted sentences from the text, as determined by the specified summarization model.
        """
        text = self.load_text()
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        text = self.summarize_text(text)
        keywords = self.extract_keywords(text)
        sentences = self.extract_sentences(text)
        text = ' '.join(sentences)
        self.build_faiss_index(text)
        return text, keywords, sentences
