from nlp_preprocessor_engine.remove_stopwords.__basic__ import remove_stopwords as stop_words_basic
from nlp_preprocessor_engine.remove_stopwords.__intermediate__ import remove_stopwords as stop_words_intermediate
from nlp_preprocessor_engine.remove_stopwords.__advanced__ import remove_stopwords as stop_words_advanced
from nlp_preprocessor_engine.lemmatize_text.__basic__ import lemmatize_text as lemmatize_basic
from nlp_preprocessor_engine.lemmatize_text.__intermediate__ import lemmatize_text as lemmatize_intermediate
import os


def count_tokens(string):
    count = len(string)
    result = count / 4
    return result


def traverse_and_print_txt_files(folder_path):
    average = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                file_text = f.read()
                actual_length = count_tokens(str(file_text))
                stopwords_text_length = count_tokens(remove_stopwords(str(file_text)))
                average = (actual_length - stopwords_text_length) / actual_length * 100 + average
                print("Average performance for Stop Words: ", average / len(os.listdir(folder_path)), ".")
                average = 0

                lemma_text_length = count_tokens(lemmatize_text(remove_stopwords(str(file_text))))
                average = (actual_length - lemma_text_length) / actual_length * 100 + average
                print("Average performance for Lemmatized Words: ", average / len(os.listdir(folder_path)), ".")
                average = 0

    return average / len(os.listdir(folder_path))


print("average: ", traverse_and_print_txt_files("../data"))
