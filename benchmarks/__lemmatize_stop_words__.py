from nlp_preprocessor_engine.remove_stopwords.__basic__ import remove_stopwords as stop_words_basic
from nlp_preprocessor_engine.remove_stopwords.__intermediate__ import remove_stopwords as stop_words_intermediate
from nlp_preprocessor_engine.remove_stopwords.__advanced__ import remove_stopwords as stop_words_advanced
from nlp_preprocessor_engine.lemmatize_text.__basic__ import lemmatize_text as lemmatize_basic
from nlp_preprocessor_engine.lemmatize_text.__intermediate__ import lemmatize_text as lemmatize_intermediate
from __count_tokens__ import count_tokens
import os


def traverse_and_print_txt_files(folder_path, function_stopwords, function_lemmatize, function_name):
    average = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                file_text = f.read()
                actual_length = count_tokens(str(file_text))
                stopwords_text = function_stopwords(str(file_text))
                lemmatize_text_length = count_tokens(function_lemmatize(stopwords_text))
                average = (actual_length - lemmatize_text_length) / actual_length * 100 + average
    return function_name, average / len(os.listdir(folder_path))


print("Lemmatize + Stop Words")
print(traverse_and_print_txt_files("../data", stop_words_basic, lemmatize_basic, "Basic + Basic"))
print(traverse_and_print_txt_files("../data", stop_words_intermediate, lemmatize_basic, "Basic + Intermediate"))
print(traverse_and_print_txt_files("../data", stop_words_advanced, lemmatize_basic, "Basic + Advanced"))
print(traverse_and_print_txt_files("../data", stop_words_basic, lemmatize_intermediate, "Intermediate + Basic"))
print(traverse_and_print_txt_files("../data", stop_words_intermediate, lemmatize_intermediate, "Intermediate + "
                                                                                               "Intermediate"))
print(traverse_and_print_txt_files("../data", stop_words_advanced, lemmatize_intermediate, "Intermediate + Advanced"))
