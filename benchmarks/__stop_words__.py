from nlp_preprocessor_engine.remove_stopwords.__basic__ import remove_stopwords as basic
from nlp_preprocessor_engine.remove_stopwords.__intermediate__ import remove_stopwords as intermediate
from nlp_preprocessor_engine.remove_stopwords.__advanced__ import remove_stopwords as advanced
from __count_tokens__ import count_tokens
import os


def traverse_and_print_txt_files(folder_path, function, function_name):
    average = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                file_text = f.read()
                actual_length = count_tokens(str(file_text))
                stopwords_text_length = count_tokens(function(str(file_text)))
                average = (actual_length - stopwords_text_length) / actual_length * 100 + average
    return function_name, average / len(os.listdir(folder_path))


print(traverse_and_print_txt_files("../data", basic, "Basic"))
print(traverse_and_print_txt_files("../data", intermediate, "Intermediate"))
print(traverse_and_print_txt_files("../data", advanced, "Advanced"))
