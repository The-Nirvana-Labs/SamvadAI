# from nlp_preprocessor_engine.remove_stopwords.__advanced__ import remove_stopwords
from nlp_preprocessor_engine.lemmatize_text.__intermediate__ import lemmatize_text
import os


# def count_tokens(string):
#     count = len(string)
#     result = count / 4
#     return result
#
#
# def traverse_and_print_txt_files(folder_path):
#     average = 0
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.txt'):
#             with open(os.path.join(folder_path, file_name), 'r') as f:
#                 file_text = f.read()
#                 actual_length = count_tokens(str(file_text))
#                 lemma_text = count_tokens(lemmatize_text(remove_stopwords(str(file_text))))
#     return lemma_text
#
#
# print("average: ", traverse_and_print_txt_files("../data"))


string_ = "she is eating the cake and he is running on the road as a biker"
print(lemmatize_text(string_))
