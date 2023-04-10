from query_preprocessor_engine.remove_stopwords.word_tokenize.__word_tokenize__ import remove_stopwords as __word_tokenize__
from query_preprocessor_engine.remove_stopwords.en_core_web_sm_advanced.__en_core_web_sm__ import remove_stopwords as __en_core_web_sm_advance__
from query_preprocessor_engine.remove_stopwords.en_core_web_sm.__en_core_web_sm__ import remove_stopwords as __en_core_web_sm__
from query_preprocessor_engine.lemmatize_text.__wordnet__ import lemmatize_text as __wordnet__
from query_preprocessor_engine.lemmatize_text.__en_core_web_sm__ import lemmatize_text as __en_core_web_sm_wordnet__
from query_preprocessor_engine.extract_chunks.__en_core_web_sm__ import extract_chunks
from __count_len_list_nounchunks__ import concatenate_strings
import os
from __count_tokens__ import count_tokens


def traverse_and_print_txt_files(folder_path, fun_stopwords, fun_lemmatize, fun_extract_chunks):
    average = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                file_text = f.read()
                actual_length = count_tokens(str(file_text))

                lemma_text_length = count_tokens(concatenate_strings(fun_stopwords(str(file_text))))
                # lemma_text_length = count_tokens(concatenate_strings(fun_lemmatize(fun_stopwords(str(file_text)))))
                # lemma_text_length = count_tokens(concatenate_strings(fun_extract_chunks(fun_lemmatize(fun_stopwords(str(file_text))))))
                average = (actual_length - lemma_text_length) / actual_length * 100 + average
    return average / len(os.listdir(folder_path))


print("average: ", traverse_and_print_txt_files("../data", __word_tokenize__,__wordnet__,extract_chunks))
print("average: ", traverse_and_print_txt_files("../data", __en_core_web_sm__,__wordnet__,extract_chunks))
print("average: ", traverse_and_print_txt_files("../data", __en_core_web_sm_advance__,__wordnet__,extract_chunks))
print("average: ", traverse_and_print_txt_files("../data", __word_tokenize__,__en_core_web_sm_wordnet__,extract_chunks))
print("average: ", traverse_and_print_txt_files("../data", __en_core_web_sm__,__en_core_web_sm_wordnet__,extract_chunks))
print("average: ", traverse_and_print_txt_files("../data", __en_core_web_sm_advance__,__en_core_web_sm_wordnet__,extract_chunks))
