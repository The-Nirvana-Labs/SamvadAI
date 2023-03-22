import streamlit as st
import transformers
import re
import random

# Load the GPT-2 tokenizer
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

# Define a list of bright colors
BRIGHT_COLORS = [
    "#FFA07A",  # orange
    "#FFF68F",  # yellow
    "#ADD8E6",  # blue
    "#D8BFD8",  # purple
    "#98FB98",  # green
    "#FFB6C1",  # pink
]

# Define regular expressions to match different token types
RE_KEYWORD = r"^(if|else|for|while|return)$"
RE_NUMBER = r"^\d+(\.\d+)?$"


# Define a function to tokenize a string and annotate it with random colors
def tokenize_string(text):
    encoded_text = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded_text)

    # Create a list of tuples containing the token text and a random color
    annotated_tokens = []
    for token in tokens:
        if bool(re.match(RE_KEYWORD, token)):
            token_type = "keyword"
        elif bool(re.match(RE_NUMBER, token)):
            token_type = "number"
        elif token in tokenizer.all_special_tokens:
            token_type = "punctuation"
        else:
            token_type = "text"
        color = random.choice(BRIGHT_COLORS)
        annotated_tokens.append((token.replace("Ġ", ""), color))

    # Create a single line string with colored tokens
    output = ""
    for token, color in annotated_tokens:
        output += f'<span style="background-color:{color}; padding: 3px 5px; border-radius: 5px; color: #000000;">{token}</span> '

    return output


# Set page configuration
st.set_page_config(
    page_title="GPT-2 Tokenizer",
    page_icon=":pencil2:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define the app layout
col1, col2 = st.columns((2, 3))
with col1:
    st.header("Input Text")
    text_input = st.text_area("", height=200)
with col2:
    st.header("Tokenized Text")
    if text_input:
        output = tokenize_string(text_input)
        st.markdown(output, unsafe_allow_html=True)
