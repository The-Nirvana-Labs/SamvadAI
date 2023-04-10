import requests
import streamlit as st


def main():
    st.title("String Modifier")
    text_input = st.text_input("Enter the query:")
    if st.button("Modify"):
        response = requests.post("http://localhost:8000/modify_string/", json={"text": text_input})
        modified_text = response.text
        st.write("Modified string:", modified_text)


if __name__ == "__main__":
    main()
