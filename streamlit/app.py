import requests
import streamlit as st

# Define the API endpoint URL
API_URL = "http://localhost:8000/preprocessor/"


# Define the function that makes the API call
def preprocess_text(text):
    # Prepare the request payload
    payload = {"text": text}
    # Make the API call
    response = requests.post(API_URL, json=payload)
    # Return the modified text if the response is successful
    if response.ok:
        return response.text
    # Raise an exception otherwise
    else:
        raise ValueError(response.text)


# Define the Streamlit app
def main():
    # Set the app title
    st.title("String Modifier")
    # Add a text input field for the user to enter the query
    text_input = st.text_input("Enter the query:")
    # Add a submit button to trigger the API call
    if st.button("Submit"):
        try:
            # Call the preprocess_text function to modify the input string
            modified_text = preprocess_text(text_input)
            # Display the modified string on the app
            st.write("Modified string:", modified_text)
        except ValueError as e:
            # Display the error message if the API call fails
            st.error("Failed to modify string. Error message: " + str(e))


if __name__ == "__main__":
    main()
