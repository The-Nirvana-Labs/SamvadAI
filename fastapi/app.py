from fastapi import FastAPI

app = FastAPI()


def function_a(text: str) -> str:
    """Example function to modify the input string."""
    return text.upper()


def function_b(text: str) -> str:
    """Example function to modify the input string."""
    return text.replace(" ", "_")


def function_c(text: str) -> str:
    """Example function to modify the input string."""
    return text[::-1]


@app.post("/preprocessor/")
def modify_string(text: str) -> str:
    """API endpoint to modify the input string using function_a, function_b, and function_c."""
    text = function_a(text)
    text = function_b(text)
    text = function_c(text)
    return text
