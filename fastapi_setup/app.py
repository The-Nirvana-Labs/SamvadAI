from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class TextPayload(BaseModel):
    text: str


def function_a(text: str) -> str:
    """Example function to modify the input string."""
    return text.upper()


def function_b(text: str) -> str:
    """Example function to modify the input string."""
    return text.replace(" ", "_")


@app.post("/preprocessor/")
def preprocessor(payload: TextPayload) -> str:
    """API endpoint to modify the input string using function_a, function_b, and function_c."""
    text = function_a(payload.text)
    text = function_b(text)
    return text
