from pathlib import Path

from .core import pipeline
from .loader import LoadTextStep, LoadTextContext, LoadTextOutput


def main():
    entry_context = LoadTextContext("LICENSE", Path("."))

    execution_pipeline = pipeline.Pipeline(entry_context)

    execution_pipeline.load_steps(LoadTextStep())

    execution_pipeline.start()

    outputs = execution_pipeline.get_outputs()

    return outputs
