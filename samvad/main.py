from pathlib import Path

from .core import pipeline
from .loader import LoadTextStep, LoadTextContext
from .remove_stopwords import CastRemoveStopwordContextFromLoadText, RemoveStopwordsStep


def main():
    entry_context = LoadTextContext("LICENSE", Path("."))

    execution_pipeline = pipeline.Pipeline(entry_context)

    execution_pipeline.load_steps([LoadTextStep(), CastRemoveStopwordContextFromLoadText(), RemoveStopwordsStep()])

    execution_pipeline.start()

    outputs = execution_pipeline.get_outputs()

    return outputs
