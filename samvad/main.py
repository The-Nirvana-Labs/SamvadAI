from pathlib import Path

from .core import pipeline, presets

from .loader import LoadTextStep, LoadTextContext
from .remove_stopwords import RemoveStopwordsStep
from .lemmetize_text import LemmatizeTextEcwsStep
from .ner import StepEcwl as NerEcwlStep, CastNEROutputToText


def main():
    entry_context = LoadTextContext("LICENSE", Path("."))

    execution_pipeline = pipeline.Pipeline(entry_context)

    execution_pipeline.load_steps([
        LoadTextStep(),
        presets.CastStringToStringGeneral(),
        RemoveStopwordsStep(),
        presets.CastStringToStringGeneral(),
        LemmatizeTextEcwsStep(),
        presets.CastStringToStringGeneral(),
        NerEcwlStep(),
        CastNEROutputToText(),

    ])

    execution_pipeline.start()

    outputs = execution_pipeline.get_outputs()

    return outputs
