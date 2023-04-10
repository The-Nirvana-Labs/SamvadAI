from pathlib import Path

from .core import pipeline, presets

from .loader import LoadTextStep, LoadTextContext
from .stopwords import RemoveStopwordsStep
from .lemmatizer import LemmatizeTextEcwsStep
from .ner import StepEcwl as NerEcwlStep
from .chunks import ExtractChunksStep, CastChunksToText
from .embedding import EmbeddingAmlvStep, CastEmbeddingTensorToNdArray, CastEmbeddingNdArrayToText,\
                       EmbeddingToIndexesAnnoyStep
from .summarization import SummarizeTextPegasusStep


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
        presets.CastStringToStringGeneral(),
        ExtractChunksStep(),
        CastChunksToText(),
        EmbeddingAmlvStep(),
        CastEmbeddingTensorToNdArray(),
        EmbeddingToIndexesAnnoyStep(),
        CastEmbeddingNdArrayToText(),
        SummarizeTextPegasusStep(),
        presets.CastStringToStringGeneral(),
        # Final Extract Sentences Step
    ])

    execution_pipeline.start()

    outputs = execution_pipeline.get_outputs()

    return outputs
