from samvad.core import casting, presets
from ._step import EmbeddingToIndexesOutput, EmbeddingOutput, EmbeddingToIndexesContext


class CastEmbeddingTensorToNdArray(casting.CastOutputToContext):
    def cast(self, context: presets.StringGeneralContext, output: EmbeddingOutput) -> EmbeddingToIndexesContext:
        pass


class CastEmbeddingNdArrayToText(casting.CastOutputToContext):
    def cast(self, context: EmbeddingToIndexesContext, output: EmbeddingToIndexesOutput) -> presets.StringGeneralContext:
        pass
