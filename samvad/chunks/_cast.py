from samvad.core import casting, api, presets

from ._step import ExtractChunksOutput


class CastChunksToText(casting.CastOutputToContext):
    def cast(self, _: presets.StringGeneralContext, output: ExtractChunksOutput) -> presets.StringGeneralContext:
        chunks = output.chunks
        text = " ".join(chunks)
        return presets.StringGeneralContext(input_text=text)
