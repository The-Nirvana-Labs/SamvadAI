from samvad.core import casting, presets

from ._step import PerformNEROutput


class CastNEROutputToText(casting.CastOutputToContext):
    def cast(self, context: presets.StringGeneralContext, output: PerformNEROutput) -> presets.StringGeneralContext:
        return presets.StringGeneralOutput("Not Implemented...")
