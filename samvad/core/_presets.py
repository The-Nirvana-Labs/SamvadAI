from dataclasses import dataclass

from ._api import Output, Context
from ._casting import CastOutputToContext


# Step Objects Presets
@dataclass
class StringGeneralContext(Context):
    input_text: str


@dataclass
class StringGeneralOutput(Output):
    output_text: str


# Casting Presets
class CastStringToStringGeneral(CastOutputToContext):
    def cast(self, _: Context, output: StringGeneralOutput) -> Context:
        next_context: StringGeneralContext = StringGeneralContext(output.output_text)
        return next_context
