from samvad.core import api, presets

from ._base import normalizer


class StepNormalizer(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        output_text: str = normalizer(context.input_text)
        self.set_output(presets.StringGeneralOutput(output_text=output_text))

