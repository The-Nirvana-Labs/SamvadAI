from samvad.core import presets, api

from ._base import summarize_text_prophet, summarize_text_pegasus


class SummarizeTextProphetStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        output: str = summarize_text_prophet(context.input_text)
        self.set_output(output=presets.StringGeneralOutput(output_text=output))


class SummarizeTextPegasusStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        output: str = summarize_text_pegasus(context.input_text)
        self.set_output(output=presets.StringGeneralOutput(output_text=output))
