from samvad.core import api, presets

from ._base import perform_ner_ecwl, perform_ner_flair


class StepEcwl(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        output_text: str = perform_ner_ecwl(context.input_text)
        self.set_output(presets.StringGeneralOutput(output_text=output_text))


class StepFlair(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        output_text: str = perform_ner_flair(context.input_text)
        self.set_output(presets.StringGeneralOutput(output_text=output_text))
