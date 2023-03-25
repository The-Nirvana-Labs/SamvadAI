from samvad.core import api, presets

from ._base import lemmatize_text_ecws, lemmatize_text_wordnet


class LemmatizeTextEcwsStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        output_text = lemmatize_text_ecws(context.input_text)
        self.set_output(presets.StringGeneralOutput(output_text))


class LemmatizeTextWordnetStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        output_text = lemmatize_text_wordnet(context.input_text)
        self.set_output(presets.StringGeneralOutput(output_text))
