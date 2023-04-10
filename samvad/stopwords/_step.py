from samvad.core import api, presets

from ._base import remove_stopwords


class RemoveStopwordsStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        filtered_text: str = remove_stopwords(context.input_text)
        output: presets.StringGeneralOutput = presets.StringGeneralOutput(filtered_text)
        self.set_output(output)
