from dataclasses import dataclass

from samvad.core import api

from ._base import remove_stopwords


@dataclass
class RemoveStopwordsContext(api.Context):
    input_text: str


@dataclass
class RemoveStopwordsOutput(api.Output):
    filtered_text: str


class RemoveStopwordsStep(api.Step):
    def run(self) -> None:
        context: RemoveStopwordsContext = self.get_context(RemoveStopwordsContext)
        filtered_text: str = remove_stopwords(context.input_text)
        output: RemoveStopwordsOutput = RemoveStopwordsOutput(filtered_text)
        self.set_output(output)
