from samvad.core import casting, api
from samvad.loader import LoadTextOutput, LoadTextContext

from ._step import RemoveStopwordsContext


class CastRemoveStopwordContextFromLoadText(casting.CastOutputToContext):
    def cast(self, context: LoadTextContext, output: LoadTextOutput) -> api.Context:
        next_context: RemoveStopwordsContext = RemoveStopwordsContext(output.output_text)
        return next_context
