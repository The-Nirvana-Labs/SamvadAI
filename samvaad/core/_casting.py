from abc import abstractmethod
from dataclasses import dataclass

from ._api import Step, Context, Output


@dataclass
class CastingContext(Context):
    context: Context
    output:  Output


@dataclass
class CastingOutput(Output):
    next_context: Context = None


class CastOutputToContext(Step):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        context: CastingContext = self.get_context()
        next_context: Context = self.cast(context.context, context.output)
        self._output = CastingOutput(context.step_id)
        self._output.next_context = next_context

    @abstractmethod
    def cast(self, context: Context, output: Output) -> Context:
        pass
