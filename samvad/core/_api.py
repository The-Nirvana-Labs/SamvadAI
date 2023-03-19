from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

from samvad.utils import generate_random_string


@dataclass
class StepObject:
    step_id: str = field(init=False)


class Context(StepObject):
    pass


class Output(StepObject):
    pass


class Step(object):
    def __init__(self) -> None:
        self._id: str = generate_random_string(5)
        self._context: Context | None = None
        self._output: Output | None = None

    @abstractmethod
    def run(self) -> None:
        pass

    def set_context(self, context) -> None:
        self._context = context

    def get_context(self, context_class=None) -> Context:
        if context_class and not isinstance(self._context, context_class):   # Optional typechecking
            raise AssertionError("Context class type not matching. Maybe wrong context has been passed")
        return self._context

    def get_output(self) -> Output:
        return self._output

    def set_output(self, output: Output):
        self._output = output

    def get_id(self) -> str:
        return self._id
