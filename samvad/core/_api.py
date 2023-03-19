from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from samvad.utils import generate_random_string


@dataclass
class StepObject:
    step_id: str


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

    def get_context(self) -> Context:
        return self._context

    def get_output(self) -> Output:
        return self._output

    def get_id(self) -> str:
        return self._id
