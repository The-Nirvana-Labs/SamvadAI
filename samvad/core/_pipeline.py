from __future__ import annotations

from queue import SimpleQueue, Empty as QueueEmptyException
from typing import List

from ._api import Step, Output, Context
from ._casting import CastOutputToContext, CastingContext, CastingOutput


class Pipeline(object):
    """
    A simple pipeline with queue under the hood to run steps one by one.
    It expects that after each step, there should be a casing step which casts output from current step
    to context for next step.
    At any time, if a normal step is running, it will try to infuse cast of previous step, if any.
    All steps are required to have standalone-context already inserted while queuing.
    """
    def __init__(self, entry_context: Context):
        self._execution_context: ExecutionContext = ExecutionContext()
        self._entry_context: Context = entry_context

    def load_steps(self, arg: List[Step] | Step):
        if not isinstance(arg, list):
            arg = [arg]
        for step in arg:
            self._execution_context.put(step)

    def get_running_step(self):
        return self._execution_context.get_running_step()

    def start(self) -> int:
        previous_output: Output | None = None
        previous_context: Context | None = None
        total_executed_steps: int = 0
        while True:
            step = self._execution_context.consume()
            if step is None:
                break
            if total_executed_steps == 0:
                # First step
                self._entry_context.step_id = step.get_id()
                step.set_context(self._entry_context)
            if isinstance(step, CastOutputToContext):
                casting_context = CastingContext(previous_context, previous_output)
                casting_context.step_id = step.get_id()
                step.set_context(casting_context)
            else:
                if previous_output is not None and isinstance(previous_output, CastingOutput):
                    previous_output: CastingOutput = previous_output
                    step.set_context(previous_output.next_context)
            step.run()
            previous_context = step.get_context()
            previous_output = step.get_output()
            previous_output.step_id = step.get_id()
            self._execution_context.put_output(previous_output)
            total_executed_steps += 1
        return total_executed_steps

    def get_outputs(self) -> SimpleQueue[Output]:
        return self._execution_context.get_outputs()


class ExecutionContext(object):
    """
    A simple queue wrapper to track which step is running at instance level.
    It ensures that after every non cast-output-to-context step, there is a cast-output-to-context-step
    """
    def __init__(self):
        self._steps: SimpleQueue[Step] = SimpleQueue()
        self._outputs: SimpleQueue[Output] = SimpleQueue()

        self._running_step: Step | None = None
        self._previous_step: Step | None = None
        self._previous_insertion: Step | None = None

    def put(self, step: Step) -> None:
        if self._previous_insertion is not None:
            if not isinstance(step, CastOutputToContext):
                if not isinstance(self._previous_insertion, CastOutputToContext):
                    raise AssertionError("Previous Step is not of CastOutputToContext type.")
        self._put(step)
        self._previous_insertion = step

    def put_output(self, output: Output) -> None:
        self._put(output)

    def consume(self) -> Step | None:
        self._previous_step = self._running_step
        self._running_step = self._consume()
        return self._running_step

    def get_running_step(self) -> Step:
        return self._running_step

    def get_outputs(self) -> SimpleQueue[Output]:
        return self._outputs

    def _consume(self) -> Step | None:
        try:
            return self._steps.get(block=False)
        except QueueEmptyException:
            return None

    def _put(self, obj: Step | Output) -> None:
        if isinstance(obj, Step):
            self._steps.put(obj)
        elif isinstance(obj, Output):
            self._outputs.put(obj)
