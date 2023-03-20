from dataclasses import dataclass
from typing import List, Dict

from samvad.core import api, presets

from ._base import perform_ner_ecwl, perform_ner_flair


@dataclass
class PerformNEROutput(api.Output):
    entities: List[Dict[str, str]]


class StepEcwl(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        entities: List[Dict[str, str]] = perform_ner_ecwl(context.input_text)
        self.set_output(PerformNEROutput(entities))


class StepFlair(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context(presets.StringGeneralContext)
        entities: List[Dict[str, str]] = perform_ner_flair(context.input_text)
        self.set_output(PerformNEROutput(entities))
