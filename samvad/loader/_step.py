from dataclasses import dataclass
from pathlib import Path
from os.path import join as path_join

from samvad.core import api

from ._base import load_text_from_file


class LoadTextContext(api.Context):
    def __init__(self, text_file_name: str, base_path: Path):
        self.path = Path(path_join(base_path, text_file_name))


@dataclass
class LoadTextOutput(api.Output):
    output_text: str


class LoadTextStep(api.Step):
    def run(self) -> None:
        context: LoadTextContext = self.get_context(LoadTextContext)
        text = load_text_from_file(context.path)
        self.set_output(LoadTextOutput(output_text=text))
