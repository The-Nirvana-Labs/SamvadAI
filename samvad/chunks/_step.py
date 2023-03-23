from dataclasses import dataclass
from typing import List

from samvad.core import api, presets

from ._base import extract_chunks


@dataclass
class ExtractChunksOutput(api.Output):
    chunks: List[str]


class ExtractChunksStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context()
        chunks: List[str] = extract_chunks(context.input_text)
        output = ExtractChunksOutput(chunks=chunks)
        self.set_output(output)
