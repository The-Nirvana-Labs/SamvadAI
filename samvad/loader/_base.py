from pathlib import Path

from ._utils import clean


def load_text_from_file(path: Path) -> str:
    with open(path, 'r') as fp:
        text = fp.read()
        text = clean(text)
    return text
