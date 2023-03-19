def clean(raw_string: str) -> str:
    # TODO: Find optimised way to do this. Eg: StringBuffer from Java
    raw_string = raw_string.strip()
    raw_string = raw_string.replace("\n", " ")
    # Further filtering here.
    return raw_string
