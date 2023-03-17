def sum_string_lengths(lst):
    total_length = 0
    for s in lst:
        total_length += len(s)
    return total_length


def concatenate_strings(lst):
    result = ""
    for s in lst:
        result += s
    return result

