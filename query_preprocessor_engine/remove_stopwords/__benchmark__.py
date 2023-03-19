import time
from memory_profiler import memory_usage


def benchmark_remove_stopwords(remove_stopwords, text):
    """
    Benchmarks the remove_stopwords function.

    Args:
    - text (str): The input text to filter.

    Returns:
    - dict: A dictionary containing the benchmark results.
    """
    start_time = time.time()
    mem_usage_before = memory_usage()[0]

    filtered_text = remove_stopwords(text)

    mem_usage_after = memory_usage()[0]
    end_time = time.time()

    benchmark_results = {
        'text_size': len(text),
        'filtered_text_size': len(filtered_text),
        'execution_time': end_time - start_time,
        'memory_usage': mem_usage_after - mem_usage_before
    }

    return benchmark_results
