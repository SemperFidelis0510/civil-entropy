import os
from datetime import timedelta
import time


def normalize_path(path_str):
    # Split by both UNIX and Windows separators
    parts = path_str.replace('\\', '/').split('/')
    # Join with the appropriate OS separator
    return os.path.join(*parts)


def print_progress_bar(text, iteration, total, start_time=None, length=50):
    """Prints a progress bar with the percentage of completion and time statistics."""
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + '-' * (length - filled_length)

    if start_time is not None:
        elapsed_time = time.time() - start_time
        total_time = elapsed_time * total / max(1, iteration)
        t_left = elapsed_time * (total - iteration) / max(1, iteration)

        elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
        t_left_str = str(timedelta(seconds=int(t_left)))
        total_time_str = str(timedelta(seconds=int(total_time)))

        progress_bar = f"{bar} | Completed: {iteration}/{total}, {percent}% | Time elapsed: {elapsed_time_str}" \
                       f"/{total_time_str} | Time left: ~= {t_left_str} |"
    else:
        progress_bar = f"{bar} | {percent}% Complete {iteration}/{total} instances."

    print(f'\r{text}: {progress_bar}', end='', flush=True)
