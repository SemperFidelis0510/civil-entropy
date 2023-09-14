import os
import time


def normalize_path(path_str):
    # Split by both UNIX and Windows separators
    parts = path_str.replace('\\', '/').split('/')
    # Join with the appropriate OS separator
    return os.path.join(*parts)


def print_progress_bar(text, iteration, total, start_time=None, length=50):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + '-' * (length - filled_length)

    if start_time is not None:
        elapsed_time = time.time() - start_time
        total_time = elapsed_time * total / max(1, iteration)
        t_left = elapsed_time * (total - iteration) / max(1, iteration)
        mins, secs = divmod(int(elapsed_time), 60)
        elapsed_time = f"{mins:02d}:{secs:02d}"
        mins, secs = divmod(int(t_left), 60)
        t_left = f"{mins:02d}:{secs:02d}"
        mins, secs = divmod(int(total_time), 60)
        total_time = f"{mins:02d}:{secs:02d}"

        progress_bar = f"{bar} | Completed: {iteration}/{total} {percent}% | Time elapsed: {elapsed_time}" \
                       f"/{total_time} | Time left: ~{t_left}"
    else:
        progress_bar = f"{bar} | {percent}% Complete {iteration}/{total} instances."

    print(f'\r{text}: {progress_bar}', end='', flush=True)
