import logging
import sys
import os
import os.path as osp
from contextlib import contextmanager

def mkdir_if_missing(directory):
    os.makedirs(directory, exist_ok=True)

def log_print(msg):
    logging.info(msg)
    print(msg)

class Logger(object):
    def __init__(self, fpath=None, mode="a"):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            try:
                # Use append mode so logs from different runs (e.g., new random seeds)
                # are preserved instead of overwriting previous results.
                self.file = open(fpath, mode)
            except IOError as e:
                print(f"Error opening log file {fpath}: {e}", file=sys.stderr)
                self.file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            try:
                self.file.write(msg)
            except IOError:
                # Silently handle write errors or log to stderr if needed
                pass 

    def flush(self):
        self.console.flush()
        if self.file is not None:
            try:
                self.file.flush()
                os.fsync(self.file.fileno())
            except IOError:
                # Silently handle flush errors or log to stderr if needed
                pass

    def close(self):
        # Do not close sys.stdout
        if self.file is not None:
            try:
                self.file.close()
            except IOError:
                pass
            finally:
                self.file = None

# Example usage:
# with Logger("path/to/log.txt") as log_capture:
#     sys.stdout = log_capture
#     print("This goes to both console and file.")