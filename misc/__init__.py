import contextlib
import logging
from datetime import datetime
from logging.handlers import MemoryHandler
from pathlib import Path

from .TqdmLoggingHandler import TqdmLoggingHandler


@contextlib.contextmanager
def decorate_print(print_func, string, char_deco="=", len_deco=79):
    print_func(f" {string} ".center(len_deco, char_deco))
    yield
    print_func(char_deco * len_deco)


def make_parent_dir(filename: str):
    parent = Path(filename).parent
    if not parent.exists():
        parent.mkdir()


def initialize_root_logger(enable_logfile, log_filename=None, sim_case_description=""):
    date = datetime.now()
    if enable_logfile and log_filename is None:
        log_filename = (
            f"./logs/{date.strftime('%Y-%m-%d_%H-%M-%S')}_{sim_case_description}.log"
        )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(level=logging.ERROR)
    tqdm_handler = TqdmLoggingHandler(logging.INFO)
    tqdm_handler.setFormatter(logging.Formatter("{message}", style="{"))
    logger.addHandler(tqdm_handler)
    if enable_logfile:
        make_parent_dir(log_filename)
        file_handler = logging.FileHandler(
            filename=log_filename, encoding="utf-8", delay=True
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("{levelname:<5} | {message}", style="{")
        )
        memory_handler = MemoryHandler(
            capacity=1000, flushLevel=logging.ERROR, target=file_handler
        )
        logger.addHandler(memory_handler)
