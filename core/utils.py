import logging
import sys


def fix_lightning_logger():
    """
    https://github.com/Lightning-AI/lightning/issues/16081
    You can fix this by tweaking your root logger to print to stdout for INFO and below, and stderr for everything else.
     All the other loggers will inherit from this
    """
    _logger = logging.getLogger('pytorch_lightning')
    _logger.setLevel(logging.INFO)
    _logger.handlers.clear()

    # log lower levels to stdout
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
    _logger.addHandler(stdout_handler)

    # log higher levels to stderr (red)
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.addFilter(lambda rec: rec.levelno > logging.INFO)
    _logger.addHandler(stderr_handler)
