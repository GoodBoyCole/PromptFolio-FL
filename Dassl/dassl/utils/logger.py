import os
import sys
import time
import os.path as osp

from .tools import mkdir_if_missing

__all__ = ["Logger", "setup_logger"]


class Logger:
    """Write console output to exter