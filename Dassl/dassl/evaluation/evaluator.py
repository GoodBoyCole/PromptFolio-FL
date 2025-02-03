import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(sel