
import functools
import torch.nn as nn

from .build import HEAD_REGISTRY


class MLP(nn.Module):

    def __init__(
        self,
        in_features=2048,
        hidden_layers=[],
        activation="relu",
        bn=True,
        dropout=0.0,