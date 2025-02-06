import torch.nn as nn
from torch.nn import functional as F

__all__ = ["Attention"]


class Attention(nn.Module):
    """Attention from `"Dynamic Domain Generalization" <https://github.com/MetaVisionLab/DDG>`_.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        squeeze=None,
        bias: bool = True
    ):
        super(Attention, self).__init__()
        self.squeeze = squeeze if squeeze else in_channels // 16
        assert self.squeeze > 0
      