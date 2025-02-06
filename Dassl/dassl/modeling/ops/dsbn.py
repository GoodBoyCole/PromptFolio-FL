import torch.nn as nn


class _DSBN(nn.Module):
    """Domain Specific Batch Normalization.

    Args:
        num_features (int): number of features.
        n_domain (int): number of domains.
        bn_type (str): type of bn. Choices are ['1d', '2d'].
    """

    def __init__(self, num_features, n_domain, bn_type):
        super().__init__()
        if bn_type == "1d":
            BN = nn.Batch