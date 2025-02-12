import torch
import torch.nn as nn
from torch.nn import functional as F


class OptimalTransport(nn.Module):

    @staticmethod
    def distance(batch1, batch2, dist_metric="cosine"):
        if dist_metric == "cosine":
            batch1 = F.normalize(batch1, p=2, dim=1)
            batch2 = F.normalize(batch2, p=2, dim=1)
            dist_mat = 1 - torch.mm(batch1, batch2.t())
        elif dist_metric == "euclidean":
            m, n = batch1.size(0), batch2.size(0)
            dist_mat = (
                torch.pow(batch1, 2).sum(dim=1, keepdim=True).expand(m, n) +
                torch.pow(batch2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            )
            dist_mat.addmm_(
                1, -2, batch1, batch2.t()
            )  # squared euclidean distance
        elif dist_metric == "fast_euclidean":
            batch1 = batch1.unsqueeze(-2)
            batch2 = batch2.unsqueeze(-3)
            dist_mat = torch.sum((torch.abs(batch1 - batch2))**2, -1)
        else:
            raise 