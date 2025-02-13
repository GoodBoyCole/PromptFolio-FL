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
            raise ValueError(
                "Unknown cost function: {}. Expected to "
                "be one of [cosine | euclidean]".format(dist_metric)
            )
        return dist_mat


class SinkhornDivergence(OptimalTransport):
    thre = 1e-3

    def __init__(
        self,
        dist_metric="cosine",
        eps=0.01,
        max_iter=5,
        bp_to_sinkhorn=False
    ):
        super().__init__()
        self.dist_metric = dist_metric
        self.eps = eps
        self.max_iter = max_iter
        self.bp_to_sinkhorn = bp_to_sinkhorn

    def forward(self, x, y):
        # x, y: two batches of data with shape (batch, dim)
        W_xy = self.transport_cost(x, y)
        W_xx = self.transport_cost(x, x)
        W_yy = self.transport_cost(y, y)
        return 2*W_xy - W_xx - W_yy

    def transport_cost(self, x, y, return_pi=False):
        C = self.distance(x, y, dist_metric=self.dist_metric)
        pi = self.sinkhorn_iterate(C, self.eps, self.max_iter, self.thre)
        if n