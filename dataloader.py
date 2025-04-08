import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
import copy
from collections import OrderedDict, defaultdict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from math import sqrt
import torch.nn as nn
import time
import random

from dataset import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CelebA_custom, FEMNIST, Generated, genData, CharacterDataset, SubFEMNIST

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
   