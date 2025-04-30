
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from functools import partial
from typing import Optional, Callable
from torch.utils.model_zoo import tqdm
import PIL
import tarfile

import os
import os.path
import logging
import torchvision.datasets.utils as utils
import pickle
import string
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset
# if meet problem with cannot set attribute, consider add following line:
# from Dassl.dassl.data.datasets.base_dataset import Datum
# and delete Datum class
from Dassl.dassl.utils import check_isfile

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        data (float): data.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, data, label=0, domain=0, classname=""):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._data = data
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

    # @property
    # def target(self):
    #     return self._target
    #
    # @property
    # def gttarget(self):
    #     return self._gttarget
    #
    # @label.setter
    # def label(self, value):
    #     self._label = value
    #
    # @gttarget.setter
    # def gttarget(self, value):
    #     self._gttarget = value
    #
    # @target.setter
    # def target(self, value):
    #     self._target = value


class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None: