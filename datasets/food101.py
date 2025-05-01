import os
import pickle

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


class Food101(D