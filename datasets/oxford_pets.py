import os
import pickle
import math
import random
from collections import defaultdict

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import read_json, write_json, mkdir_if_missing


# @DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):