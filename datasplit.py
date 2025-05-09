import numpy as np
import logging
import random
import time
from collections import defaultdict

from dataloader import load_mnist_data, load_fmnist_data, load_cifar10_data, load_cifar100_data, load_svhn_data, load_celeba_data, load_femnist_data
from dataset import mkdirs

logging.basicConfig()
logger = logg