import numpy as np
import logging
import random
import time
from collections import defaultdict

from dataloader import load_mnist_data, load_fmnist_data, load_cifar10_data, load_cifar100_data, load_svhn_data, load_celeba_data, load_femnist_data
from dataset import mkdirs

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def record_net_data_stats(y_train, net_dataidx_map, logdir=None):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights

def partition_data(dataset, datadir, partition, n_parties, beta=0.4, logdir=None):
    # np.