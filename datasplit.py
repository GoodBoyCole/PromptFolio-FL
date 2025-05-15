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
    # np.random.seed(2020)
    # torch.manual_seed(2020)

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = load_cifar10_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar100':
        # X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = load_cifar100_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)

    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 ==1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 >0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0 ,3999 ,4000 ,dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy" ,X_train)
        np.save("data/generated/X_test.npy" ,X_test)
        np.save("data/generated/y_train.npy" ,y_train)
        np.save("data/generated/y_test.npy" ,y_test)

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if partition == "homo":
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)

        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}

    elif partition == "iid-label100":
        seed = 12345
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = ra