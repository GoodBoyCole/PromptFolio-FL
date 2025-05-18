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
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples_train = y_train.shape[0]
        n_samples_test = y_test.shape[0]

        selected_indices_train = rng.sample(list(range(n_samples_train)), n_samples_train)
        selected_indices_test = rng.sample(list(range(n_samples_test)), n_samples_test)

        n_samples_by_client_train = int((n_samples_train / n_parties) // 5)
        n_samples_by_client_test = int((n_samples_test / n_parties) // 5)

        indices_by_fine_labels_train = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_train = {k: list() for k in range(n_coarse_labels)}

        indices_by_fine_labels_test = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_test = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices_train:
            fine_label = y_train[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_train[fine_label].append(idx)
            indices_by_coarse_labels_train[coarse_label].append(idx)

        for idx in selected_indices_test:
            fine_label = y_test[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_test[fine_label].append(idx)
            indices_by_coarse_labels_test[coarse_label].append(idx)

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_train[fine_label]), n_samples_by_client_train)
                net_dataidx_map_train[client_idx] = np.append(net_dataidx_map_train[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_train[fine_label].remove(idx)

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_test[fine_label]), n_samples_by_client_test)
                net_dataidx_map_test[client_idx] = np.append(net_dataidx_map_test[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_test[fine_label].remove(idx)

    elif partition == "noniid-labeluni":
        if dataset == "cifar10":
            num = 2
        elif dataset == "cifar100":
            num = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar100':
            K = 100
        elif dataset == 'cifar10':
            K = 10
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i],
                                                         data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i],
                                                        data_class_idx_test[c][:end_idx_test])
                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        if dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100
        elif dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]
        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

                np.random.shuffle(train_idx_k)
                np.random.shuffle(test_idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array \
                    ([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions,