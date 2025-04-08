
# This code is used to generate non-iid data with Feature Skew

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from Dassl.dassl.data.datasets.base_dataset import Datum
from PIL import Image
import os
from collections import Counter


def prepare_data_domainNet(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    lab2cname = clipart_trainset.lab2cname
    classnames = clipart_trainset.classnames
    clipart_trainset = clipart_trainset.data_detailed
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False).data_detailed
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train).data_detailed
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False).data_detailed
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train).data_detailed
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False).data_detailed
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train).data_detailed
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False).data_detailed
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train).data_detailed
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False).data_detailed
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train).data_detailed
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False).data_detailed

    # min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset))
    # print("Train data number: ", min_data_len)
    train_data_num_list = [len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)]
    test_data_num_list = [len(clipart_testset), len(infograph_testset), len(painting_testset), len(quickdraw_testset), len(real_testset), len(sketch_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]
    test_set = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]

    return train_set, test_set, classnames, lab2cname


def prepare_data_domainNet_partition_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    min_require_size = 5
    n_clients = 5
    print("Clipart: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'clipart', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('clipart', cfg.DATASET.BETA, split_test=False, n_parties=cfg.DATASET.USERS, min_require_size=2)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('clipart', train_ratio[0])
    clipart_trainset = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_train, transform=transform_train)
    clipart_testset = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    lab2cname = clipart_trainset.lab2cname
    classnames = clipart_trainset.classnames
    clipart_trainset = clipart_trainset.data_detailed
    
    print("Infograph: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'infograph', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('infograph', train_ratio[0])
    infograph_trainset = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_train, transform=transform_train).data_detailed
    infograph_testset = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Painting: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'painting', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('painting', train_ratio[2])
    painting_trainset = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_train, transform=transform_train).data_detailed
    painting_testset = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Quickdraw: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'quickdraw', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('quickdraw', train_ratio[1])
    quickdraw_trainset = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_train, transform=transform_train).data_detailed
    quickdraw_testset = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Real")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'real', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('real', train_ratio[0])
    real_trainset = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_train, transform=transform_train).data_detailed
    real_testset = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Sketch")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'sketch', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('sketch', train_ratio[0])
    sketch_trainset = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_train, transform=transform_train).data_detailed
    sketch_testset = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_test, transform=transform_test, train=False).data_detailed

    train_data_num_list = [len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)]
    test_data_num_list = [len(clipart_testset), len(infograph_testset), len(painting_testset), len(quickdraw_testset), len(real_testset), len(sketch_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]
    test_set = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]

    return train_set, test_set, classnames, lab2cname

def prepare_data_domainNet_partition_client_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    min_require_size = 2
    n_clients = 5
    print("Clipart: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'clipart', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('clipart', cfg.DATASET.BETA, split_test=False, n_parties=cfg.DATASET.USERS, min_require_size=2)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('clipart', train_ratio[0])
    clipart_trainset = [[] for i in range(n_clients)]
    clipart_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        clipart_trainset[i] = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_train[i], transform=transform_train)
        clipart_testset[i] = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
        lab2cname = clipart_trainset[i].lab2cname
        classnames = clipart_trainset[i].classnames
        clipart_trainset[i] = clipart_trainset[i].data_detailed
    
    print("Infograph: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'infograph', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('infograph', train_ratio[0])
    infograph_trainset = [[] for i in range(n_clients)]
    infograph_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        infograph_trainset[i] = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_train[i], transform=transform_train).data_detailed
        infograph_testset[i] = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Painting: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'painting', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('painting', train_ratio[2])
    painting_trainset = [[] for i in range(n_clients)]
    painting_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        painting_trainset[i] = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_train[i], transform=transform_train).data_detailed
        painting_testset[i] = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Quickdraw: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'quickdraw', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('quickdraw', train_ratio[1])
    quickdraw_trainset = [[] for i in range(n_clients)]
    quickdraw_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        quickdraw_trainset[i] = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_train[i], transform=transform_train).data_detailed
        quickdraw_testset[i] = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Real")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'real', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('real', train_ratio[0])
    real_trainset = [[] for i in range(n_clients)]
    real_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        real_trainset[i] = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_train[i], transform=transform_train).data_detailed
        real_testset[i] = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Sketch")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'sketch', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('sketch', train_ratio[0])
    sketch_trainset = [[] for i in range(n_clients)]
    sketch_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        sketch_trainset[i] = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_train[i], transform=transform_train).data_detailed
        sketch_testset[i] = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    for dataset in [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]:
        for i in range(n_clients):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    for dataset in [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]:
        for i in range(n_clients):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    return train_set, test_set, classnames, lab2cname

def prepare_data_office(cfg, data_base_path):
    data_base_path = data_base_path
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    lab2cname = amazon_trainset.lab2cname
    classnames = amazon_trainset.classnames
    amazon_trainset = amazon_trainset.data_detailed
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False).data_detailed
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office).data_detailed
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False).data_detailed
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office).data_detailed
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False).data_detailed
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office).data_detailed
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False).data_detailed

    train_data_num_list = [len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset)]
    test_data_num_list = [len(amazon_testset), len(caltech_testset), len(dslr_testset), len(webcam_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set =  [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    test_set = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]

    return train_set, test_set, classnames, lab2cname


def prepare_data_office_partition_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    K = 10
    min_require_size = 2
    n_clients = 3
    # amazon
    print("Amazon: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'amazon', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_office(data_base_path,'amazon', cfg.DATASET.BETA, split_test=False, n_parties=cfg.DATASET.USERS, min_require_size=min_img_num)
    # net_dataidx_map_test = Adjust_test_dataset_office('amazon', train_ratio[0])
    amazon_trainset = [[] for i in range(n_clients)]
    amazon_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        amazon_trainset[i] = OfficeDataset_sub(data_base_path, 'amazon', net_dataidx_map_train[i], transform=transform_train)
        amazon_testset[i] = OfficeDataset_sub(data_base_path, 'amazon', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed
        lab2cname = amazon_trainset[i].lab2cname
        classnames = amazon_trainset[i].classnames
        amazon_trainset[i] = amazon_trainset[i].data_detailed

    # caltech
    print("Caltech: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'caltech', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # caltech_trainset = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_train, transform=transform_train).data_detailed
    # caltech_testset = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_test, transform=transform_train).data_detailed
    caltech_trainset = [[] for i in range(n_clients)]
    caltech_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        caltech_trainset[i] = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_train[i], transform=transform_train).data_detailed
        caltech_testset[i] = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed

    
    # dslr
    print("dslr: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'dslr', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # dslr_trainset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_train, transform=transform_train).data_detailed
    # dslr_testset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_test, transform=transform_train).data_detailed
    dslr_trainset = [[] for i in range(n_clients)]
    dslr_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        dslr_trainset[i] = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_train[i], transform=transform_train).data_detailed
        dslr_testset[i] = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed
    
    
    # webcam
    print("Webcam: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'webcam', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # webcam_trainset = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_train, transform=transform_train).data_detailed
    # webcam_testset = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_test, transform=transform_train).data_detailed
    webcam_trainset = [[] for i in range(n_clients)]
    webcam_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        webcam_trainset[i] = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_train[i], transform=transform_train).data_detailed
        webcam_testset[i] = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    for dataset in [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]:
        for i in range(n_clients):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    for dataset in [amazon_testset, caltech_testset, dslr_testset, webcam_testset]:
        for i in range(n_clients):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    return train_set, test_set, classnames, lab2cname


def prepare_data_digits(cfg, data_base_path):
    data_base_path = data_base_path
    percent = 0.1
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = DigitsDataset(data_base_path, data_path="digits/MNIST", channels=1, percent=percent, train=True,  transform=transform_mnist)
    mnist_testset      = DigitsDataset(data_base_path, data_path="digits/MNIST", channels=1, percent=percent, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = DigitsDataset(data_path='data/digits/SVHN', channels=3, percent=percent,  train=True,  transform=transform_svhn)
    svhn_testset       = DigitsDataset(data_path='data/digits/SVHN', channels=3, percent=percent,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = DigitsDataset(data_path='data/digits/USPS', channels=1, percent=percent,  train=True,  transform=transform_usps)
    usps_testset       = DigitsDataset(data_path='data/digits/USPS', channels=1, percent=percent,  train=False, transform=transform_usps)