import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from itertools import product
from functools import reduce, partial

# import backbone architectures
import networks.resnet as resnet
from networks.wide_resnet import Wide_ResNet, Wide_ResNet_Sep
from networks.tcn import TCN
from networks.deepsea import DeepSEA
from networks.fno import Net2d
from networks.deepcon import DeepCon
from networks.wrn1d import ResNet1D
from networks.unet2d import UNet, UNet_small
from networks.unet1d import UNet1D

# import data loaders, task-specific losses and metrics
# from data_loaders import load_cifar, load_mnist, load_deepsea, load_darcy_flow, load_psicov, load_music, load_ecg, load_satellite, load_ninapro, load_cosmic, load_spherical, load_fsd, load_geobench, load_traffic, load_nucleotide_transformer
import sys 
sys.path.append('./')
from src.data_loaders import load_deepsea, load_deepsea_full, load_genomic_benchmarks, load_nucleotide_transformer, load_deepstarr
# from satellite_data_loader.bigearth_loader import load_BigEarthNet
# from satellite_data_loader.canadian_cropland_loader import load_cropland
# from satellite_data_loader.fmow_loader import load_fmow
# from task_utils import FocalLoss, LpLoss, BCEWithLogitsLoss
# from task_utils import mask, accuracy, accuracy_onehot, auroc, psicov_mae, ecg_f1, fnr, map_value, mean_average_precision1, mean_average_precision2
# from task_utils import mcc
# from data.ts_datasets import get_timeseries_dataloaders
from pathlib import Path

# ts_file_dict = {
#     "ETTh1": "/ts_datasets/all_six_datasets/ETT-small/ETTh1.csv",
#     "ETTh2": "/ts_datasets/all_six_datasets/ETT-small/ETTh2.csv",
#     "ETTm1": "/ts_datasets/all_six_datasets/ETT-small/ETTm1.csv",
#     "ETTm2": "/ts_datasets/all_six_datasets/ETT-small/ETTm2.csv",
#     "ECL": "/ts_datasets/all_six_datasets/electricity/electricity.csv",
#     "ER": "/ts_datasets/all_six_datasets/exchange_rate/exchange_rate.csv",
#     "ILI": "/ts_datasets/all_six_datasets/illness/national_illness.csv",
#     "Traffic": "/ts_datasets/all_six_datasets/traffic/traffic.csv",
#     "Weather": "/ts_datasets/all_six_datasets/weather/weather.csv",
# }

# import customized optimizers
# from optimizers import ExpGrad

def get_data(root_dir, dataset, batch_size, arch, valid_split, split_state = 42):
    data_kwargs = None
    root = "/home/wenduoc/ORCA/L2G/src/datasets/"

    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None

    # elif len(dataset.split("_")) > 1 and dataset.split("_")[0] in ts_file_dict:
    #     params = dataset.split("_")
    #     prefix = params[0]
    #     horizon = int(params[1])
    #     input_length = 96 if prefix == "ILI" else 512
    #     train_loader, val_loader, test_loader = get_timeseries_dataloaders(
    #         Path(root_dir+ts_file_dict[prefix]),
    #         batch_size=batch_size,
    #         seq_len=input_length,
    #         forecast_horizon=horizon
    #     ) 
    
    elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
        train_loader, val_loader, test_loader = load_nucleotide_transformer(root, batch_size, one_hot = True, valid_split=valid_split, dataset_name = dataset)
    elif dataset in ['ISCXVPN2016', 'USTC-TFC2016', 'CICIoT2022', 'ISCXTor2016']:
        train_loader, val_loader, test_loader = load_traffic(dataset, root_dir, batch_size, valid_split=valid_split)
    elif dataset == "big_earth_net":
        train_loader, val_loader, test_loader = load_geobench(batch_size=batch_size, dataset='bigearthnet', root_dir=root_dir, valid_split=valid_split)
    elif dataset == "brick_kiln":
        train_loader, val_loader, test_loader = load_geobench(batch_size=batch_size, dataset='brickkiln', root_dir=root_dir, valid_split=valid_split)
    elif dataset == "eurosat":
        train_loader, val_loader, test_loader = load_geobench(batch_size=batch_size, dataset='eurosat', root_dir=root_dir, valid_split=valid_split)
    elif dataset == "so2sat":
        train_loader, val_loader, test_loader = load_geobench(batch_size = batch_size, dataset='so2sat', root_dir=root_dir, valid_split=valid_split)
    elif dataset == "forestnet":
        train_loader, val_loader, test_loader = load_geobench(batch_size=batch_size, dataset='forestnet', root_dir=root_dir, valid_split=valid_split)
    elif dataset == "pv4ger":
        train_loader, val_loader, test_loader = load_geobench(batch_size=batch_size, dataset='pv4ger', root_dir=root_dir, valid_split=valid_split)
    elif dataset == "BigEarth":
        train_loader, val_loader, test_loader = load_BigEarthNet(batch_size=batch_size, root_dir=root_dir)
        n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)
        if not valid_split:
            val_loader = test_loader
            n_val = n_test

    elif dataset == "canadian_cropland":
        train_loader, val_loader, test_loader = load_cropland(batch_size=batch_size, root_dir=root_dir)
        n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)
        if not valid_split:
            val_loader = test_loader
            n_val = n_test
    elif dataset == "fmow":
        train_loader, val_loader, test_loader = load_fmow(batch_size=batch_size, root_dir=root_dir)
        n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)
        if not valid_split:
            val_loader = test_loader
            n_val = n_test
    elif dataset == "CIFAR10":
        train_loader, val_loader, test_loader = load_cifar(root_dir, 10, batch_size, valid_split=valid_split)
    elif dataset == "CIFAR10-PERM":
        train_loader, val_loader, test_loader = load_cifar(root_dir, 10, batch_size, permute=True, valid_split=valid_split)
    elif dataset == "CIFAR100":
        train_loader, val_loader, test_loader = load_cifar(root_dir, 100, batch_size, valid_split=valid_split)
    elif dataset == "CIFAR100-PERM":
        train_loader, val_loader, test_loader = load_cifar(root_dir, 100, batch_size, permute=True, valid_split=valid_split)
    elif dataset == "MNIST":
        train_loader, val_loader, test_loader = load_mnist(root_dir, batch_size, valid_split=valid_split)
    elif dataset == "MNIST-PERM":
        train_loader, val_loader, test_loader = load_mnist(root_dir, batch_size, permute=True, valid_split=valid_split)
    elif dataset == "SPHERICAL":
        train_loader, val_loader, test_loader = load_spherical(root_dir, batch_size, valid_split=valid_split)
    elif dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(root_dir, batch_size, valid_split=valid_split)
    elif dataset == "DARCY-FLOW-5":
        train_loader, val_loader, test_loader, y_normalizer = load_darcy_flow(root_dir, batch_size, sub = 5, arch = arch, valid_split=valid_split)
        data_kwargs = {"decoder": y_normalizer}
    elif dataset == 'PSICOV':
        train_loader, val_loader, test_loader, _, _ = load_psicov(root_dir, batch_size, valid_split=valid_split)
    elif dataset[:5] == 'MUSIC':
        if dataset[6] == 'J': length = 255
        elif dataset[6] == 'N': length = 513
        train_loader, val_loader, test_loader = load_music(root_dir, batch_size, dataset[6:], length=length, valid_split=valid_split)
    elif dataset == "ECG":
        train_loader, val_loader, test_loader = load_ecg(root_dir, batch_size, valid_split=valid_split)
    elif dataset == "SATELLITE":
        train_loader, val_loader, test_loader = load_satellite(root_dir, batch_size, valid_split=valid_split)
    elif dataset == "NINAPRO":
        train_loader, val_loader, test_loader = load_ninapro(root_dir, batch_size, arch, valid_split=valid_split)
    elif dataset == "COSMIC":
        valid_split = True
        train_loader, val_loader, test_loader = load_cosmic(root_dir, batch_size, valid_split=valid_split)
        data_kwargs = {'transform': mask}
    elif dataset == "FSD":
        train_loader, val_loader, test_loader = load_fsd(root_dir, batch_size, valid_split=valid_split)

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split or valid_split == 0:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_model(arch, sample_shape, num_classes, config_kwargs, ks = None, ds = None, dropout = None):
    in_channel = sample_shape[1]
    activation = config_kwargs['activation']
    remain_shape = config_kwargs['remain_shape']
    if dropout is None:
        dropout = config_kwargs['dropout']
    pool_k = config_kwargs['pool_k']
    squeeze = config_kwargs['squeeze']

    tokenized = config_kwargs['tokenized']
    num_embeddings = config_kwargs['num_embeddings']
    embedding_dim = config_kwargs['embedding_dim']

    if len(sample_shape) == 4:

        if arch == 'your_new_arch': # modify this to experiment with a new architecture
            model = None
        elif arch == 'unet2d':
            model = UNet(in_channel, num_classes, ks = ks, ds = ds)
        elif arch == 'unet2d_small':
            model = UNet_small(in_channel, num_classes, ks = ks, ds = ds)
        elif 'wrn' in arch:
            if 'sep' in arch:
                wrn = Wide_ResNet_Sep
            else:
                wrn = Wide_ResNet
            try:
                splits = arch.split('-')
                model = wrn(int(splits[1]), int(splits[2]), dropout, in_channel=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, pool_k=pool_k, squeeze=squeeze)
            except IndexError:
                model = wrn(28, 10, 0.3, in_channel=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, pool_k=pool_k, squeeze=squeeze)
        elif 'convnext' in arch:
            from networks.convnext import convnext_xtiny, convnext_tiny
            model = convnext_xtiny(in_chans=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif 'resnet' in arch:
            model = resnet.__dict__[arch](in_channel=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, pool_k=pool_k, squeeze=squeeze)
        elif 'fno' in arch:
            model = Net2d(12, 32, op=arch[4:], einsum=True, ks = ks, ds = ds)
        elif arch == 'deepcon':
            model = DeepCon(L=128, num_blocks=8, width=16, expected_n_channels=57, no_dilation=False, ks = ks, ds = ds)
    
    else:
        if 'mid_channels' in config_kwargs.keys():
            mid_channels = config_kwargs['mid_channels']
        else:
            mid_channels = min(4 ** (num_classes // 10 + 1), 64)
       

        if arch == 'your_new_arch': # modify this to experiment with a new architecture
            model = None
        elif arch == 'TCN':
            model = TCN(in_channel, num_classes, [100] * 8, kernel_size=7, dropout=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif arch == 'wrn':
            if tokenized:
                model = ResNet1D(in_channels = embedding_dim, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, tokenized=tokenized, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            else:
                model = ResNet1D(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif arch == 'unet':
            if tokenized:
                model = UNet1D(n_channels=embedding_dim, num_classes=num_classes, ks=ks, ds=ds, tokenized=tokenized, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            else:
                model = UNet1D(n_channels=in_channel, num_classes=num_classes, ks=ks, ds=ds)
        elif arch == 'deepsea':
            model = DeepSEA(ks = ks, ds = ds,in_channel=in_channel, num_classes=num_classes)
   
    return model


def get_config(dataset, args=None):
    einsum = True
    base, accum = 0.2, 1
    validation_freq = 1
    clip, retrain_clip = 1, -1
    quick_search, quick_retrain = 0.2, 1
    config_kwargs = {'temp': 1, 'arch_retrain_default': None, 'grad_scale': 100, 'activation': None, 'remain_shape': False, 'pool_k': 8, 'squeeze': False, 'dropout': 0, 'tokenized': False, 'num_embeddings': None, 'embedding_dim': None,}
    
    if dataset == "your_new_task": # modify this to experiment with a new task
        dims, sample_shape, num_classes = None, None, None
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        loss = None

        batch_size = 64
        arch_default = 'wrn'

    # elif len(dataset.split("_")) > 1 and dataset.split("_")[0] in ts_file_dict:
    #     params = dataset.split("_")
    #     prefix = params[0]
    #     horizon = int(params[1])
    #     dims, sample_shape, num_classes = 1, (1, 1, 512), horizon
    #     ks = [7, 9, 11, 13]
    #     if horizon == 192:
    #         ks = [9, 11, 13, 15]
    #     if horizon == 336:
    #         ks = [11, 13, 15, 17]
    #     if horizon == 720:
    #         ks = [13, 15, 17, 19]
    #     kernel_choices_default, dilation_choices_default = ks, [1, 3, 7, 15]
    #     loss = nn.MSELoss()

    #     batch_size = 64
    #     arch_default = 'wrn'

    # elif dataset in ['ISCXVPN2016', 'USTC-TFC2016', 'CICIoT2022', 'ISCXTor2016']:
    #     data2cls = {"ISCXVPN2016": 7, "ISCXTor2016": 8, "USTC-TFC2016": 20, "CICIoT2022": 10}
    #     dims, sample_shape, num_classes = 1, (1, 1, 512), data2cls[dataset]
    #     kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
    #     loss = nn.CrossEntropyLoss()
    #     if dataset == 'ISCXTor2016':
    #         config_kwargs['grad_scale'] = 500
    #     config_kwargs['tokenized'] = True
    #     config_kwargs['num_embeddings'] = 60005
    #     config_kwargs['embedding_dim'] = 256
    #     config_kwargs['mid_channels'] = 128

    #     batch_size = 32
    #     arch_default = 'wrn'

    elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9, 11], [1, 3, 5, 7]
        loss = nn.CrossEntropyLoss()
        
        batch_size = 128 
        arch_default = 'wrn'
        config_kwargs['mid_channels'] = 128
        if dataset == "enhancers":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 2
        elif dataset == "enhancers_types":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 3
        elif dataset in ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac']:
            dims, sample_shape, num_classes = 1, (1, 5, 500), 2
        elif dataset in ['promoter_all', 'promoter_no_tata', 'promoter_tata']:
            dims, sample_shape, num_classes = 1, (1, 5, 300), 2
        elif dataset in ['splice_sites_acceptors', 'splice_sites_donors']:
            dims, sample_shape, num_classes = 1, (1, 5, 600), 2
        elif dataset == 'splice_sites_all':
            dims, sample_shape, num_classes = 1, (1, 5, 400), 3

    elif dataset == "so2sat":
        dims, sample_shape, num_classes = 2, (1, 18, 64, 64), 17
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 16
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8'
        config_kwargs['grad_scale'] = 500

    
    elif dataset == "big_earth_net":
        dims, sample_shape, num_classes = 2, (1, 12, 120, 120), 43
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        # posweight = torch.tensor([9.8225, 45.6200, 18.4553, 36.4532, 28.9850,  2.4530, 87.4956, 36.9507,
        #  3.1263,  3.1876, 42.1965, 20.8579,  3.7393, 47.1928, 42.3839, 20.4592,
        # 34.5872, 11.5865, 23.6609, 42.0108,  2.8001, 22.5294,  2.6941, 21.3464,
        # 18.6271,  1.9727, 13.9365,  3.7048, 19.1816, 12.2275, 70.9424, 23.8756,
        # 23.7831, 87.1057, 29.9598, 15.6806,  9.4932, 39.0802, 18.2678,  2.4252,
        # 19.3666, 10.1545, 16.2861]).cuda()
        # loss = BCEWithLogitsLoss(pos_weight=posweight, label_smoothing=0.1)
        loss = nn.MultiLabelSoftMarginLoss()

        batch_size = 16
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8' 
        config_kwargs['grad_scale'] = 100

    elif dataset == "brick_kiln":
        dims, sample_shape, num_classes = 2, (1, 13, 96, 96), 2
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 5, 7] 
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 16
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8' 
        config_kwargs['grad_scale'] = 500

    elif dataset == "eurosat":
        dims, sample_shape, num_classes = 2, (1, 13, 64, 64), 10
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 16
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8' 
        config_kwargs['grad_scale'] = 1000

    elif dataset == "BigEarth":
        dims, sample_shape, num_classes = 2, (1, 12, 120, 120), 19
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        # posweight = torch.tensor([6.4300, 45.4816, 1.8763, 8.7540, 4.7109, 4.4508, 3.0148,
        # 18.1973, 2.9350, 1.7611, 1.7313, 42.4056, 33.0567, 2.3729,
        # 425.7350, 19.4616, 333.1415, 5.3012, 6.1223]).cuda()
        # loss = BCEWithLogitsLoss(pos_weight=posweight, label_smoothing=0.1)
        loss = nn.MultiLabelSoftMarginLoss()

        batch_size = 8
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8'
        config_kwargs['grad_scale'] = 100

    elif dataset == "forestnet":
        dims, sample_shape, num_classes = 2, (1, 6, 332, 332), 12
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 5, 7]
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 8
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4' 
        config_kwargs['grad_scale'] = 100

    elif dataset == "pv4ger":
        dims, sample_shape, num_classes = 2, (1, 3, 320, 320), 2
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 5, 7]
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 16
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8' 
        # arch_default = 'unet2d_small'
        # config_kwargs['arch_retrain_default'] = 'unet2d'
        config_kwargs['grad_scale'] = 500
    
    elif dataset == "canadian_cropland":
        dims, sample_shape, num_classes = 2, (1, 12, 65, 65), 10
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 5, 7]
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 16
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8' 
        config_kwargs['grad_scale'] = 1000

    elif dataset == "fmow":
        dims, sample_shape, num_classes = 2, (1, 13, 96, 96), 62
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 5, 7]
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 16
        arch_default = 'wrn-22-1'
        config_kwargs['arch_retrain_default'] = 'wrn-22-8' 
        config_kwargs['grad_scale'] = 1000


    elif dataset[:5] == "CIFAR":
        dims, sample_shape, num_classes = 2, (1, 3, 32, 32), 10 if dataset in ['CIFAR10', 'CIFAR10-PERM'] else 100
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss()

        batch_size = 24
        arch_default = 'wrn-22-1' 
        config_kwargs['arch_retrain_default'] = 'wrn-22-8' 
        config_kwargs['grad_scale'] = 5000


    elif dataset == 'SPHERICAL':
        dims, sample_shape, num_classes = 2, (1, 3, 60, 60), 100
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
        loss = nn.CrossEntropyLoss() 

        batch_size = 64
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4' 
        config_kwargs['grad_scale'] = 500


    elif dataset == "DARCY-FLOW-5":
        dims, sample_shape, num_classes = 2, (1, 3, 85, 85), 1
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]  
        loss = LpLoss(size_average=False)
        
        batch_size = 10
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['remain_shape'] = config_kwargs['squeeze'] = True 
        

    elif dataset == "PSICOV":
        dims, sample_shape, num_classes = 2, (1, 57, 128, 128), 1
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]  
        loss = nn.MSELoss(reduction='mean')

        batch_size = 8
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['remain_shape']  = True


    elif dataset == "NINAPRO": 
        dims, sample_shape, num_classes = 2, (1, 1, 16, 52), 18
        kernel_choices_default, dilation_choices_default = [3, 5, 7], [1, 3, 7]
        loss = FocalLoss(alpha=1)

        batch_size = 128
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['pool_k'] = 0


    elif dataset == "COSMIC":
        dims, sample_shape, num_classes = 2, (1, 1, 128, 128), 1
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
        loss = nn.BCELoss()

        batch_size = 4
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['activation'] = 'sigmoid'
        config_kwargs['remain_shape'] = True
        config_kwargs['grad_scale'] = 1000


    elif dataset == 'FSD':
        dims, sample_shape, num_classes = 2, (1, 1, 96, 102), 200
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
        loss = nn.BCEWithLogitsLoss(pos_weight=10 * torch.ones((200, )))
        
        batch_size = 128
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['pool_k'] = 0


    elif dataset[:5] == "MNIST":
        dims, sample_shape, num_classes = 1, (1, 1, 784), 10
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15, 31, 63, 127] 
        loss = F.nll_loss

        batch_size = 256      
        arch_default = 'wrn'
        config_kwargs['activation'] = 'softmax'


    elif dataset[:5] == "MUSIC":
        if dataset[6] == 'J': length = 255 
        elif dataset[6] == 'N': length = 513
        dims, sample_shape, num_classes = 1, (1, 88, length - 1), 88
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.BCELoss()

        batch_size = 4
        arch_default = 'wrn'
        config_kwargs['activation'] = 'sigmoid'
        config_kwargs['remain_shape'] = True

    
    elif dataset == "ECG": 
        dims, sample_shape, num_classes = 1, (1, 1, 1000), 4
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss()

        batch_size = 1024
        arch_default = 'wrn'
        config_kwargs['activation'] = 'softmax'
        

    elif dataset == "SATELLITE":
        dims, sample_shape, num_classes = 1, (1, 1, 46), 24
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss()

        batch_size = 256
        arch_default = 'wrn'


    elif dataset == "DEEPSEA":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 36
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))

        batch_size = 256
        arch_default = 'wrn'  
        config_kwargs['grad_scale'] = 10 

    if args is not None:

        if args.grad_scale is not None:
            config_kwargs['grad_scale'] = args.grad_scale
        if args.pool_k is not None:
            config_kwargs['pool_k'] = args.pool_k
        if args.arch is not None:
            arch_default = args.arch
            if args.arch == "unet2d_small":
                config_kwargs['arch_retrain_default'] = "unet2d"
            elif args.arch == "wrn-22-1":
                config_kwargs['arch_retrain_default'] = "wrn-22-8"
            elif args.arch == "wrn-16-1":
                config_kwargs['arch_retrain_default'] = "wrn-16-4"
    
    lr, arch_lr = (1e-2, 5e-3) if config_kwargs['remain_shape'] else (0.1, 0.05) 

    # if arch_default == 'wrn':
    if args.arch == 'wrn':
        # epochs_default, retrain_epochs = 100, 150
        # epochs_default, retrain_epochs = 100, 100
        epochs_default, retrain_epochs = 80, 10
        retrain_freq = epochs_default
        opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        weight_decay = 5e-4 
        
        search_sched = [30, 60, 80]
        def weight_sched_search(epoch):
            optim_factor = 0
            for i in range(len(search_sched)):
                if epoch > search_sched[len(search_sched) - 1 - i]:
                    optim_factor = len(search_sched) - i
                    break

            return math.pow(base, optim_factor)

        if dims == 1:
            sched = [30, 60, 90, 120, 160]
        else:
            sched = [60, 120, 160]
        
        def weight_sched_train(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(base, optim_factor)
        
        hpo_sched = (np.array(sched)/5).astype(int)
        def weight_sched_hpo(epoch):
            optim_factor = 0
            for i in range(len(hpo_sched)):
                if epoch > hpo_sched[len(hpo_sched) - 1 - i]:
                    optim_factor = len(hpo_sched) - i
                    break

            return math.pow(base, optim_factor)

    elif arch_default == 'convnext':
        epochs_default, retrain_epochs, retrain_freq = 100, 300, 100
        opt, arch_opt = torch.optim.AdamW, torch.optim.AdamW
        lr, arch_lr = 4e-3, 1e-2
        weight_decay = 0.05
            
        base_value = lr
        final_value = 1e-6
        niter_per_ep = 392 
        warmup_iters = 0
        epochs = retrain_epochs
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]) / base_value

        def weight_sched_search(iter):
            return schedule[iter]
        
        def weight_sched_train(iter):
            return schedule[iter]

    elif arch_default == 'TCN':
        epochs_default, retrain_epochs, retrain_freq = 20, 40, 20
        opt, arch_opt = torch.optim.Adam, torch.optim.Adam
        weight_decay = 1e-4
        
        def weight_sched_search(epoch):
            return base ** (epoch // 10)
        
        def weight_sched_train(epoch):
            return base ** (epoch // 20)
        
    elif args.arch  == 'unet':
        epochs_default, retrain_epochs = 80, 10 # 100, 200
        retrain_freq = epochs_default
        opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        weight_decay = 5e-4 
        
        search_sched = [30, 60, 80]
        def weight_sched_search(epoch):
            optim_factor = 0
            for i in range(len(search_sched)):
                if epoch > search_sched[len(search_sched) - 1 - i]:
                    optim_factor = len(search_sched) - i
                    break

            return math.pow(base, optim_factor)

        if dims == 1:
            sched = [30, 60, 90, 120, 160]
        else:
            sched = [60, 120, 160]
        
        def weight_sched_train(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(base, optim_factor)
        
        hpo_sched = (np.array(sched)/10).astype(int)
        def weight_sched_hpo(epoch):
            optim_factor = 0
            for i in range(len(hpo_sched)):
                if epoch > hpo_sched[len(hpo_sched) - 1 - i]:
                    optim_factor = len(hpo_sched) - i
                    break

            return math.pow(base, optim_factor)

    elif arch_default == 'unet2d' or arch_default == 'unet2d_small':
        lr, arch_lr = 2e-4, 1e-4
        epochs_default, retrain_epochs = 60, 100
        # epochs_default, retrain_epochs = 1, 10
        retrain_freq = epochs_default
        # opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        opt, arch_opt = torch.optim.AdamW, torch.optim.AdamW
        weight_decay = 0.05
        
        def weight_sched_search(step, warmup=5, decay=(epochs_default+5)):
            if step < warmup:
                return float(step) / float(warmup)
            else:
                current = decay - step
                total = decay - warmup
                return max(current / total, 0)
        
        def weight_sched_train(step, warmup=5, decay=(retrain_epochs+5)):
            if step < warmup:
                return float(step) / float(warmup)
            else:
                current = decay - step
                total = decay - warmup
                return max(current / total, 0)
            
        def weight_sched_hpo(step, warmup=5, decay=(80+5)):
            if step < warmup:
                return float(step) / float(warmup)
            else:
                current = decay - step
                total = decay - warmup
                return max(current / total, 0)
            
    elif args.arch  == 'deepsea':
        # epochs_default, retrain_epochs = 100, 150
        # epochs_default, retrain_epochs = 100, 100
        epochs_default, retrain_epochs = 80, 10
        retrain_freq = epochs_default
        opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        weight_decay = 5e-4 
        
        search_sched = [30, 60, 80]
        def weight_sched_search(epoch):
            optim_factor = 0
            for i in range(len(search_sched)):
                if epoch > search_sched[len(search_sched) - 1 - i]:
                    optim_factor = len(search_sched) - i
                    break

            return math.pow(base, optim_factor)

        if dims == 1:
            sched = [30, 60, 90, 120, 160]
        else:
            sched = [60, 120, 160]
        
        def weight_sched_train(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(base, optim_factor)
        
        hpo_sched = (np.array(sched)/5).astype(int)
        def weight_sched_hpo(epoch):
            optim_factor = 0
            for i in range(len(hpo_sched)):
                if epoch > hpo_sched[len(hpo_sched) - 1 - i]:
                    optim_factor = len(hpo_sched) - i
                    break

            return math.pow(base, optim_factor)

    # arch_opt = ExpGrad

    return dims, sample_shape, num_classes, batch_size, epochs_default, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, weight_sched_hpo, accum, clip, retrain_clip, validation_freq, retrain_freq,\
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs


# def get_metric(dataset):
#     if dataset == "your_new_task": # modify this to experiment with a new task
#         return accuracy, np.max
#     # if len(dataset.split("_")) > 1 and dataset.split("_")[0] in ts_file_dict:
#     #     return nn.MSELoss(), np.min

#     if dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
#         return mcc, np.max 
#     if dataset in ['ISCXVPN2016', 'USTC-TFC2016', 'CICIoT2022', 'ISCXTor2016']:
#         return accuracy, np.max
#     if dataset[:5] == "CIFAR" or dataset[:5] == "MNIST" or dataset == "SATELLITE" or dataset == "SPHERICAL":
#         return accuracy, np.max
#     if dataset == "DEEPSEA":
#         return auroc, np.max
#     if dataset == "DARCY-FLOW-5":
#         return LpLoss(size_average=True), np.min
#     if dataset == 'PSICOV':
#         return psicov_mae(), np.min
#     if dataset[:5] == 'MUSIC':
#         return nn.BCELoss(), np.min
#     if dataset == 'ECG':
#         return ecg_f1, np.max
#     if dataset == 'NINAPRO':
#         return accuracy_onehot, np.max
#     if dataset == 'COSMIC':
#         return fnr, np.min
#     if dataset == 'FSD':
#         return map_value, np.max
#     if dataset == 'big_earth_net':
#         return mean_average_precision1, np.max
#     if dataset == 'brick_kiln':
#         return accuracy, np.max
#     if dataset == 'eurosat':
#         return accuracy, np.max
#     if dataset == 'so2sat':
#         return accuracy, np.max
#     if dataset == 'BigEarth':
#         return mean_average_precision2, np.max
#     if dataset == 'forestnet':
#         return accuracy, np.max
#     if dataset == 'pv4ger':
#         return accuracy, np.max
#     if dataset == 'canadian_cropland':
#         return accuracy, np.max
#     if dataset == 'fmow':
#         return accuracy, np.max


def get_hp_configs(dataset, n_train, arch = "wrn"):
    ratio = 1.0
    if n_train < 50:
        subsamping_ratio = ratio
        # subsamping_ratio = 0.01
        # subsamping_ratio = 0.2
    elif n_train < 100:
        subsamping_ratio = ratio
        # subsamping_ratio = 0.01
        # subsamping_ratio = 0.1
    elif n_train < 500:
        subsamping_ratio = ratio
        # subsamping_ratio = 0.01
        # subsamping_ratio = 0.05
    else:
        subsamping_ratio = ratio
        # subsamping_ratio = 0.01
        # subsamping_ratio = 0.01

    
    if arch == "unet2d":
        # epochs = 80
        epochs = 20
        momentum = [0.9]
        lrs = 0.1 ** np.arange(2, 5)

        dropout_rates = [0, 0.05]
        wd = [1e-3, 1e-2]
    else:
        # epochs = 80
        # epochs = 100
        epochs = 20
        # lrs = 0.1 ** np.arange(2, 3)
        # dropout_rates = [0]
        # wd = [5e-4]
        # momentum = [0.9]
        lrs = 0.1 ** np.arange(1, 4)
        dropout_rates = [0, 0.05]
        wd = [5e-4, 5e-6]
        momentum = [0.9, 0.99]

    if dataset in ['PSICOV', 'COSMIC', 'FSD']: # 2D dense
        lrs = 0.1 ** np.arange(2, 5)
        wd = [5e-5]
        dropout_rates = [0]
        momentum = [0.99]
    configs = list(product(lrs, dropout_rates, wd, momentum))

    return configs, epochs, subsamping_ratio


def get_optimizer(type='SGD', momentum=0.9, weight_decay=5e-4):
    if type == 'AdamW':
        return partial(torch.optim.AdamW, weight_decay=weight_decay)
    return partial(torch.optim.SGD, momentum=momentum, weight_decay=weight_decay, nesterov=(momentum!=0))

