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
from networks.vq import Encoder

# import data loaders, task-specific losses and metrics
# from data_loaders import load_cifar, load_mnist, load_deepsea, load_darcy_flow, load_psicov, load_music, load_ecg, load_satellite, load_ninapro, load_cosmic, load_spherical, load_fsd 
# from task_utils import FocalLoss, LpLoss
# from task_utils import mask, accuracy, accuracy_onehot, auroc, psicov_mae, ecg_f1, fnr, map_value
# from task_utils import mcc
# from sklearn import metrics
import sys 
sys.path.append('./')
from src.data_loaders import load_deepsea_full, load_genomic_benchmarks, load_nucleotide_transformer # 
# import customized optimizers
# from optimizers import ExpGrad

def get_data(dataset, batch_size, arch, valid_split, one_hot=True):
    data_kwargs = None
    root = "./src/datasets"
    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None
        
    elif dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]: 
        train_loader, val_loader, test_loader = load_genomic_benchmarks(root, batch_size, one_hot = one_hot, valid_split=valid_split, dataset_name = dataset, quantize=False,rc_aug=False, shift_aug=False)
    elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
        train_loader, val_loader, test_loader = load_nucleotide_transformer(root, batch_size, one_hot = one_hot, valid_split=valid_split, dataset_name = dataset, quantize=False,rc_aug=False, shift_aug=False)
    elif dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(root, batch_size,one_hot = one_hot, valid_split=valid_split,quantize=False,rc_aug=False, shift_aug=False)
    elif dataset == "DEEPSEA_FULL":
        train_loader, val_loader, test_loader = load_deepsea_full(root, batch_size, one_hot = one_hot,valid_split=valid_split,quantize=False,rc_aug=False, shift_aug=False)

    # elif dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]: 
    #     train_loader, val_loader, test_loader = load_genomic_benchmarks(batch_size, one_hot = True, valid_split=valid_split, dataset_name = dataset)
    #     # train_loader, val_loader, test_loader = load_genomic_benchmarks(batch_size, one_hot = False, valid_split=valid_split, dataset_name = dataset)
    # elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
    #     train_loader, val_loader, test_loader = load_nucleotide_transformer(batch_size, one_hot = True, valid_split=valid_split, dataset_name = dataset)
    # elif dataset == "DEEPSEA":
    #     train_loader, val_loader, test_loader = load_deepsea(batch_size, valid_split=valid_split)
    # elif dataset == "DEEPSEA_FULL":
    #     train_loader, val_loader, test_loader = load_deepsea_full(batch_size=256, one_hot = True, valid_split=-1)


    # elif dataset == "CIFAR10":
    #     train_loader, val_loader, test_loader = load_cifar(10, batch_size, valid_split=valid_split)
    # elif dataset == "CIFAR10-PERM":
    #     train_loader, val_loader, test_loader = load_cifar(10, batch_size, permute=True, valid_split=valid_split)
    # elif dataset == "CIFAR100":
    #     train_loader, val_loader, test_loader = load_cifar(100, batch_size, valid_split=valid_split)
    # elif dataset == "CIFAR100-PERM":
    #     train_loader, val_loader, test_loader = load_cifar(100, batch_size, permute=True, valid_split=valid_split)
    # elif dataset == "MNIST":
    #     train_loader, val_loader, test_loader = load_mnist(batch_size, valid_split=valid_split)
    # elif dataset == "MNIST-PERM":
    #     train_loader, val_loader, test_loader = load_mnist(batch_size, permute=True, valid_split=valid_split)
    # elif dataset == "SPHERICAL":
    #     train_loader, val_loader, test_loader = load_spherical(batch_size, valid_split=valid_split)
    
    # elif dataset == "DARCY-FLOW-5":
    #     train_loader, val_loader, test_loader, y_normalizer = load_darcy_flow(batch_size, sub = 5, arch = arch, valid_split=valid_split)
    #     data_kwargs = {"decoder": y_normalizer}
    # elif dataset == 'PSICOV':
    #     train_loader, val_loader, test_loader, _, _ = load_psicov(batch_size, valid_split=valid_split)
    # elif dataset[:5] == 'MUSIC':
    #     if dataset[6] == 'J': length = 255
    #     elif dataset[6] == 'N': length = 513
    #     train_loader, val_loader, test_loader = load_music(batch_size, dataset[6:], length=length, valid_split=valid_split)
    # elif dataset == "ECG":
    #     train_loader, val_loader, test_loader = load_ecg(batch_size, valid_split=valid_split)
    # elif dataset == "SATELLITE":
    #     train_loader, val_loader, test_loader = load_satellite(batch_size, valid_split=valid_split)
    # elif dataset == "NINAPRO":
    #     train_loader, val_loader, test_loader = load_ninapro(batch_size, arch, valid_split=valid_split)
    # elif dataset == "COSMIC":
    #     valid_split = True
    #     train_loader, val_loader, test_loader = load_cosmic(batch_size, valid_split=valid_split)
    #     data_kwargs = {'transform': mask}
    # elif dataset == "FSD":
    #     train_loader, val_loader, test_loader = load_fsd(batch_size, valid_split=valid_split)

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test
    print('n_val',n_val)
    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_model(arch, sample_shape, num_classes, config_kwargs, ks = None, ds = None, dropout = None):
    in_channel = sample_shape[1]
    activation = config_kwargs['activation']
    remain_shape = config_kwargs['remain_shape']
    if dropout is None:
        dropout = config_kwargs['dropout']
    pool_k = config_kwargs['pool_k']
    squeeze = config_kwargs['squeeze']

    if len(sample_shape) == 4:

        if arch == 'your_new_arch': # modify this to experiment with a new architecture
            model = None
        elif 'unet' in arch:
            model = None
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
        mid_channels = min(4 ** (num_classes // 10 + 1), 64)

        if arch == 'your_new_arch': # modify this to experiment with a new architecture
            model = None
        elif arch == 'TCN':
            model = TCN(in_channel, num_classes, [100] * 8, kernel_size=7, dropout=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif arch == 'wrn':
            model = ResNet1D(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif arch == 'deepsea':
            model = DeepSEA(ks = ks, ds = ds)
        elif arch == 'unet':
            embed_dim = 1024
            output_shape = 2
            print('135',sample_shape[-1]) 
            model = Encoder(in_channels=embed_dim, f_channel=sample_shape[-1], num_class=output_shape, ks = ks, ds = ds)
   
    return model


def get_config(dataset):
    einsum = True
    base, accum = 0.2, 1
    validation_freq = 1
    clip, retrain_clip = 1, -1
    quick_search, quick_retrain = 0.2, 1
    config_kwargs = {'temp': 1, 'arch_retrain_default': None, 'grad_scale': 100, 'activation': None, 'remain_shape': False, 'pool_k': 8, 'squeeze': False, 'dropout': 0}
    
    if dataset == "your_new_task": # modify this to experiment with a new task
        dims, sample_shape, num_classes = None, None, None
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        loss = None

        batch_size = 64
        arch_default = 'wrn'

    elif dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]:  
        # kernel_choices_default, dilation_choices_default = [3, 5, 7, 9, 11], [1, 3, 5, 7]
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9, 11], [1, 3, 5, 7]
        loss = nn.CrossEntropyLoss()
        
        batch_size = 256 # 64
        arch_default = 'wrn'
        # arch_default = 'unet'
        
        # config_kwargs['activation'] = 'softmax'
        # config_kwargs['grad_scale'] = 10

        if dataset == "dummy_mouse_enhancers_ensembl":
            # dims, sample_shape, num_classes = 1, (1, 9, 4710), 2
            dims, sample_shape, num_classes = 1, (1, 5, 4707), 2
            # dims, sample_shape, num_classes = 1, (1, 1, 4707), 2
        elif dataset == "demo_coding_vs_intergenomic_seqs":
            # dims, sample_shape, num_classes = 1, (1, 7, 202), 2
            dims, sample_shape, num_classes = 1, (1, 5, 200), 2
            # dims, sample_shape, num_classes = 1, (1, 1, 200), 2
        elif dataset == "demo_human_or_worm":
            # dims, sample_shape, num_classes = 1, (1, 8, 202), 2
            dims, sample_shape, num_classes = 1, (1, 5, 200), 2
            # dims, sample_shape, num_classes = 1, (1, 1, 200), 2
        elif dataset == "human_enhancers_cohn":
            # dims, sample_shape, num_classes = 1, (1, 7, 502), 2
            dims, sample_shape, num_classes = 1, (1, 5, 500), 2
            # dims, sample_shape, num_classes = 1, (1, 1, 500), 2
        elif dataset == "human_enhancers_ensembl":
            # dims, sample_shape, num_classes = 1, (1, 9, 576), 2 
            dims, sample_shape, num_classes = 1, (1, 5, 573), 2
            # dims, sample_shape, num_classes = 1, (1, 1, 573), 2
        elif dataset == "human_ensembl_regulatory":
            # dims, sample_shape, num_classes = 1, (1, 9, 805), 3
            dims, sample_shape, num_classes = 1, (1, 5, 802), 3
            # dims, sample_shape, num_classes = 1, (1, 1, 802), 3
        elif dataset == "human_nontata_promoters":
            # dims, sample_shape, num_classes = 1, (1, 7, 253), 1
            dims, sample_shape, num_classes = 1, (1, 5, 251), 2
            # dims, sample_shape, num_classes = 1, (1, 1, 251), 2
        elif dataset == "human_ocr_ensembl":
            # dims, sample_shape, num_classes = 1, (1, 9, 596), 2
            dims, sample_shape, num_classes = 1, (1, 5, 593), 2
            # dims, sample_shape, num_classes = 1, (1, 1, 593), 2

    elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9, 11], [1, 3, 5, 7]
        loss = nn.CrossEntropyLoss()
        
        batch_size = 256 # 64
        arch_default = 'wrn'
        # arch_default = 'unet'
        if dataset == "enhancer":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 2
        elif dataset == "enhancer_types":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 3
        elif dataset in ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac']:
            dims, sample_shape, num_classes = 1, (1, 5, 500), 2
        elif dataset in ['promoter_all', 'promoter_no_tata', 'promoter_tata']:
            dims, sample_shape, num_classes = 1, (1, 5, 300), 2
        elif dataset in ['splice_sites_acceptors', 'splice_sites_donors']:
            dims, sample_shape, num_classes = 1, (1, 5, 600), 2
        elif dataset == 'splice_sites_all':
            dims, sample_shape, num_classes = 1, (1, 5, 400), 3

    elif dataset == "DEEPSEA":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 36
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))

        batch_size = 256 # 256 
        arch_default = 'wrn'  
        # config_kwargs['grad_scale'] = 10 

    elif dataset == "DEEPSEA_FULL":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 919
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((919, )))

        batch_size = 256 # 256 
        arch_default = 'wrn'  
        config_kwargs['grad_scale'] = 10

    # elif dataset[:5] == "CIFAR":
    #     dims, sample_shape, num_classes = 2, (1, 3, 32, 32), 10 if dataset in ['CIFAR10', 'CIFAR10-PERM'] else 100
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
    #     loss = nn.CrossEntropyLoss()

    #     batch_size = 64
    #     arch_default = 'wrn-16-1' 
    #     config_kwargs['arch_retrain_default'] = 'wrn-16-4' 
    #     config_kwargs['grad_scale'] = 5000

    # elif dataset == 'SPHERICAL':
    #     dims, sample_shape, num_classes = 2, (1, 3, 60, 60), 100
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
    #     loss = nn.CrossEntropyLoss() 

    #     batch_size = 64
    #     arch_default = 'wrn-16-1'
    #     config_kwargs['arch_retrain_default'] = 'wrn-16-4' 
    #     config_kwargs['grad_scale'] = 500


    # elif dataset == "DARCY-FLOW-5":
    #     dims, sample_shape, num_classes = 2, (1, 3, 85, 85), 1
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]  
    #     loss = LpLoss(size_average=False)
        
    #     batch_size = 10
    #     arch_default = 'wrn-16-1'
    #     config_kwargs['arch_retrain_default'] = 'wrn-16-4'
    #     config_kwargs['remain_shape'] = config_kwargs['squeeze'] = True 
        

    # elif dataset == "PSICOV":
    #     dims, sample_shape, num_classes = 2, (1, 57, 128, 128), 1
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]  
    #     loss = nn.MSELoss(reduction='mean')

    #     batch_size = 8
    #     arch_default = 'wrn-16-1'
    #     config_kwargs['arch_retrain_default'] = 'wrn-16-4'
    #     config_kwargs['remain_shape']  = True


    # elif dataset == "NINAPRO": 
    #     dims, sample_shape, num_classes = 2, (1, 1, 16, 52), 18
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7], [1, 3, 7]
    #     loss = FocalLoss(alpha=1)

    #     batch_size = 128
    #     arch_default = 'wrn-16-1'
    #     config_kwargs['arch_retrain_default'] = 'wrn-16-4'
    #     config_kwargs['pool_k'] = 0


    # elif dataset == "COSMIC":
    #     dims, sample_shape, num_classes = 2, (1, 1, 128, 128), 1
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
    #     loss = nn.BCELoss()

    #     batch_size = 4
    #     arch_default = 'wrn-16-1'
    #     config_kwargs['arch_retrain_default'] = 'wrn-16-4'
    #     config_kwargs['activation'] = 'sigmoid'
    #     config_kwargs['remain_shape'] = True
    #     config_kwargs['grad_scale'] = 1000


    # elif dataset == 'FSD':
    #     dims, sample_shape, num_classes = 2, (1, 1, 96, 102), 200
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
    #     loss = nn.BCEWithLogitsLoss(pos_weight=10 * torch.ones((200, )))
        
    #     batch_size = 128
    #     arch_default = 'wrn-16-1'
    #     config_kwargs['arch_retrain_default'] = 'wrn-16-4'
    #     config_kwargs['pool_k'] = 0


    # elif dataset[:5] == "MNIST":
    #     dims, sample_shape, num_classes = 1, (1, 1, 784), 10
    #     kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15, 31, 63, 127] 
    #     loss = F.nll_loss

    #     batch_size = 256      
    #     arch_default = 'wrn'
    #     config_kwargs['activation'] = 'softmax'


    # elif dataset[:5] == "MUSIC":
    #     if dataset[6] == 'J': length = 255 
    #     elif dataset[6] == 'N': length = 513
    #     dims, sample_shape, num_classes = 1, (1, 88, length - 1), 88
    #     kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
    #     loss = nn.BCELoss()

    #     batch_size = 4
    #     arch_default = 'wrn'
    #     config_kwargs['activation'] = 'sigmoid'
    #     config_kwargs['remain_shape'] = True

    
    # elif dataset == "ECG": 
    #     dims, sample_shape, num_classes = 1, (1, 1, 1000), 4
    #     kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
    #     loss = nn.CrossEntropyLoss()

    #     batch_size = 1024
    #     arch_default = 'wrn'
    #     config_kwargs['activation'] = 'softmax'
        

    # elif dataset == "SATELLITE":
    #     dims, sample_shape, num_classes = 1, (1, 1, 46), 24
    #     kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
    #     loss = nn.CrossEntropyLoss()

    #     batch_size = 256
    #     arch_default = 'wrn'


    

    lr, arch_lr = (1e-2, 5e-3) if config_kwargs['remain_shape'] else (0.1, 0.05) 

    if arch_default == 'wrn':
        epochs_default, retrain_epochs = 60, 10
        retrain_freq = epochs_default
        opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        weight_decay = 5e-4 
        
        sched = [60, 120, 160]
        def weight_sched_search(epoch):
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
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
    
    elif arch_default == 'unet':
        epochs_default, retrain_epochs = 10, 10
        retrain_freq = epochs_default
        # opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        opt, arch_opt = torch.optim.AdamW, torch.optim.AdamW
        weight_decay = 5e-4 
        
        sched = [60, 120, 160]
        def weight_sched_search(epoch):
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
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
        
        

    # arch_opt = ExpGrad

    return dims, sample_shape, num_classes, batch_size, epochs_default, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq,\
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs


# def get_metric(dataset):
#     if dataset == "your_new_task": # modify this to experiment with a new task
#         return accuracy, np.max
#     if dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]:  
#         return accuracy, np.max
#     if dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac']: 
#         return mcc, np.max 
#     if dataset in ['promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors', 'splice_sites_all']: 
#         return mcc, np.max
#     if dataset == "DEEPSEA":
#         return auroc, np.max
#     if dataset == "DEEPSEA_FULL":
#         return auroc, np.max
#     # if dataset[:5] == "CIFAR" or dataset[:5] == "MNIST" or dataset == "SATELLITE" or dataset == "SPHERICAL":
#     #     return accuracy, np.max
#     # if dataset == "DARCY-FLOW-5":
#     #     return LpLoss(size_average=True), np.min
#     # if dataset == 'PSICOV':
#     #     return psicov_mae(), np.min
#     # if dataset[:5] == 'MUSIC':
#     #     return nn.BCELoss(), np.min
#     # if dataset == 'ECG':
#     #     return ecg_f1, np.max
#     # if dataset == 'NINAPRO':
#     #     return accuracy_onehot, np.max
#     # if dataset == 'COSMIC':
#     #     return fnr, np.min
#     # if dataset == 'FSD':
#     #     return map_value, np.max


def get_hp_configs(dataset, n_train):

    epochs = 40
    if n_train < 50:
        subsamping_ratio = 0.2
    elif n_train < 100:
        subsamping_ratio = 0.1
    elif n_train < 500:
        subsamping_ratio = 0.05
    else:
        subsamping_ratio = 0.01

    lrs = 0.1 ** np.arange(1, 3)
    # lrs = 0.1 ** np.arange(1, 5)

    dropout_rates = [0.05]
    wd = [5e-4]
    momentum = [0.9, 0.99]

    if dataset in ['PSICOV', 'COSMIC', 'FSD']: # 2D dense
        lrs = 0.1 ** np.arange(2, 5)
        wd = [5e-5]
        dropout_rates = [0]
        momentum = [0.99]
    configs = list(product(lrs, dropout_rates, wd, momentum))

    return configs, epochs, subsamping_ratio


def get_optimizer(type='SGD', momentum=0.9, weight_decay=5e-4):
    
    return partial(torch.optim.SGD, momentum=momentum, weight_decay=weight_decay, nesterov=(momentum!=0))

